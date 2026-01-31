import os
import jax
from typing import Any
from jax import lax
import jax.numpy as jnp
import orbax
import orbax.checkpoint as ocp
import flax
from flax.training import train_state
from flax import jax_utils
from flax import traverse_util
import torch
from torch.utils.data import DataLoader
from network import PiMAE
from utils_data import dataset_3d, get_sample, dataset_3d_infer
from utils_imageJ import save_tiff_imagej_compatible
from utils_metrics import ms_ssim_3d
from utils_huggingface_jax import load_jax_vit_base_model
import optax
import pickle
import numpy as np
import random
import tqdm
import functools
import glob
from skimage.io import imread
from skimage.transform import resize
from utils_eval import eval_nrmse
from utils_eval import percentile_norm
from utils_data import min_max_norm
import pprint
import pdb
from jax.experimental import enable_x64
import copy
import gc
from orbax.checkpoint import tree as ocp_tree 
class TrainState(train_state.TrainState):
  batch_stats: Any


def rec_loss(x, rec, mask=None):
    if mask is None:
        mask = jnp.ones_like(x)
    l1_loss = jnp.abs((rec - x))
    l1_loss = (l1_loss * mask).sum() / mask.sum()
    x_norm = (x - x.min()) / (x.max() - x.min())
    rec_norm = (rec - x.min()) / (x.max() - x.min())
    ms_ssim_loss = jnp.mean(1 - ms_ssim_3d(x_norm, rec_norm, win_size=5))
    loss = 0.875 * l1_loss + 0.125 * ms_ssim_loss
    return loss



def TV_Loss(img):
    img = img.reshape([-1, img.shape[-3], img.shape[-2], img.shape[-1]])
    img = img / img.mean()
    batch_size = img.shape[0]
    z, y, x = img.shape[1:4]
    
    def _tensor_size(t):
        return t.shape[-3]*t.shape[-2]*t.shape[-1]
    
    cz = _tensor_size(img[:, 1:, :, :])
    cy = _tensor_size(img[:, :, 1:, :])
    cx = _tensor_size(img[:, :, :, 1:])
    
    hz = lax.pow(jnp.abs(img[:, 1:, :, :] - img[:, :z-1, :, :]), 2.).sum()
    hy = lax.pow(jnp.abs(img[:, :, 1:, :] - img[:, :, :y-1, :]), 2.).sum()
    hx = lax.pow(jnp.abs(img[:, :, :, 1:] - img[:, :, :, :x-1]), 2.).sum()
    
    if cz == 0:
        return (hy/cy + hx/cx) / batch_size
    else:
        return (hz/cz + hy/cy + hx/cx) / batch_size



def center_loss(img):
    if img.shape[-3] == 1:
        vz = jnp.array([0])
    else:
        vz = jnp.linspace(-1, 1, img.shape[-3])
    vy = jnp.linspace(-1, 1, img.shape[-2])
    vx = jnp.linspace(-1, 1, img.shape[-1])
    grid_z, grid_y, grid_x = jnp.meshgrid(vz, vy, vx, indexing='ij')
    grid_z = grid_z[jnp.newaxis, jnp.newaxis, ...]
    grid_y = grid_y[jnp.newaxis, jnp.newaxis, ...]
    grid_x = grid_x[jnp.newaxis, jnp.newaxis, ...]

    img_z = (grid_z * img).sum(axis=(-3, -2, -1)) / img.sum(axis=(-3, -2, -1))
    img_y = (grid_y * img).sum(axis=(-3, -2, -1)) / img.sum(axis=(-3, -2, -1))
    img_x = (grid_x * img).sum(axis=(-3, -2, -1)) / img.sum(axis=(-3, -2, -1))
    img_c = jnp.sqrt(img_z**2 + img_y**2 + img_x**2)
    return img_c.mean()




def compute_metrics(x, state, params, args, rng, train=True):
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    
    result, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, x, args, train, rngs={'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, mutable=['batch_stats'])
    
    if train and args.mask_ratio > 0.3:
        rec = rec_loss(result["x_real"], result["rec_real"], result["mask"]).mean()
    else:
        rec = rec_loss(result["x_real"], result["rec_real"]).mean()
    psf_tv = TV_Loss(result["psf"]).mean()
    lp_tv = TV_Loss(result["light_pattern"]).mean()
    ct_loss = center_loss(result["psf"]).mean()
    deconv_tv = TV_Loss(result["deconv"]).mean()

    loss = rec + args.tv_loss * psf_tv + args.psfc_loss * ct_loss + args.lp_tv * lp_tv
    return {"loss": loss, "rec_loss": rec, "psf_tv_loss": psf_tv, "lp_tv_loss": lp_tv, "psf_center_loss": ct_loss, "deconv_tv": deconv_tv}, result, updates


def worker_init_fn(_):
    seed = torch.utils.data.get_worker_info().seed
    np.random.seed(seed % 2**32)
    random.seed(seed)


def pipeline(args, writer):
    # data
    patch_size_z = args.patch_size[0] if len(args.patch_size) > 2 else 1
    train_set = dataset_3d(args.trainset, args.crop_size, minimum_number=args.min_datasize, 
                          patch_size_z=patch_size_z, adapt_pattern_dimension=args.adapt_pattern_dimension, 
                          target_pattern_frames=args.target_pattern_frames, random_pattern_sampling=args.random_pattern_sampling)
    trainloader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True, worker_init_fn=worker_init_fn)

    test_set = dataset_3d(args.testset, args.crop_size, use_gt=args.use_gt, minimum_number=args.batchsize*2, 
                         sampling_rate=args.sampling_rate, patch_size_z=patch_size_z, 
                         adapt_pattern_dimension=args.adapt_pattern_dimension, target_pattern_frames=args.target_pattern_frames, random_pattern_sampling=args.random_pattern_sampling)
    testloader = DataLoader(test_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True, worker_init_fn=worker_init_fn)

    x_exp = get_sample(trainloader)
    print("Tensor shape:", x_exp.shape)

    # Select training devices (allow reserving extra GPU for MC logging)
    all_devices = jax.devices()
    train_devices = all_devices if getattr(args, "train_num_devices", None) is None else all_devices[:args.train_num_devices]
    num_train_devices = len(train_devices)
    if num_train_devices == 0:
        raise RuntimeError("No JAX devices available for training")
    if args.batchsize % num_train_devices != 0:
        raise ValueError(f"batchsize={args.batchsize} must be divisible by train_num_devices={num_train_devices}")

    def net_model():
        image_size = [x_exp.shape[2], args.crop_size[0] * args.rescale[0], args.crop_size[1] * args.rescale[1]]
        return PiMAE(image_size, args.patch_size, args.psf_size, args.lrc)
    
    # init
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # train step
    @functools.partial(jax.pmap, axis_name='Parallelism', devices=train_devices)
    def apply_model(state, x, rng):
        def loss_fn(params):
            metrics, res, updates = compute_metrics(x, state, params, args, rng, train=True)
            return metrics['loss'], (metrics, res, updates)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)
        metrics, res, updates = aux
        grads = jax.lax.pmean(grads, 'Parallelism')
        loss = jax.lax.pmean(loss, 'Parallelism')
        batch_stats = jax.lax.pmean(updates["batch_stats"], "Parallelism")
        new_state = state.apply_gradients(grads=grads, batch_stats=batch_stats)
        return metrics, res, new_state
    
    @functools.partial(jax.pmap, axis_name='Parallelism', devices=train_devices)
    def infer_model(state, x, rng):
        metrics, res, updates = compute_metrics(x, state, state.params, args, rng, train=False)
        return metrics, res, state
    
    warmup_steps = 7 * len(trainloader)
    lr_schedule = optax.linear_schedule(init_value=0, end_value=args.lr, transition_steps=warmup_steps)
    # warmup_steps = 0.1 * args.epoch * len(trainloader)
    # decay_steps = (args.epoch - 0.1 * args.epoch) * len(trainloader)
    # lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0, peak_value=args.lr, warmup_steps=warmup_steps, decay_steps=decay_steps, end_value=0.0)
    
    # opt = optax.chain(optax.adam(args.lr), 
    #                   optax.contrib.reduce_on_plateau(factor=0.5, patience=10, rtol=0.0001, atol=0.0, cooldown=0, accumulation_size=100))

    #################### create_train_state ####################
    @jax.jit
    def create_train_state(rng):
        net = net_model()
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        x_init = get_sample(trainloader)
        variables = net.init({"params": rng1, 'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, x_init, args, True)
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule),
            )
        
        if args.accumulation_step is not None:
            tx = optax.MultiSteps(tx, every_k_schedule=args.accumulation_step)

        return TrainState.create(
            apply_fn=net.apply, params=variables['params'], batch_stats=variables['batch_stats'], tx=tx)
    
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng)
    variables = state.params
    start_epoch = 0

    #################### Resume ####################
    # checkpoint
    options = ocp.CheckpointManagerOptions(
        max_to_keep=5,
        best_fn=lambda metrics: -metrics['test_rec_loss'] if 'test_rec_loss' in metrics else 0.0,  # negative because we want minimum
        keep_time_interval=None,
        keep_period=None
    )
    
    # Initialize best loss tracking
    best_test_rec_loss = float('inf')
    
    if args.resume_pretrain and not args.resume:
        with open("./ckpt/pretrain_params.pkl", 'rb') as f: 
            pretrain_params = pickle.load(f)

        N = {"Loaded": 0, "Excluded": 0, "Unloaded": 0}
        for key, value in pretrain_params.items():
            # Skip patch embedding and position embedding
            if key in ['patch_embed', 'pos_embed'] or 'patch_embed' in key or 'pos_embed' in key:
                N["Excluded"] += 1
                print(f"Excluded {key} from loading (patch/pos embedding)")
                continue
                
            if key in variables['pt_predictor']['MAE'].keys():
                variables['pt_predictor']['MAE'][key] = value
                N["Loaded"] += 1
            else:
                N["Unloaded"] += 1
        print("Resume pretrain loading summary:", N)
        print(f"\033[93mNote: patch_embed and pos_embed were excluded and will be re-initialized\033[0m")
        state = state.replace(params=variables)
    
    if args.resume_pickle is not None and not args.resume:
        print(f"\033[94mLoading BioSR pretrained checkpoint from: {args.resume_pickle}\033[0m")
        with open(args.resume_pickle, 'rb') as f:
            biosr_pretrain = pickle.load(f)
        
        # BioSR pretrained checkpoints have 'params' and optionally 'batch_stats'
        if 'params' not in biosr_pretrain:
            raise ValueError(f"Invalid BioSR pretrained checkpoint format. Expected 'params' key, got: {list(biosr_pretrain.keys())}")
        
        # Load params
        state = state.replace(params=biosr_pretrain['params'])
        
        # Load batch_stats if available
        if 'batch_stats' in biosr_pretrain:
            state = state.replace(batch_stats=biosr_pretrain['batch_stats'])
        
        print("\033[92mSuccessfully loaded pretrained model\033[0m")
    
    if args.resume_s1_path is not None:
        checkpoint_dir = os.path.abspath(os.path.join(args.resume_s1_path, "state"))
        
        if args.not_resume_s1_opt:
            with ocp.CheckpointManager(
                checkpoint_dir,
                options=options,
            ) as mngr:
                if args.resume_s1_iter is not None:
                    step = args.resume_s1_iter
                else:
                    step = mngr.latest_step()
                resume_state = mngr.restore(step, args=ocp.args.StandardRestore())
            state = state.replace(params=resume_state['params'])
        else:
            with ocp.CheckpointManager(
                checkpoint_dir,
                options=options,
            ) as mngr:
                if args.resume_s1_iter is not None:
                    step = args.resume_s1_iter
                else:
                    step = mngr.latest_step()
                state = mngr.restore(step, args=ocp.args.StandardRestore())
        
        state = state.replace(step=0)
        print("\033[94m" + args.resume_s1_path + " %d"%step + "\033[0m")
    
    if args.resume:
        checkpoint_dir = os.path.abspath(os.path.join(args.save_dir, "state"))
        if os.path.exists(checkpoint_dir):
            with ocp.CheckpointManager(
                checkpoint_dir,
                options=options,
            ) as mngr:
                step = mngr.latest_step()
                state = mngr.restore(step, args=ocp.args.StandardRestore())
            start_epoch = step // len(trainloader)
            print("\033[34m", "Resume from", checkpoint_dir, "at epoch", start_epoch, "\033[0m")
    
    #################### Training ####################
    state = jax_utils.replicate(state, devices=train_devices)
    
    for epoch in range(start_epoch, args.epoch):
        # train
        def add_image_writer(name, img, df='CNHW'):
            if img.max() > img.min():
                img = percentile_norm(img)
            writer.add_image(name, np.array(img), epoch, dataformats=df)
        
        pbar = tqdm.tqdm(trainloader)
        metrics_train = {'loss': 0.0, "rec_loss": 0.0, "psf_tv_loss": 0.0, "lp_tv_loss": 0.0, "psf_center_loss": 0.0, "deconv_tv": 0.0}
        
        for data in pbar:
            rng_new, rng = jax.random.split(rng, 2)
            x = jnp.array(data['img'])
            x = jax.lax.stop_gradient(x)
            x = x.reshape([num_train_devices, -1, *x.shape[1:]])
            # jax.profiler.start_trace('/tmp/timeline')
            metrics, res, state = apply_model(state, x, jax_utils.replicate(rng_new, devices=train_devices))
            current_lr = np.array(lr_schedule(state.step)).mean()
            # jax.profiler.stop_trace()
            pbar.set_postfix(rec="%.2e"%np.array(metrics['loss']).mean(), 
                             R="%.2f%%"%np.array(100*res['rec_real'].mean()/res['x_real'].mean()), 
                             lr="%.2e"%current_lr.mean(), 
                             mask="%.2f"%np.array(res['mask']).mean(),
                             lp_tv="%.2e"%np.array(metrics['lp_tv_loss']).mean())
            metrics_train = {k: v + metrics[k].mean() for k, v in metrics_train.items()}      
        metrics_train = {k:np.asarray(v / len(trainloader)) for k, v in metrics_train.items()}
        
        # tensorboard 
        res = jax_utils.unreplicate(res)
        writer.add_scalar('train/loss', metrics_train['loss'], epoch+1)
        writer.add_scalar('train/rec_loss', metrics_train['rec_loss'], epoch+1)
        writer.add_scalar('train/psf_tv_loss', metrics_train['psf_tv_loss'], epoch+1)
        writer.add_scalar('train/lp_tv_loss', metrics_train['lp_tv_loss'], epoch+1)
        writer.add_scalar('train/psf_center_loss', metrics_train['psf_center_loss'], epoch+1)
        writer.add_scalar('train/deconv_tv', metrics_train['deconv_tv'], epoch+1)
        writer.add_scalar('train/lr', current_lr, epoch+1)

        add_image_writer('train/1_x', res['x_up'][0][0:1], 'CNHW')
        add_image_writer('train/2_deconv', res['deconv'][0][0:1], 'CNHW')
        add_image_writer('train/3_psf', res['psf'][0], 'CNHW')
        add_image_writer('train/4_reconstruction', res['rec_up'][0][0:1], 'CNHW')
        add_image_writer('train/5_light_pattern', res['light_pattern'][0][0:1], 'CNHW')
        add_image_writer('train/6_background', res['background'][0][0:1], 'CNHW')
        
        # save
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "psf.tif"), np.array(res['psf'][0][0]), "ZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_x.tif"), np.array(res['x_up']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_mask_real.tif"), np.array(res['mask']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_rec.tif"), np.array(res['rec_up']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_deconv.tif"), np.array(res['deconv']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_light_pattern.tif"), np.array(res['light_pattern']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_background.tif"), np.array(res['background']), "TCZYX")

        # snapshot - removed from here, will be done after test evaluation
        
        # test
        if not args.use_gt:
            metrics_eval = {"rec_loss": 0.0}
            for data in testloader:
                rng_new, rng = jax.random.split(rng, 2)
                x = jnp.array(data['img'])
                x = jax.lax.stop_gradient(x)
                x = x.reshape([num_train_devices, -1, *x.shape[1:]])
                metrics, res, _ = infer_model(state, x, jax_utils.replicate(rng_new, devices=train_devices))
                metrics_eval = {k: v + metrics[k].mean().astype(np.float32) for k, v in metrics_eval.items()}      
            metrics_eval = {k: np.asarray(v / len(testloader)) for k, v in metrics_eval.items()}

            writer.add_scalar('test/rec_loss', metrics_eval['rec_loss'], epoch+1)
            print("epoch=%d/%d"%(epoch+1, args.epoch), "rec=%.2e"%np.array(metrics_eval['rec_loss']))
        else:
            folder = os.path.dirname(args.testset)
            metrics_eval = {"rec_loss": 0.0}

            emitter_nrmse_list = []
            lp_nrmse_list = []
            for data in testloader:
                rng_new, rng = jax.random.split(rng, 2)
                x = jnp.array(data['img'])
                x = jax.lax.stop_gradient(x)
                x = x.reshape([num_train_devices, -1, *x.shape[1:]])
                metrics, res, _ = infer_model(state, x, jax_utils.replicate(rng_new, devices=train_devices))
                metrics_eval = {k: v + metrics[k].mean().astype(np.float32) for k, v in metrics_eval.items()}
                
                emitter_gt = data['emitter_gt'].reshape([num_train_devices, -1, *data['emitter_gt'].shape[1:]])
                emitter_nrmse = eval_nrmse(res['deconv'], emitter_gt)
                emitter_nrmse_list.append(emitter_nrmse)
                lp_gt = data['lp_gt'].reshape([num_train_devices, -1, *data['lp_gt'].shape[1:]])
                lp_nrmse = eval_nrmse(res['light_pattern'], lp_gt)
                lp_nrmse_list.append(lp_nrmse)


            metrics_eval = {k: np.asarray(v / len(testloader)) for k, v in metrics_eval.items()}
            writer.add_scalar('test/rec_loss', metrics_eval['rec_loss'], epoch+1)
            writer.add_scalar('test/emitter_loss', np.mean(emitter_nrmse_list), epoch+1)
            writer.add_scalar('test/light_pattern_loss', np.mean(lp_nrmse_list), epoch+1)
            print("epoch=%d/%d"%(epoch+1, args.epoch), "rec=%.2e"%np.array(metrics_eval['rec_loss']), "emitter=%.2e"%np.mean(emitter_nrmse_list), "light_pattern=%.2e"%np.mean(lp_nrmse_list))
            
        # tensorboard 
        res = jax_utils.unreplicate(res)
        add_image_writer('test/1_x', res['x_up'][0][0:1], 'CNHW')
        add_image_writer('test/2_deconv', res['deconv'][0][0:1], 'CNHW')
        add_image_writer('test/4_reconstruction', res['rec_up'][0][0:1], 'CNHW')
        add_image_writer('test/5_light_pattern', res['light_pattern'][0][0:1], 'CNHW')
        add_image_writer('test/7_background', res['background'][0][0:1], 'CNHW')
        if args.use_gt:
            add_image_writer('test/3_deconv_gt', emitter_gt[0, 0, 0:1], 'CNHW')
            add_image_writer('test/6_light_pattern_gt', lp_gt[0, 0, 0:1], 'CNHW')

        # MC Dropout uncertainty logging on first test batch (if enabled)
        devices = jax.devices()
        mc_samples = getattr(args, "mc_dropout_train_samples", 0)
        # 选择用于 MC logging 的设备：优先指定 GPU id，其次最后一张 GPU，否则首个可用 CPU
        mc_devices = []
        try:
            if getattr(args, "mc_device", "cpu") == "gpu":
                gpus = [d for d in devices if d.platform == "gpu"]
                if gpus:
                    preferred_id = getattr(args, "mc_device_id", None)
                    if preferred_id is not None:
                        chosen = [d for d in gpus if d.id == preferred_id]
                        mc_devices = chosen if chosen else [gpus[-1]]
                    else:
                        # 默认使用列表最后一张 GPU，以便避开训练占用的前几张
                        mc_devices = [gpus[-1]]
            if not mc_devices:
                cpu_devs = jax.devices("cpu")
                mc_devices = [cpu_devs[0]] if cpu_devs else []
        except Exception:
            mc_devices = [devices[0]] if devices else []

        if mc_samples > 1 and mc_devices:
            # reuse first batch in testloader
            mc_args = copy.copy(args)
            if getattr(args, "mc_disable_noise", False):
                mc_args.add_noise = 0.0
            mc_args.mask_ratio = getattr(args, "mc_mask_ratio", 0.0)
            # unreplicate state for host-side MC sampling to avoid extra pmap compilation
            state_single = jax_utils.unreplicate(state)
            # move params/batch_stats to the dedicated MC device to avoid fragmenting training GPUs
            # jax.tree_map was removed in JAX v0.6.0; use tree_util.tree_map for compatibility
            state_single = jax.tree_util.tree_map(lambda x: jax.device_put(x, mc_devices[0]), state_single)

            def single_forward(x_np, rng):
                x_j = jnp.array(x_np)
                rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
                res, _ = state_single.apply_fn(
                    {'params': state_single.params, 'batch_stats': state_single.batch_stats},
                    x_j, mc_args, True,
                    rngs={'dropout': rng2, "random_masking": rng3, "drop_path": rng4},
                    mutable=['batch_stats'])  # allow BN stats to update to avoid ModifyScopeVariableError
                return res["deconv"]

            # 顺序采样，避免多卡附加显存；样本量 1
            x_sample = np.array(data['img'][:1])

            deconv_samples = []
            for _ in range(mc_samples):
                rng, sub = jax.random.split(rng)
                with jax.default_device(mc_devices[0]):
                    deconv_samples.append(np.array(single_forward(x_sample, sub)))
            deconv_stack = np.stack(deconv_samples, axis=0)
            deconv_std = deconv_stack.std(axis=0)
            writer.add_image('uncertainty/deconv_std', percentile_norm(deconv_std[0][0:1]), epoch+1, dataformats='CNHW')
            writer.add_histogram('uncertainty/deconv_std_hist', deconv_std.flatten(), epoch+1)
            # release temporary buffers to mitigate fragmentation
            del deconv_samples, deconv_stack, deconv_std, x_sample
            gc.collect()
        
        # save
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_x.tif"), np.array(res['x_up']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_rec.tif"), np.array(res['rec_up']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_deconv.tif"), np.array(res['deconv']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_light_pattern.tif"), np.array(res['light_pattern']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_background.tif"), np.array(res['background']), "TCZYX")
        del res
        # Save checkpoint with test metrics for best checkpoint selection
        os.makedirs(os.path.join(args.save_dir, "state"), exist_ok=True)
        state_to_save = jax_utils.unreplicate(state)
        checkpoint_dir = os.path.abspath(os.path.join(args.save_dir, "state"))
        
        # Prepare checkpoint metrics
        checkpoint_metrics = {
            'test_rec_loss': float(metrics_eval['rec_loss']),
            'epoch': epoch,
            'step': int(state_to_save.step)
        }
        
        # Update best loss tracking
        current_test_rec_loss = float(metrics_eval['rec_loss'])
        if current_test_rec_loss < best_test_rec_loss:
            best_test_rec_loss = current_test_rec_loss
            print(f"\033[92m*** New best checkpoint saved! Test rec loss: {current_test_rec_loss:.2e} ***\033[0m")
        else:
            print(f"Current test rec loss: {current_test_rec_loss:.2e}, Best: {best_test_rec_loss:.2e}")
            
        with ocp.CheckpointManager(
            checkpoint_dir,
            options=options,
        ) as mngr:
            mngr.save(
                state_to_save.step, 
                args=ocp.args.StandardSave(state_to_save),
                metrics=checkpoint_metrics
            )
            mngr.wait_until_finished()
        del state_to_save
            
    return state


def pipeline_supervised(args):
    # data
    train_set = dataset_3d(args.trainset, args.crop_size, minimum_number=args.min_datasize,
                          adapt_pattern_dimension=args.adapt_pattern_dimension, target_pattern_frames=args.target_pattern_frames,
                          random_pattern_sampling=args.random_pattern_sampling)
    trainloader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)

    test_set = dataset_3d(args.testset, args.crop_size, use_gt=args.use_gt, sampling_rate=0.1,
                         adapt_pattern_dimension=args.adapt_pattern_dimension, target_pattern_frames=args.target_pattern_frames,
                         random_pattern_sampling=args.random_pattern_sampling)
    testloader = DataLoader(test_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)

    x_exp = get_sample(trainloader)
    print("Tensor shape:", x_exp.shape)

def pipeline_infer(args, writer=None):
    import os, glob
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from orbax.checkpoint import tree as ocp_tree

    # ------------------------------------------------------------------
    # 1. 和训练时完全一致的网络构造
    # ------------------------------------------------------------------
    def net_model():
        image_size = [
            args.num_p,
            args.crop_size[0] * args.rescale[0],
            args.crop_size[1] * args.rescale[1],
        ]
        return PiMAE(image_size, args.patch_size, args.psf_size, args.lrc)

    # ------------------------------------------------------------------
    # 2. 和训练时结构一致的 create_train_state（关键是 tx/opt_state 结构）
    #    这里只是把 get_sample(trainloader) 换成了一个同形状的 dummy 输入
    # ------------------------------------------------------------------
    def create_train_state(rng):
        net = net_model()

        # 和训练时一样拆 rng
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        # 训练时 get_sample(trainloader) 的作用只是提供一个正确形状的 x_init
        # 这里用全 0 的 dummy 输入替代，形状保持一致即可
        x_init = jnp.zeros((1, 1, args.num_p, *args.crop_size), dtype=jnp.float32)

        variables = net.init(
            {"params": rng1, "dropout": rng2, "random_masking": rng3, "drop_path": rng4},
            x_init,
            args,
            True,
        )

        # ★ tx 的结构要和训练时完全一致
        lr_schedule = optax.linear_schedule(init_value=0, end_value=1e-4, transition_steps=100)
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule),  # 训练时就是这样写的
        )

        # # 如果训练时用了梯度累积，这里也要保持一致
        # if args.accumulation_step is not None:
        #     tx = optax.MultiSteps(tx, every_k_schedule=args.accumulation_step)

        return TrainState.create(
            apply_fn=net.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            tx=tx,
        )

    # ------------------------------------------------------------------
    # 3. 先在当前设备（单卡）上构造一个具有正确结构的 TrainState
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng)

    # ------------------------------------------------------------------
    # 4. 用 Orbax CheckpointManager 从多卡训练的 ckpt 恢复到当前单卡
    # ------------------------------------------------------------------
    print(args.resume_path)
    options = ocp.CheckpointManagerOptions()
    checkpoint_dir = os.path.abspath(os.path.join(args.resume_path, "state"))

    if os.path.exists(checkpoint_dir):
        with ocp.CheckpointManager(
            checkpoint_dir,
            options=options,
        ) as mngr:
            if getattr(args, "resume_s1_iter", None) is not None:
                step = args.resume_s1_iter
            else:
                step = mngr.latest_step()

            # 用抽象的 shape/dtype 结构作为 restore 的 target，
            # 这样 Orbax 会按当前单卡拓扑重建参数，而不是死磕原来的多卡设备 id
            abstract_state = jax.tree.map(
                ocp_tree.to_shape_dtype_struct,
                state,
            )

            # 恢复完整的 TrainState（包含 params / batch_stats / opt_state 等）
            state = mngr.restore(
                step,
                args=ocp.args.StandardRestore(abstract_state),
            )

        print("\033[34m", "Resume from", checkpoint_dir, "\033[0m")
    else:
        raise ValueError("No checkpoint found")

    # ------------------------------------------------------------------
    # 5. 处理待推理的数据文件列表
    # ------------------------------------------------------------------
    if any(ext in args.data_dir for ext in ["*.tif", "*.png", "*.jpg"]):
        file_names = glob.glob(args.data_dir)
        print("Images num:", len(file_names))
    else:
        file_names = [args.data_dir]

    # ------------------------------------------------------------------
    # 6. 推理函数（保持你原来逻辑不变）
    # ------------------------------------------------------------------
    @jax.jit
    def eval_model(x, rng):
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        result, _ = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            x,
            args,
            False,
            rngs={"dropout": rng2, "random_masking": rng3, "drop_path": rng4},
            mutable=["batch_stats"],
        )
        return result

    # ------------------------------------------------------------------
    # 7. 下面继续用 file_names + eval_model 做推理、保存结果
    #    这里可以直接沿用你原来的 infer/pipeline_infer 代码
    # ------------------------------------------------------------------
    # 例如：
    # for fn in file_names:
    #     img = load_your_tif(fn)   # 自己的读图函数
    #     ...
    #     out = eval_model(x, rng_eval)
    #     save_out(...)
    # return ...


    @jax.jit
    def eval_model_mc(x, rng, mc_args, mutable_batch_stats=False):
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        mutable = ['batch_stats'] if mutable_batch_stats else False
        result, _ = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x, mc_args, True, rngs={'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, mutable=mutable)
        return result
    
    file_names.sort()
    # ###########################################
    # file_names = file_names[:3]

    nrmse_list = []
    for i, file in enumerate(file_names):
        if '*.tif' or '*.png' or '*.jpg' in args.data_dir:
            relative_path = os.path.relpath(file, os.path.commonpath([args.data_dir.replace('*', ''), file]))
            relative_path_star = os.path.relpath(args.data_dir, os.path.commonpath([args.data_dir.replace('*', ''), file]))
            star_indices = [i for i, part in enumerate(relative_path_star.split(os.sep)) if "*" in part]
            relative_parts = relative_path.split(os.sep)
            selected_parts = [relative_parts[i] for i in star_indices]
            target_dir = os.path.join(args.save_dir, *selected_parts[:-1])
        else:
            target_dir = args.save_dir

        img = imread(file).astype(np.float32) # CZXY
        if len(img.shape) == 2:
            img = img[np.newaxis, np.newaxis, :, :]
        elif len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]
        print(file, img.shape)
        
        patch_size_z = args.patch_size[0] if len(args.patch_size) > 2 else 1
        test_set = dataset_3d_infer(img, args.crop_size, args.rescale, patch_size_z=patch_size_z,
                                   adapt_pattern_dimension=args.adapt_pattern_dimension, target_pattern_frames=args.target_pattern_frames,
                                   random_pattern_sampling=args.random_pattern_sampling)
        test_dataloader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        pbar = tqdm.tqdm(test_dataloader)
        for x, patch_index in pbar:
            x = jnp.array(x)
            if getattr(args, "mc_dropout", False) and args.mc_samples > 1:
                mc_args = copy.copy(args)
                if getattr(args, "mc_disable_noise", False):
                    mc_args.add_noise = 0.0
                mc_args.mask_ratio = getattr(args, "mc_mask_ratio", 0.0)
                samples = []
                for s in range(args.mc_samples):
                    rng, sub = jax.random.split(rng)
                    res = eval_model_mc(x, sub, mc_args, mutable_batch_stats=getattr(args, "mc_keep_stats", False))
                    res = {key: np.array(value) for key, value in res.items()}
                    samples.append(res)
                deconv_stack = np.stack([r["deconv"] for r in samples], axis=0)
                result = {}
                result["deconv"] = deconv_stack.mean(axis=0)
                result["uncertainty"] = deconv_stack.std(axis=0)
                for k in ["rec_up", "light_pattern", "background", "psf", "mask"]:
                    if k in samples[0]:
                        result[k] = np.stack([r[k] for r in samples], axis=0).mean(axis=0)
            else:
                result = eval_model(x, rng)
                result = {key: np.array(value) for key, value in result.items()}
            patch_index = [np.array(i) for i in patch_index]
            test_set.assemble_patch(result, patch_index)
        
        
        image_in = test_set.image_in
        image_in = jax.image.resize(image_in, shape=(image_in.shape[0], image_in.shape[1], image_in.shape[2]*args.rescale[0], image_in.shape[3]*args.rescale[1]), method='linear')
        image_in = image_in.astype(np.float32)
        image_in = min_max_norm(image_in)

        # ssl_sim_deconv = percentile_norm(test_set.deconv)
        # reconstruciton = percentile_norm(test_set.reconstruction)
        # lightfield = percentile_norm(test_set.lightfield)
        # background = percentile_norm(test_set.background)
        
        # p_low = np.percentile(img, p_low)
        # wf = image_in.mean(axis=0, keepdims=True)
        # mask = wf > np.percentile(wf, 0.2)
        # closed operation
        # mask = ndimage.binary_closing(mask, iterations=2)
        # lightfield = lightfield * mask
        
        # res_list = np.concatenate([image_in, reconstruciton, ssl_sim_deconv, lightfield, background], axis=1)
        save_emitters_dir = os.path.join(target_dir, "test")
        os.makedirs(save_emitters_dir, exist_ok=True)
        save_meta_dir = os.path.join(target_dir, "test_meta")
        os.makedirs(save_meta_dir, exist_ok=True)
        
        file_name, _ = os.path.splitext(os.path.split(file)[-1])

        save_tiff_imagej_compatible(os.path.join(save_emitters_dir, file_name + ".tif"), test_set.deconv.astype(np.float32).squeeze(), "YX")
        if hasattr(test_set, "uncertainty"):
            save_tiff_imagej_compatible(os.path.join(save_emitters_dir, file_name + "_uncertainty.tif"), test_set.uncertainty.astype(np.float32).squeeze(), "YX")
        save_tiff_imagej_compatible(os.path.join(save_meta_dir, file_name + "_lp.tif"), test_set.light_pattern.astype(np.float32).squeeze(), "ZYX")
        save_tiff_imagej_compatible(os.path.join(save_meta_dir, file_name + "_bg.tif"), test_set.background.astype(np.float32).squeeze(), "YX")
        
        if i == 0:
            save_tiff_imagej_compatible(os.path.join(args.save_dir, "psf.tif"), result["psf"][0,0, ...].astype(np.float32), "ZYX")

        # TensorBoard logging for emitter uncertainty distribution
        if writer is not None and hasattr(test_set, "uncertainty"):
            unc = test_set.uncertainty.astype(np.float32)
            writer.add_histogram("uncertainty/emitter_std", unc.flatten(), i)
            writer.add_scalar("uncertainty/emitter_std_mean", float(unc.mean()), i)
            writer.flush()


def pipeline_infer_pkl(args, writer=None):
    import os, glob, pickle, copy
    import numpy as np
    import tqdm

    import jax
    import jax.numpy as jnp
    import optax

    from tifffile import imread
    from torch.utils.data import DataLoader

    # flax 工具：用于扁平化参数树，做“按 key 匹配”的合并加载
    from flax.core import freeze, unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict

    # ------------------------------------------------------------
    # 0) 处理待推理数据列表（和你原来逻辑一致）
    # ------------------------------------------------------------
    if any(ext in args.data_dir for ext in ["*.tif", "*.png", "*.jpg"]):
        file_names = glob.glob(args.data_dir)
        print("Images num:", len(file_names))
    else:
        file_names = [args.data_dir]
    file_names.sort()

    if len(file_names) == 0:
        raise ValueError(f"No input files found from args.data_dir={args.data_dir}")

    # ------------------------------------------------------------
    # 1) 决定网络用的 num_p（尽量兼容你给的 x_exp.shape[2] 思路）
    # ------------------------------------------------------------
    # 优先用 args.num_p；如果没有，就尽量用 target_pattern_frames；
    # 再不行就从第一张图推断 Z 维。
    num_p = getattr(args, "num_p", None)
    if num_p is None:
        if getattr(args, "adapt_pattern_dimension", False) and getattr(args, "target_pattern_frames", None) is not None:
            num_p = int(args.target_pattern_frames)
        else:
            img0 = imread(file_names[0]).astype(np.float32)
            if img0.ndim == 2:
                num_p = 1
            elif img0.ndim == 3:
                # 认为是 ZYX
                num_p = int(img0.shape[0])
            elif img0.ndim == 4:
                # 认为是 CZYX
                num_p = int(img0.shape[1])
            else:
                raise ValueError(f"Unsupported image ndim={img0.ndim} for inferring num_p")
        setattr(args, "num_p", num_p)

    # ------------------------------------------------------------
    # 2) 和训练时完全一致的网络构造
    # ------------------------------------------------------------
    def net_model():
        image_size = [
            num_p,
            args.crop_size[0] * args.rescale[0],
            args.crop_size[1] * args.rescale[1],
        ]
        return PiMAE(image_size, args.patch_size, args.psf_size, args.lrc)

    # ------------------------------------------------------------
    # 3) 构造同结构 TrainState（用 dummy 输入 init）
    # ------------------------------------------------------------
    def create_train_state(rng):
        net = net_model()
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        x_init = jnp.zeros((1, 1, num_p, *args.crop_size), dtype=jnp.float32)

        variables = net.init(
            {"params": rng1, "dropout": rng2, "random_masking": rng3, "drop_path": rng4},
            x_init,
            args,
            True,
        )

        # 推理不需要 opt_state，但 TrainState.create 要 tx，这里保持和你原来一致
        lr_schedule = optax.linear_schedule(init_value=0, end_value=1e-4, transition_steps=100)
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule),
        )

        return TrainState.create(
            apply_fn=net.apply,
            params=variables["params"],
            batch_stats=variables.get("batch_stats", {}),
            tx=tx,
        )

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng)

    # ------------------------------------------------------------
    # 4) 从 pkl 加载并合并到 state（核心替换点）
    # ------------------------------------------------------------
    def _resolve_pkl_path():
        p = getattr(args, "pkl_path", None) or getattr(args, "resume_path", None)
        if p is None:
            raise ValueError("Please set args.pkl_path (recommended) or args.resume_path to a .pkl file")

        p = os.path.abspath(p)
        if os.path.isdir(p):
            cands = sorted(glob.glob(os.path.join(p, "*.pkl")))
            if not cands:
                raise ValueError(f"No .pkl found under directory: {p}")
            return cands[-1]

        if not os.path.isfile(p):
            raise ValueError(f"pkl file not found: {p}")
        return p

    def _as_dict(tree):
        # FrozenDict -> dict
        try:
            return unfreeze(tree)
        except Exception:
            return tree

    def _adapt_leaf(loaded_leaf, target_leaf):
        """把 loaded_leaf 适配成 target_leaf 的 shape/dtype：
        - shape 相同：直接用（必要时 cast）
        - (n, ...target_shape)：认为是 replicated，取第 0 个
        - 否则返回 None（表示不加载这个 leaf）
        """
        if loaded_leaf is None:
            return None

        # target_leaf 可能不是 array（但一般都是）
        try:
            tgt_arr = jnp.asarray(target_leaf)
            tgt_shape = tuple(tgt_arr.shape)
            tgt_dtype = tgt_arr.dtype
        except Exception:
            return None

        arr = jnp.asarray(loaded_leaf)

        # replicated -> unreplicate
        if arr.ndim == len(tgt_shape) + 1 and tuple(arr.shape[1:]) == tgt_shape:
            arr = arr[0]

        if tuple(arr.shape) != tgt_shape:
            return None

        if arr.dtype != tgt_dtype:
            arr = arr.astype(tgt_dtype)
        return arr

    def _detect_best_prefix(target_keys_set, loaded_keys_set, max_prefix_depth=6):
        # 构造候选 prefix：target key 的所有前缀 + 空前缀
        prefixes = {()}
        for k in target_keys_set:
            for d in range(1, min(len(k), max_prefix_depth) + 1):
                prefixes.add(k[:d])

        best_prefix = ()
        best_score = -1.0
        loaded_keys_list = list(loaded_keys_set)
        denom = max(1, len(loaded_keys_list))

        for pfx in prefixes:
            hits = 0
            for lk in loaded_keys_list:
                if (pfx + lk) in target_keys_set:
                    hits += 1
            score = hits / denom
            # 同分时选更长的 prefix（更具体）
            if (score > best_score) or (score == best_score and len(pfx) > len(best_prefix)):
                best_score = score
                best_prefix = pfx

        return best_prefix, best_score

    def load_pkl_into_state(state, pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        loaded_params = None
        loaded_batch_stats = None

        # 兼容几种常见保存方式
        if hasattr(obj, "params"):
            loaded_params = obj.params
            loaded_batch_stats = getattr(obj, "batch_stats", None)
        elif isinstance(obj, dict):
            if "params" in obj:
                loaded_params = obj["params"]
                loaded_batch_stats = obj.get("batch_stats", None)
            elif "state" in obj and isinstance(obj["state"], dict) and "params" in obj["state"]:
                loaded_params = obj["state"]["params"]
                loaded_batch_stats = obj["state"].get("batch_stats", None)
            else:
                # 直接就是 params pytree
                loaded_params = obj
        else:
            raise TypeError(f"Unsupported pkl content type: {type(obj)}")

        # 扁平化后按 key 合并
        tgt_params_dict = _as_dict(state.params)
        tgt_params_flat = flatten_dict(tgt_params_dict)

        loaded_params_dict = _as_dict(loaded_params)
        loaded_params_flat = flatten_dict(loaded_params_dict)

        target_keys = set(tgt_params_flat.keys())
        loaded_keys = set(loaded_params_flat.keys())

        best_prefix, score = _detect_best_prefix(target_keys, loaded_keys)
        print(f"\033[34mPKL load:\033[0m best_prefix={best_prefix}  coverage={score:.3f}")

        # 可选：像你 pretrain 那样跳过 patch/pos embedding
        # 默认不跳过；如果你需要，设置 args.pkl_exclude_patch_pos_embed=True
        exclude_patch_pos = bool(getattr(args, "pkl_exclude_patch_pos_embed", False))

        new_params_flat = dict(tgt_params_flat)
        n_loaded = 0
        n_skipped_shape = 0
        n_missing_key = 0
        n_excluded = 0

        for lk, lv in loaded_params_flat.items():
            if exclude_patch_pos and len(lk) == 1 and lk[0] in ("patch_embed", "pos_embed"):
                n_excluded += 1
                continue

            tk = best_prefix + lk
            if tk not in new_params_flat:
                # 再试一次：有些 pkl 可能本来就是完整 key（不用 prefix）
                if lk in new_params_flat:
                    tk = lk
                else:
                    n_missing_key += 1
                    continue

            adapted = _adapt_leaf(lv, new_params_flat[tk])
            if adapted is None:
                n_skipped_shape += 1
                continue

            new_params_flat[tk] = adapted
            n_loaded += 1

        merged_params = freeze(unflatten_dict(new_params_flat))

        # batch_stats（若 pkl 有就加载，没有就保留 init 的）
        merged_batch_stats = state.batch_stats
        if loaded_batch_stats is not None and hasattr(state, "batch_stats"):
            tgt_bs_dict = _as_dict(state.batch_stats)
            tgt_bs_flat = flatten_dict(tgt_bs_dict)

            loaded_bs_dict = _as_dict(loaded_batch_stats)
            loaded_bs_flat = flatten_dict(loaded_bs_dict)

            target_bs_keys = set(tgt_bs_flat.keys())
            loaded_bs_keys = set(loaded_bs_flat.keys())
            bs_prefix, bs_score = _detect_best_prefix(target_bs_keys, loaded_bs_keys)

            new_bs_flat = dict(tgt_bs_flat)
            bs_loaded = 0
            bs_skipped = 0
            for lk, lv in loaded_bs_flat.items():
                tk = bs_prefix + lk
                if tk not in new_bs_flat:
                    if lk in new_bs_flat:
                        tk = lk
                    else:
                        continue
                adapted = _adapt_leaf(lv, new_bs_flat[tk])
                if adapted is None:
                    bs_skipped += 1
                    continue
                new_bs_flat[tk] = adapted
                bs_loaded += 1

            merged_batch_stats = freeze(unflatten_dict(new_bs_flat))
            print(f"batch_stats loaded={bs_loaded}, skipped_shape={bs_skipped}, best_prefix={bs_prefix}, coverage={bs_score:.3f}")

        print(
            f"\033[34mPKL params summary:\033[0m "
            f"loaded={n_loaded}, skipped_shape={n_skipped_shape}, missing_key={n_missing_key}, excluded={n_excluded}"
        )

        # 写回 state
        if hasattr(state, "batch_stats"):
            return state.replace(params=merged_params, batch_stats=merged_batch_stats)
        else:
            return state.replace(params=merged_params)

    pkl_path = _resolve_pkl_path()
    print("Load pkl from:", pkl_path)
    state = load_pkl_into_state(state, pkl_path)

    # ------------------------------------------------------------
    # 5) 推理函数（沿用你原来的 apply 逻辑）
    # ------------------------------------------------------------
    @jax.jit
    def eval_model(x, rng):
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        result, _ = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            x,
            args,
            False,
            rngs={"dropout": rng2, "random_masking": rng3, "drop_path": rng4},
            mutable=["batch_stats"],
        )
        return result

    @jax.jit
    def eval_model_mc(x, rng, mc_args, mutable_batch_stats=False):
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        mutable = ["batch_stats"] if mutable_batch_stats else False
        result, _ = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            x,
            mc_args,
            True,
            rngs={"dropout": rng2, "random_masking": rng3, "drop_path": rng4},
            mutable=mutable,
        )
        return result

    # ------------------------------------------------------------
    # 6) 下面基本完全照搬你的推理/保存逻辑（只修了一个 if 判断 bug）
    # ------------------------------------------------------------
    for i, file in enumerate(file_names):
        if any(ext in args.data_dir for ext in ["*.tif", "*.png", "*.jpg"]):
            relative_path = os.path.relpath(file, os.path.commonpath([args.data_dir.replace("*", ""), file]))
            relative_path_star = os.path.relpath(args.data_dir, os.path.commonpath([args.data_dir.replace("*", ""), file]))
            star_indices = [ii for ii, part in enumerate(relative_path_star.split(os.sep)) if "*" in part]
            relative_parts = relative_path.split(os.sep)
            selected_parts = [relative_parts[ii] for ii in star_indices]
            target_dir = os.path.join(args.save_dir, *selected_parts[:-1])
        else:
            target_dir = args.save_dir

        img = imread(file).astype(np.float32)  # CZXY (你的注释)
        if len(img.shape) == 2:
            img = img[np.newaxis, np.newaxis, :, :]
        elif len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]
        print(file, img.shape)

        patch_size_z = args.patch_size[0] if len(args.patch_size) > 2 else 1
        test_set = dataset_3d_infer(
            img,
            args.crop_size,
            args.rescale,
            patch_size_z=patch_size_z,
            adapt_pattern_dimension=args.adapt_pattern_dimension,
            target_pattern_frames=args.target_pattern_frames,
            random_pattern_sampling=args.random_pattern_sampling,
        )
        test_dataloader = DataLoader(
            test_set,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        pbar = tqdm.tqdm(test_dataloader)
        for x, patch_index in pbar:
            x = jnp.array(x)

            if getattr(args, "mc_dropout", False) and args.mc_samples > 1:
                mc_args = copy.copy(args)
                if getattr(args, "mc_disable_noise", False):
                    mc_args.add_noise = 0.0
                mc_args.mask_ratio = getattr(args, "mc_mask_ratio", 0.0)

                samples = []
                for s in range(args.mc_samples):
                    rng, sub = jax.random.split(rng)
                    res = eval_model_mc(
                        x,
                        sub,
                        mc_args,
                        mutable_batch_stats=getattr(args, "mc_keep_stats", False),
                    )
                    res = {key: np.array(value) for key, value in res.items()}
                    samples.append(res)

                deconv_stack = np.stack([r["deconv"] for r in samples], axis=0)
                result = {}
                result["deconv"] = deconv_stack.mean(axis=0)
                result["uncertainty"] = deconv_stack.std(axis=0)

                for k in ["rec_up", "light_pattern", "background", "psf", "mask"]:
                    if k in samples[0]:
                        result[k] = np.stack([r[k] for r in samples], axis=0).mean(axis=0)
            else:
                result = eval_model(x, rng)
                result = {key: np.array(value) for key, value in result.items()}

            patch_index = [np.array(ii) for ii in patch_index]
            test_set.assemble_patch(result, patch_index)

        image_in = test_set.image_in
        image_in = jax.image.resize(
            image_in,
            shape=(
                image_in.shape[0],
                image_in.shape[1],
                image_in.shape[2] * args.rescale[0],
                image_in.shape[3] * args.rescale[1],
            ),
            method="linear",
        )
        image_in = image_in.astype(np.float32)
        image_in = min_max_norm(image_in)

        save_emitters_dir = os.path.join(target_dir, "test")
        os.makedirs(save_emitters_dir, exist_ok=True)
        save_meta_dir = os.path.join(target_dir, "test_meta")
        os.makedirs(save_meta_dir, exist_ok=True)

        file_name, _ = os.path.splitext(os.path.split(file)[-1])

        save_tiff_imagej_compatible(
            os.path.join(save_emitters_dir, file_name + ".tif"),
            test_set.deconv.astype(np.float32).squeeze(),
            "YX",
        )
        if hasattr(test_set, "uncertainty"):
            save_tiff_imagej_compatible(
                os.path.join(save_emitters_dir, file_name + "_uncertainty.tif"),
                test_set.uncertainty.astype(np.float32).squeeze(),
                "YX",
            )

        save_tiff_imagej_compatible(
            os.path.join(save_meta_dir, file_name + "_lp.tif"),
            test_set.light_pattern.astype(np.float32).squeeze(),
            "ZYX",
        )
        save_tiff_imagej_compatible(
            os.path.join(save_meta_dir, file_name + "_bg.tif"),
            test_set.background.astype(np.float32).squeeze(),
            "YX",
        )

        if i == 0 and "psf" in result:
            save_tiff_imagej_compatible(
                os.path.join(args.save_dir, "psf.tif"),
                result["psf"][0, 0, ...].astype(np.float32),
                "ZYX",
            )

        # TensorBoard logging for emitter uncertainty distribution
        if writer is not None and hasattr(test_set, "uncertainty"):
            unc = test_set.uncertainty.astype(np.float32)
            writer.add_histogram("uncertainty/emitter_std", unc.flatten(), i)
            writer.add_scalar("uncertainty/emitter_std_mean", float(unc.mean()), i)
            writer.flush()

    return
