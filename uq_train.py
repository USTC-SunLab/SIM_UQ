# train_uq_multi.py
# -*- coding: utf-8 -*-
"""
核心训练脚本（pmap 多卡）：
- args 从 uq_args.py 导入
- debug 可视化/保存从 uq_vis.py 导入（训练脚本不再写 matplotlib 逻辑）

模型输出约定：
  rec, light_pattern, emitter, psf = model(x, ...)

Loss:
  rec_loss（L1 + MS-SSIM）
+ args.tv_loss   * TV(psf)
+ args.lp_tv     * TV(light_pattern)
+ args.psfc_loss * center_loss(psf)

新增：预测不确定性（MC Dropout）
- 仅在 debug 保存阶段触发（避免训练 step 额外开销）
- training=True 多次 forward，估计 rec/deconv 的均值与标准差 std（不确定度）
- 可选把 MC 放到 cpu 或指定 gpu，尽量避免训练 GPU 显存碎片
"""

from __future__ import annotations

import pickle
import os
import csv
import glob
import time
import copy
import gc
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Mapping

from utils_metrics import ms_ssim_3d

import numpy as np
import torch
from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
from jax import lax
import optax
from flax.training import train_state, checkpoints
from flax import jax_utils
from flax.core import freeze, unfreeze

import uq_data
from uq_data import dataset_2d_sim_supervised

from uq_args import parse_args
from uq_vis import save_debug_artifacts


# =============================================================================
# 0) training crop: 静默版本（避免刷屏）
# =============================================================================
def _norm_5d_leading_dims(t):
    """把 (1,B,...) 纠正成 (B,1,...)；其它情况原样返回。"""
    if t is None or t.ndim != 5:
        return t
    if (t.shape[0] == 1) and (t.shape[1] != 1):
        t = jnp.transpose(t, (1, 0, 2, 3, 4))
    return t


def _to_b1_5d(t):
    """
    统一把输入变成 (B,1,...) 的 5D：
    - 4D: (B,Z,Y,X) -> (B,1,Z,Y,X)
    - 5D: 修正 (1,B,...) -> (B,1,...)；其余不动
    - 3D: (Z,Y,X) -> (1,1,Z,Y,X) 兜底
    """
    if t is None:
        return None
    if t.ndim == 5:
        return _norm_5d_leading_dims(t)
    if t.ndim == 4:
        return t[:, None, ...]
    if t.ndim == 3:
        return t[None, None, ...]
    return t


def _format_param_paths(paths, max_items: int = 5) -> str:
    if not paths:
        return ""
    show = ["/".join([str(p) for p in path]) for path in paths[:max_items]]
    more = f" (+{len(paths) - max_items} more)" if len(paths) > max_items else ""
    return ", ".join(show) + more


def _merge_pretrained_params(init_params, loaded_params):
    if loaded_params is None:
        return init_params, {"missing": [], "unexpected": [], "mismatched": []}

    init = unfreeze(init_params)
    loaded = unfreeze(loaded_params)
    missing = []
    unexpected = []
    mismatched = []

    def _merge(i, l, path):
        if isinstance(i, Mapping):
            if not isinstance(l, Mapping):
                mismatched.append(path)
                return i
            out = {}
            for k, v in i.items():
                if k in l:
                    out[k] = _merge(v, l[k], path + (k,))
                else:
                    missing.append(path + (k,))
                    out[k] = v
            for k in l.keys():
                if k not in i:
                    unexpected.append(path + (k,))
            return out

        if isinstance(l, Mapping):
            mismatched.append(path)
            return i

        try:
            if hasattr(i, "shape") and hasattr(l, "shape") and i.shape != l.shape:
                mismatched.append(path)
                return i
        except Exception:
            mismatched.append(path)
            return i
        return l

    merged = _merge(init, loaded, ())
    report = {"missing": missing, "unexpected": unexpected, "mismatched": mismatched}
    return freeze(merged), report


def _normalize_pretrain_tree_for_base(tree, label: str):
    if tree is None:
        return tree
    if isinstance(tree, Mapping):
        if "base" in tree and any(k in tree["base"] for k in ("pt_predictor", "PSF_predictor", "psf_seed")):
            print(f"[pretrain] unwrapping {label} from 'base' for base model")
            return tree["base"]
    return tree


def _align_like(target, t):
    """
    把 t 的维度/前两维布局尽量对齐到 target（只处理你描述的 4D/5D + 单通道情形）。
    - target 5D: 期望 t 也变成 5D (B,1,Z,Y,X)
    - target 4D: 期望 t 也变成 4D (B,Z,Y,X)
    """
    if t is None:
        return None

    target = _norm_5d_leading_dims(target)
    t = _norm_5d_leading_dims(t)

    if target.ndim == t.ndim:
        return t

    # target 是 5D，而 t 是 4D -> 给 t 加单通道维
    if target.ndim == 5 and t.ndim == 4:
        return t[:, None, ...]  # (B,Z,Y,X) -> (B,1,Z,Y,X)

    # target 是 4D，而 t 是 5D -> 去掉单通道维（优先 squeeze axis=1）
    if target.ndim == 4 and t.ndim == 5:
        if t.shape[1] == 1:
            return jnp.squeeze(t, axis=1)  # (B,1,Z,Y,X) -> (B,Z,Y,X)
        if t.shape[0] == 1:
            return jnp.squeeze(t, axis=0)  # (1,B,Z,Y,X) -> (B,Z,Y,X)
        return t

    # 额外兜底：偶尔可能是 3D (Z,Y,X)
    if target.ndim == 5 and t.ndim == 3:
        return t[None, None, ...]  # -> (1,1,Z,Y,X)
    if target.ndim == 4 and t.ndim == 3:
        return t[None, ...]  # -> (1,Z,Y,X)

    return t


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
        return t.shape[-3] * t.shape[-2] * t.shape[-1]

    cz = _tensor_size(img[:, 1:, :, :])
    cy = _tensor_size(img[:, :, 1:, :])
    cx = _tensor_size(img[:, :, :, 1:])

    hz = lax.pow(jnp.abs(img[:, 1:, :, :] - img[:, : z - 1, :, :]), 2.0).sum()
    hy = lax.pow(jnp.abs(img[:, :, 1:, :] - img[:, :, : y - 1, :]), 2.0).sum()
    hx = lax.pow(jnp.abs(img[:, :, :, 1:] - img[:, :, :, : x - 1]), 2.0).sum()

    if cz == 0:
        return (hy / cy + hx / cx) / batch_size
    else:
        return (hz / cz + hy / cy + hx / cx) / batch_size


# =============================================================================
# 1) small utils
# =============================================================================
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_bchw_9(x: np.ndarray) -> np.ndarray:
    """
    确保输入是 (B,9,H,W)
    支持：
      - (B,9,H,W)
      - (B,H,W,9)
      - (9,H,W)
      - (H,W,9)
      - (B,1,9,H,W) -> squeeze
    """
    x = np.asarray(x)
    if x.ndim == 5:
        # (B,1,9,H,W) -> (B,9,H,W)
        if x.shape[1] == 1:
            x = x[:, 0]
        else:
            x = x[:, 0]

    if x.ndim == 4:
        if x.shape[1] == 9:
            return x
        if x.shape[-1] == 9:
            return np.transpose(x, (0, 3, 1, 2))
        raise ValueError(f"Expected 9 channels, got shape {x.shape}")

    if x.ndim == 3:
        if x.shape[0] == 9:
            return x[None, ...]
        if x.shape[-1] == 9:
            return np.transpose(x[None, ...], (0, 3, 1, 2))
        raise ValueError(f"Expected 9 channels, got shape {x.shape}")

    raise ValueError(f"Unexpected input ndim={x.ndim}, shape={x.shape}")


def shard_batch(x: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    b = x.shape[0]
    if b % n_devices != 0:
        raise ValueError(f"batch_size={b} must be divisible by n_devices={n_devices}")
    return x.reshape((n_devices, b // n_devices) + x.shape[1:])


def psnr_from_mse(mse: jnp.ndarray, max_val: float = 1.0, eps: float = 1e-8) -> jnp.ndarray:
    return 20.0 * jnp.log10(max_val) - 10.0 * jnp.log10(mse + eps)


def csv_append(path: str, row: Dict[str, Any], header: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# =============================================================================
# 2) PSF regularizers (JAX)
# =============================================================================
def psf_mass_2d(psf: jnp.ndarray) -> jnp.ndarray:
    """把 psf 投影到 2D mass map，并归一化到 sum=1，输出 (B,H,W)。"""
    psf = jnp.asarray(psf).astype(jnp.float32)

    if psf.ndim == 2:
        mass = psf[None, ...]
    elif psf.ndim == 3:
        mass = jnp.sum(psf, axis=0, keepdims=True)
    elif psf.ndim == 4:
        mass = jnp.sum(psf, axis=1)
    elif psf.ndim == 5:
        mass = jnp.sum(psf, axis=(1, 2))
    else:
        axes = tuple(range(psf.ndim - 2))
        mass2d = jnp.sum(psf, axis=axes)
        mass = mass2d[None, ...]

    mass = jnp.maximum(mass, 0.0)
    mass = mass / (jnp.sum(mass, axis=(-2, -1), keepdims=True) + 1e-8)
    if mass.ndim == 2:
        mass = mass[None, ...]
    return mass


def tv_loss(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x)
    dh = jnp.abs(x[..., 1:, :] - x[..., :-1, :])
    dw = jnp.abs(x[..., :, 1:] - x[..., :, :-1])
    return jnp.mean(dh) + jnp.mean(dw)


def center_loss(psf: jnp.ndarray) -> jnp.ndarray:
    mass = psf_mass_2d(psf)  # (B,H,W)
    H, W = mass.shape[-2], mass.shape[-1]

    ys = jnp.arange(H, dtype=jnp.float32)[None, :, None]
    xs = jnp.arange(W, dtype=jnp.float32)[None, None, :]

    cy = jnp.sum(mass * ys, axis=(-2, -1))
    cx = jnp.sum(mass * xs, axis=(-2, -1))

    ty = (H - 1) * 0.5
    tx = (W - 1) * 0.5
    return jnp.mean((cy - ty) ** 2 + (cx - tx) ** 2)


def gaussian_prior_loss(psf: jnp.ndarray, sigma: float = 3.0) -> jnp.ndarray:
    mass = psf_mass_2d(psf)  # (B,H,W)
    H, W = mass.shape[-2], mass.shape[-1]

    yy = jnp.arange(H, dtype=jnp.float32)[:, None]
    xx = jnp.arange(W, dtype=jnp.float32)[None, :]
    cy = (H - 1) * 0.5
    cx = (W - 1) * 0.5

    s2 = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float32), 1e-6) ** 2
    g = jnp.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s2))
    g = g / (jnp.sum(g) + 1e-8)
    g = g[None, ...]

    return jnp.mean((mass - g) ** 2)


# =============================================================================
# 3) TrainState
# =============================================================================
class TrainState(train_state.TrainState):
    batch_stats: Any
    rng: Any


def _pmean_tree(tree):
    return jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="devices"), tree)


# =============================================================================
# 4) debug crop record (only for debug_loader num_workers=0)
# =============================================================================
DEBUG_CROP_RECORDS: List[Dict[str, Any]] = []


def _random_crop_arrays_record(arrays, h, w):
    scales = [int(x.shape[-1] / arrays[0].shape[-1]) for x in arrays]
    H, W = arrays[0].shape[-2:]
    assert h <= H and w <= W
    h0 = int(np.random.randint(0, H - h + 1))
    w0 = int(np.random.randint(0, W - w + 1))
    DEBUG_CROP_RECORDS.append(
        {
            "h0": h0,
            "w0": w0,
            "h": int(h),
            "w": int(w),
            "ref_shape": list(np.asarray(arrays[0]).shape),
            "scales": [int(s) for s in scales],
        }
    )
    out = []
    for array, scale in zip(arrays, scales):
        out.append(array[..., h0 * scale : h0 * scale + h * scale, w0 * scale : w0 * scale + w * scale])
    return out


# =============================================================================
# 5) state init + forward
# =============================================================================
def create_state(rng: jax.Array, model, args) -> TrainState:
    rng, k_params, k_do, k_dp, k_rm = jax.random.split(rng, 5)

    dummy = jnp.zeros((1, 1, 9, args.crop_h, args.crop_w), dtype=jnp.float32)
    variables = model.init(
        {"params": k_params, "dropout": k_do, "drop_path": k_dp, "random_masking": k_rm},
        dummy,
        args,
        True,
    )
    params = variables["params"]
    batch_stats = variables.get("batch_stats", None)

    tx = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats, rng=rng)


def debug_forward_single_device(state: TrainState, x: jnp.ndarray, args, debug_train: bool):
    variables = {"params": state.params}
    if state.batch_stats is not None:
        variables["batch_stats"] = state.batch_stats

    step = int(jax.device_get(state.step))
    rng = jax.random.fold_in(state.rng, step)
    rng, k_do, k_dp, k_rm = jax.random.split(rng, 4)

    outs, _ = state.apply_fn(
        variables,
        x,
        args,
        training=debug_train,
        rngs={"dropout": k_do, "drop_path": k_dp, "random_masking": k_rm},
        mutable=["batch_stats"] if state.batch_stats is not None else [],
    )
    rec, light_pattern, emitter, psf, mask, deconv = outs
    return rec, light_pattern, emitter, psf, mask, deconv


# =============================================================================
# 5.1) MC Dropout uncertainty (debug-only)
# =============================================================================
def _select_mc_devices(devices: List[jax.Device], args) -> List[jax.Device]:
    """
    选择用于 MC 的 device：
    - args.mc_device == "gpu": 优先指定 id，否则用最后一张 GPU
    - 否则用 CPU(0)
    """
    mc_devices: List[jax.Device] = []
    try:
        if getattr(args, "mc_device", "cpu") == "gpu":
            gpus = [d for d in devices if d.platform == "gpu"]
            if gpus:
                preferred_id = getattr(args, "mc_device_id", None)
                if preferred_id is not None:
                    chosen = [d for d in gpus if getattr(d, "id", None) == preferred_id]
                    mc_devices = chosen if chosen else [gpus[-1]]
                else:
                    mc_devices = [gpus[-1]]

        if not mc_devices:
            cpu_devs = jax.devices("cpu")
            if cpu_devs:
                mc_devices = [cpu_devs[0]]
    except Exception:
        mc_devices = [devices[0]] if devices else []

    return mc_devices


def mc_dropout_uncertainty(
    state_single: TrainState,
    x_np: np.ndarray,
    args,
    mc_samples: int,
    devices: List[jax.Device],
    epoch: int,
    step: int,
) -> Dict[str, Any]:
    """
    对单个 batch（建议 B=1）做 MC Dropout：
    返回 rec/deconv 的 mean/std（numpy），以及 summary 指标。
    """
    mc_samples = int(mc_samples)
    if mc_samples <= 1:
        return {}

    mc_devices = _select_mc_devices(devices, args)
    if not mc_devices:
        return {}

    mc_dev = mc_devices[0]

    # copy args for MC
    mc_args = copy.copy(args)
    if getattr(args, "mc_disable_noise", False) and hasattr(mc_args, "add_noise"):
        mc_args.add_noise = 0.0
    if hasattr(mc_args, "mask_ratio"):
        mc_args.mask_ratio = float(getattr(args, "mc_mask_ratio", getattr(args, "mask_ratio", 0.0)))

    # move state + input to MC device (tree_map must handle None)
    state_mc = jax.tree_util.tree_map(lambda x: jax.device_put(x, mc_dev) if x is not None else None, state_single)
    x_mc = jax.device_put(jnp.asarray(x_np), mc_dev)

    # base rng
    base_seed = int(getattr(args, "mc_seed", 0) or getattr(args, "debug_seed", 0) or getattr(args, "seed", 0))
    rng = jax.random.PRNGKey(base_seed)
    rng = jax.random.fold_in(rng, int(epoch))
    rng = jax.random.fold_in(rng, int(step))
    rng = jax.random.fold_in(rng, 0xC0FFEE)

    target = getattr(args, "mc_target", "both")

    def single_forward(rng_in):
        _, k_do, k_dp, k_rm = jax.random.split(rng_in, 4)

        variables = {"params": state_mc.params}
        if state_mc.batch_stats is not None:
            variables["batch_stats"] = state_mc.batch_stats

        outs, _ = state_mc.apply_fn(
            variables,
            x_mc,
            mc_args,
            True,  # training=True 才启用 dropout / drop_path / random_masking
            rngs={"dropout": k_do, "drop_path": k_dp, "random_masking": k_rm},
            mutable=["batch_stats"] if state_mc.batch_stats is not None else [],
        )
        rec, light_pattern, emitter, psf, mask, deconv = outs

        rec = _to_b1_5d(rec)
        deconv = _to_b1_5d(deconv)
        return rec, deconv

    # 在线统计（Welford），不堆叠 stack，省内存
    rec_mean = rec_m2 = None
    de_mean = de_m2 = None
    n = 0

    for _ in range(mc_samples):
        rng, sub = jax.random.split(rng)
        with jax.default_device(mc_dev):
            rec_s, de_s = single_forward(sub)

        rec_s = np.asarray(jax.device_get(rec_s), dtype=np.float32)
        de_s = np.asarray(jax.device_get(de_s), dtype=np.float32)
        n += 1

        if target in ("rec", "both"):
            if rec_mean is None:
                rec_mean = rec_s
                rec_m2 = np.zeros_like(rec_s, dtype=np.float32)
            else:
                delta = rec_s - rec_mean
                rec_mean = rec_mean + delta / n
                rec_m2 = rec_m2 + delta * (rec_s - rec_mean)

        if target in ("deconv", "both"):
            if de_mean is None:
                de_mean = de_s
                de_m2 = np.zeros_like(de_s, dtype=np.float32)
            else:
                delta = de_s - de_mean
                de_mean = de_mean + delta / n
                de_m2 = de_m2 + delta * (de_s - de_mean)

    out: Dict[str, Any] = {
        "mc_samples": mc_samples,
        "mc_device": f"{mc_dev.platform}:{getattr(mc_dev, 'id', 'na')}",
    }

    eps = 1e-8
    if rec_mean is not None:
        rec_var = rec_m2 / max(n - 1, 1)
        rec_std = np.sqrt(np.maximum(rec_var, 0.0)).astype(np.float32)
        out["rec_mean"] = rec_mean.astype(np.float32)
        out["rec_std"] = rec_std
        out["rec_cv"] = (rec_std / (np.abs(rec_mean) + eps)).astype(np.float32)
        out["rec_std_mean"] = float(rec_std.mean())
        out["rec_std_p95"] = float(np.percentile(rec_std, 95))

    if de_mean is not None:
        de_var = de_m2 / max(n - 1, 1)
        de_std = np.sqrt(np.maximum(de_var, 0.0)).astype(np.float32)
        out["deconv_mean"] = de_mean.astype(np.float32)
        out["deconv_std"] = de_std
        out["deconv_cv"] = (de_std / (np.abs(de_mean) + eps)).astype(np.float32)
        out["deconv_std_mean"] = float(de_std.mean())
        out["deconv_std_p95"] = float(np.percentile(de_std, 95))

    gc.collect()
    return out


# =============================================================================
# 6) main
# =============================================================================
def main():
    args = parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    devices = jax.devices()
    n_devices = jax.local_device_count()
    device0 = devices[0]
    print(f"[jax] local_device_count={n_devices}")
    print(f"[jax] devices={devices}")

    if args.batch_size % n_devices != 0:
        raise ValueError(f"--batch_size={args.batch_size} must be divisible by n_devices={n_devices}")

    # paths
    train_paths = sorted(glob.glob(args.train_glob))
    if len(train_paths) == 0:
        raise RuntimeError(f"No train files matched: {args.train_glob}")

    # dataset/loader
    train_ds = dataset_2d_sim_supervised(
        train_paths,
        crop_size=(args.crop_h, args.crop_w),
        use_gt=False,
        gt_paths=None,
    )
    pin = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # optional debug loader (single file)
    debug_loader = None
    debug_path = None
    if args.debug_every_epochs and args.debug_every_epochs > 0:
        if not (0 <= args.debug_index < len(train_paths)):
            raise ValueError(f"--debug_index {args.debug_index} out of range [0, {len(train_paths)-1}]")
        debug_path = train_paths[args.debug_index]
        debug_ds = dataset_2d_sim_supervised(
            [debug_path],
            crop_size=(args.crop_h, args.crop_w),
            use_gt=False,
            gt_paths=None,
        )
        debug_loader = DataLoader(
            debug_ds,
            batch_size=args.debug_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,  # 必须 0，才能稳定记录 crop 坐标
            pin_memory=False,
        )

    # model
    from network import PiMAE

    model = PiMAE(
        (9, args.crop_size[0] * args.rescale[0], args.crop_size[1] * args.rescale[1]),
        (3, 16, 16),
        (49, 49),
        args.lrc9_rank,
    )

    rng = jax.random.PRNGKey(args.seed)
    state = create_state(rng, model, args)
    print("[init] success")

    if args.resume:
        state = checkpoints.restore_checkpoint(args.ckpt_dir, state)
        print(f"[resume] restored from {args.ckpt_dir}, step={int(jax.device_get(state.step))}")
    if args.resume_pickle is not None and not args.resume:
        print(f"\033[94mLoading BioSR pretrained checkpoint from: {args.resume_pickle}\033[0m")
        with open(args.resume_pickle, "rb") as f:
            biosr_pretrain = pickle.load(f)

        # BioSR pretrained checkpoints have 'params' and optionally 'batch_stats'
        if "params" not in biosr_pretrain:
            raise ValueError(
                f"Invalid BioSR pretrained checkpoint format. Expected 'params' key, got: {list(biosr_pretrain.keys())}"
            )

        loaded_params = _normalize_pretrain_tree_for_base(biosr_pretrain["params"], "params")
        # Load params (merge to keep new params like psf_seed)
        merged_params, report = _merge_pretrained_params(state.params, loaded_params)
        state = state.replace(params=merged_params)

        if report["missing"]:
            print(
                f"[pretrain] missing params (kept init): {len(report['missing'])} :: "
                f"{_format_param_paths(report['missing'])}"
            )
        if report["unexpected"]:
            print(
                f"[pretrain] unexpected params (ignored): {len(report['unexpected'])} :: "
                f"{_format_param_paths(report['unexpected'])}"
            )
        if report["mismatched"]:
            print(
                f"[pretrain] shape/structure mismatch (kept init): {len(report['mismatched'])} :: "
                f"{_format_param_paths(report['mismatched'])}"
            )

        # Load batch_stats if available
        if "batch_stats" in biosr_pretrain:
            loaded_bs = _normalize_pretrain_tree_for_base(biosr_pretrain["batch_stats"], "batch_stats")
            state = state.replace(batch_stats=loaded_bs)

        # Reset optimizer state to match (possibly updated) params tree
        state = state.replace(opt_state=state.tx.init(state.params))
        print("[pretrain] optimizer state reset")

        print("\033[92mSuccessfully loaded pretrained model\033[0m")

    state_rep = jax_utils.replicate(state)

    # pmap train step
    tv_w = float(args.tv_loss)
    lp_w = float(args.lp_tv)
    psfc_w = float(args.psfc_loss)
    psfg_w = float(args.psfg_loss)
    psf_sigma = float(args.psf_sigma)

    def train_step_pmap(state: TrainState, x: jnp.ndarray):
        rng = jax.random.fold_in(state.rng, state.step)
        rng = jax.random.fold_in(rng, jax.lax.axis_index("devices"))
        rng, k_do, k_dp, k_rm, new_rng = jax.random.split(rng, 5)

        def loss_fn(params, train: bool = True):
            variables = {"params": params}
            if state.batch_stats is not None:
                variables["batch_stats"] = state.batch_stats

            outs, new_state = state.apply_fn(
                variables,
                x,
                args,
                train,
                rngs={"dropout": k_do, "drop_path": k_dp, "random_masking": k_rm},
                mutable=["batch_stats"] if state.batch_stats is not None else [],
            )

            # 你现在的输出 tuple：
            rec, light_pattern, emitter, psf, mask, deconv = outs

            # ---- 统一成 (B,1,...) 5D，解决 4D/5D 混用 + (1,B,...) 问题 ----
            x5 = _to_b1_5d(x)
            rec = _to_b1_5d(rec)
            light_pattern = _to_b1_5d(light_pattern)
            psf = _to_b1_5d(psf)
            mask = _to_b1_5d(mask)
            deconv = _to_b1_5d(deconv)

            new_batch_stats = new_state["batch_stats"] if (state.batch_stats is not None) else None

            # ---- 1) 重建项：严格复刻 compute_metrics 的分支逻辑 ----
            use_mask = train and getattr(args, "mask_ratio", 0.0) > 0.3 and (mask is not None)
            if use_mask:
                rec_term = rec_loss(x5, rec, mask)
            else:
                rec_term = rec_loss(x5, rec)

            # ---- 2) 正则项：TV / center ----
            psf_tv = TV_Loss(psf)
            lp_tv = TV_Loss(light_pattern)
            psf_center = center_loss(psf)

            # 仅当 metric
            deconv_tv = TV_Loss(deconv) if (deconv is not None) else jnp.zeros((), dtype=rec.dtype)

            # ---- 3) 总 loss ----
            loss = rec_term + args.tv_loss * psf_tv + args.psfc_loss * psf_center + args.lp_tv * lp_tv

            # ---- 4) 监控指标 ----
            diff = rec - x5
            rec_mse = jnp.mean(diff * diff)
            rec_mae = jnp.mean(jnp.abs(diff))
            psnr = psnr_from_mse(rec_mse, max_val=1.0)

            # 对齐占位：不炸日志/CSV
            psf_gauss = jnp.zeros((), dtype=rec.dtype)
            # psf_gauss = gaussian_prior_loss(psf, sigma=psf_sigma)

            metrics = {
                "loss": loss,
                "rec_mse": rec_mse,
                "rec_mae": rec_mae,
                "psnr": psnr,
                "psf_tv": psf_tv,
                "lp_tv": lp_tv,
                "psf_center": psf_center,
                "psf_gauss": psf_gauss,
                # extra
                "rec_loss": rec_term,
                "deconv_tv": deconv_tv,
            }

            return loss, (metrics, new_batch_stats)

        (loss, (metrics, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        grads = jax.lax.pmean(grads, axis_name="devices")
        metrics = jax.lax.pmean(metrics, axis_name="devices")

        if new_batch_stats is not None:
            new_batch_stats = _pmean_tree(new_batch_stats)

        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(
            batch_stats=new_batch_stats if new_batch_stats is not None else state.batch_stats,
            rng=new_rng,
        )
        return new_state, metrics

    P_TRAIN_STEP = jax.pmap(train_step_pmap, axis_name="devices", donate_argnums=(0,))

    # logs
    metrics_csv = os.path.join(args.ckpt_dir, "metrics.csv")
    csv_header = [
        "time",
        "epoch",
        "step",
        "loss",
        "rec_mse",
        "rec_mae",
        "psnr",
        "psf_tv",
        "lp_tv",
        "psf_center",
        "psf_gauss",
        "note",
    ]
    if not os.path.exists(metrics_csv):
        csv_append(
            metrics_csv,
            {
                "time": time.time(),
                "epoch": 0,
                "step": int(jax.device_get(state.step)),
                "loss": "",
                "rec_mse": "",
                "rec_mae": "",
                "psnr": "",
                "psf_tv": "",
                "lp_tv": "",
                "psf_center": "",
                "psf_gauss": "",
                "note": f"init n_devices={n_devices}",
            },
            header=csv_header,
        )

    # train loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        interval_buf: List[Dict[str, float]] = []
        epoch_buf: List[Dict[str, float]] = []

        for it, batch in enumerate(train_loader, start=1):
            torch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
            dl_np = _to_numpy(torch_x).astype(np.float32)  # (B,...) maybe include extra dims
            np_bchw = ensure_bchw_9(dl_np).astype(np.float32)  # (B,9,H,W)

            x = jnp.asarray(np_bchw)[:, None, ...]  # (B,1,9,H,W)
            x = shard_batch(x, n_devices)  # (n_devices, per_dev, 1,9,H,W)

            state_rep, metrics = P_TRAIN_STEP(state_rep, x)

            m = {k: float(jax.device_get(v)[0]) for k, v in metrics.items()}
            interval_buf.append(m)
            epoch_buf.append(m)

            step = int(jax.device_get(state_rep.step)[0])

            if (it % args.log_every) == 0:
                avg = {k: float(np.mean([d[k] for d in interval_buf])) for k in interval_buf[0].keys()}
                print(
                    f"[epoch {epoch:03d}][iter {it:05d}] "
                    f"loss={avg['loss']:.6f} rec_mse={avg['rec_mse']:.6f} psnr={avg['psnr']:.2f} "
                    f"psf_tv={avg['psf_tv']:.6f} lp_tv={avg['lp_tv']:.6f} "
                    f"psf_center={avg['psf_center']:.6f} psf_gauss={avg['psf_gauss']:.6f} "
                    f"(step={step})"
                )
                csv_append(
                    metrics_csv,
                    {
                        "time": time.time(),
                        "epoch": epoch,
                        "step": step,
                        "loss": avg["loss"],
                        "rec_mse": avg["rec_mse"],
                        "rec_mae": avg["rec_mae"],
                        "psnr": avg["psnr"],
                        "psf_tv": avg["psf_tv"],
                        "lp_tv": avg["lp_tv"],
                        "psf_center": avg["psf_center"],
                        "psf_gauss": avg["psf_gauss"],
                        "note": "train_interval",
                    },
                    header=csv_header,
                )
                interval_buf = []

        epoch_avg = {k: float(np.mean([d[k] for d in epoch_buf])) for k in epoch_buf[0].keys()}

        # save ckpt
        to_save = jax_utils.unreplicate(state_rep)
        step = int(jax.device_get(to_save.step))
        checkpoints.save_checkpoint(
            ckpt_dir=args.ckpt_dir,
            target=to_save,
            step=step,
            keep=args.keep_ckpts,
            overwrite=True,
        )

        dt = time.time() - t0
        print(
            f"[epoch {epoch:03d}] "
            f"loss={epoch_avg['loss']:.6f} rec_mse={epoch_avg['rec_mse']:.6f} psnr={epoch_avg['psnr']:.2f} "
            f"psf_tv={epoch_avg['psf_tv']:.6f} lp_tv={epoch_avg['lp_tv']:.6f} "
            f"psf_center={epoch_avg['psf_center']:.6f} psf_gauss={epoch_avg['psf_gauss']:.6f} "
            f"| time={dt:.1f}s step={step}"
        )
        csv_append(
            metrics_csv,
            {
                "time": time.time(),
                "epoch": epoch,
                "step": step,
                "loss": epoch_avg["loss"],
                "rec_mse": epoch_avg["rec_mse"],
                "rec_mae": epoch_avg["rec_mae"],
                "psnr": epoch_avg["psnr"],
                "psf_tv": epoch_avg["psf_tv"],
                "lp_tv": epoch_avg["lp_tv"],
                "psf_center": epoch_avg["psf_center"],
                "psf_gauss": epoch_avg["psf_gauss"],
                "note": "train_epoch",
            },
            header=csv_header,
        )

        # ---------------- optional debug saving ----------------
        if debug_loader is not None and args.debug_every_epochs > 0 and (epoch % args.debug_every_epochs == 0):
            out_dir = os.path.join(args.debug_dir, f"epoch_{epoch:03d}_step_{step:07d}")
            os.makedirs(out_dir, exist_ok=True)

            # 固定 debug seed（不影响训练）：只包住 debug 取样 + 记录 crop
            old_crop_fn = uq_data.random_crop_arrays
            uq_data.random_crop_arrays = _random_crop_arrays_record
            DEBUG_CROP_RECORDS.clear()

            rng_state = np.random.get_state()
            np.random.seed(args.debug_seed)
            debug_batch = next(iter(debug_loader))
            np.random.set_state(rng_state)

            uq_data.random_crop_arrays = old_crop_fn
            crop_record = DEBUG_CROP_RECORDS[0] if len(DEBUG_CROP_RECORDS) > 0 else None

            # batch -> numpy -> ensure -> add axis -> device0
            torch_dbg = debug_batch[0] if isinstance(debug_batch, (list, tuple)) else debug_batch
            dl_np = _to_numpy(torch_dbg).astype(np.float32)
            ensured = ensure_bchw_9(dl_np).astype(np.float32)  # (B,9,H,W)
            x_in = ensured[:, None, ...]  # (B,1,9,H,W)
            x_dbg = jax.device_put(jnp.asarray(x_in), device0)

            # single-device state（避免多卡参数 device mismatch）
            state_single = jax_utils.unreplicate(state_rep)
            state_single = jax.device_put(jax.device_get(state_single), device0)

            # forward
            rec, lp, emitter, psf, mask, deconv = debug_forward_single_device(
                state_single,
                x_dbg,
                args,
                debug_train=bool(getattr(args, "debug_train", True)),
            )

            # device_get
            x_np = np.array(jax.device_get(x_dbg)).astype(np.float32)
            rec_np = np.array(jax.device_get(rec)).astype(np.float32)
            lp_np = np.array(jax.device_get(lp)).astype(np.float32)
            emitter_np = np.array(jax.device_get(emitter)).astype(np.float32)
            psf_np = np.array(jax.device_get(psf)).astype(np.float32)

            deconv_np = None
            mask_np = None
            try:
                if deconv is not None:
                    deconv_np = np.array(jax.device_get(deconv)).astype(np.float32)
                if mask is not None:
                    mask_np = np.array(jax.device_get(mask)).astype(np.float32)
            except Exception:
                pass

            # -------- MC Dropout uncertainty (optional) --------
            mc_pack = {}
            mc_samples = int(getattr(args, "mc_dropout_train_samples", 0) or 0)
            if mc_samples > 1:
                x_for_mc = x_in[:1].astype(np.float32)  # 只对 B=1 做 MC，避免太重
                mc_pack = mc_dropout_uncertainty(
                    state_single=state_single,
                    x_np=x_for_mc,
                    args=args,
                    mc_samples=mc_samples,
                    devices=devices,
                    epoch=epoch,
                    step=step,
                )

            # save artifacts (prefer new signature; fallback to legacy if uq_vis not updated yet)
            try:
                dbg_metrics = save_debug_artifacts(
                    out_dir=out_dir,
                    args=args,
                    debug_path=str(debug_path),
                    crop_record=crop_record,
                    x_np=x_np,
                    rec_np=rec_np,
                    lp_np=lp_np,
                    emitter_np=emitter_np,
                    psf_np=psf_np,
                    mask_np=mask_np,
                    deconv_np=deconv_np,
                    mc_samples=int(mc_pack.get("mc_samples", 0)) if mc_pack else 0,
                    mc_device=str(mc_pack.get("mc_device", "")) if mc_pack else "",
                    rec_mc_mean_np=mc_pack.get("rec_mean", None),
                    rec_mc_std_np=mc_pack.get("rec_std", None),
                    rec_mc_cv_np=mc_pack.get("rec_cv", None),
                    deconv_mc_mean_np=mc_pack.get("deconv_mean", None),
                    deconv_mc_std_np=mc_pack.get("deconv_std", None),
                    deconv_mc_cv_np=mc_pack.get("deconv_cv", None),
                    mc_summary={
                        "rec_std_mean": mc_pack.get("rec_std_mean", None),
                        "rec_std_p95": mc_pack.get("rec_std_p95", None),
                        "deconv_std_mean": mc_pack.get("deconv_std_mean", None),
                        "deconv_std_p95": mc_pack.get("deconv_std_p95", None),
                    }
                    if mc_pack
                    else None,
                )
            except TypeError as e:
                print(
                    "[warn] uq_vis.save_debug_artifacts 不支持 uncertainty 参数（uq_vis.py 需要同步更新）。"
                    "将回退到旧版保存接口。错误：",
                    str(e),
                )
                dbg_metrics = save_debug_artifacts(
                    out_dir=out_dir,
                    args=args,
                    debug_path=str(debug_path),
                    crop_record=crop_record,
                    x_np=x_np,
                    rec_np=rec_np,
                    lp_np=lp_np,
                    emitter_np=emitter_np,
                    psf_np=psf_np,
                )

            note = f"debug_sample:{out_dir}"
            if mc_pack:
                note += f"|mc={mc_pack.get('mc_samples')} dev={mc_pack.get('mc_device')}"
                if "deconv_std_mean" in mc_pack:
                    note += f" de_std_mean={mc_pack['deconv_std_mean']:.4g} p95={mc_pack['deconv_std_p95']:.4g}"
                if "rec_std_mean" in mc_pack:
                    note += f" rec_std_mean={mc_pack['rec_std_mean']:.4g} p95={mc_pack['rec_std_p95']:.4g}"

            csv_append(
                metrics_csv,
                {
                    "time": time.time(),
                    "epoch": epoch,
                    "step": step,
                    "loss": "",
                    "rec_mse": dbg_metrics["rec_mse"],
                    "rec_mae": dbg_metrics["rec_mae"],
                    "psnr": dbg_metrics["psnr"],
                    "psf_tv": dbg_metrics["psf_tv"],
                    "lp_tv": dbg_metrics["lp_tv"],
                    "psf_center": dbg_metrics["psf_center"],
                    "psf_gauss": dbg_metrics["psf_gauss"],
                    "note": note,
                },
                header=csv_header,
            )
            print(f"[debug] saved to: {out_dir}")


if __name__ == "__main__":
    main()

# 例：
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run python uq_train.py --train_glob "/data/repo/SIMFormer/data/SIM-simulation/*/*/standard/train/*.tif" --batch_size 7 --epochs 500 --lr 1e-4 --ckpt_dir "./ckpts_uq3" --debug_every_epochs 0 --mc_dropout_train_samples 16 --mc_device gpu --mc_mask_ratio 0.0 --mc_disable_noise--uq_targets emitter
#
# 开启不确定度（debug 阶段 MC Dropout）示例：
#   --debug_every_epochs 5 --mc_dropout_train_samples 16 --mc_device cpu --mc_mask_ratio 0.0 --mc_disable_noise
