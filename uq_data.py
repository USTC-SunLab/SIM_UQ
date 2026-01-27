# file_name: uq_data.py
import random
from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import torch
from tqdm import tqdm
from skimage.io import imread
import os
import numpy as np
import matplotlib.pyplot as plt
from uq_image_compare import compare_images, to_2d

def vis_compare_arrays(arr_list, save_path, cmap="viridis",
                  metric=None, norm="pminmax", norm_kwargs=None, ref_idx=0):
    """
    Visualize last two dims of arrays in a list as heatmaps.
    Also (optionally) compare each to ref image (arr_list[ref_idx]).

    Args:
        arr_list (list): list of np.ndarray or torch.Tensor, each >= 2D
        save_path (str): output image path
        cmap (str): matplotlib colormap
        metric (str|None): e.g. "mse" / "psnr" / "ssim" / "ms-ssim". None -> no compare
        norm (str): "pminmax" / "minmax" / "zscore" / "none"
        norm_kwargs (dict|None): e.g. {"p": (1, 99)}
        ref_idx (int): index of reference image in arr_list
    """
    n = len(arr_list)
    fig, axes = plt.subplots(n, 1, figsize=(5, 4 * n), squeeze=False)
    axes = axes[:, 0]

    norm_kwargs = norm_kwargs or {}
    ref = to_2d(arr_list[ref_idx])

    for i, arr in enumerate(arr_list):
        arr2d = to_2d(arr)

        im = axes[i].imshow(arr2d, cmap=cmap, aspect="auto")
        title = f"Array {i}: {arr2d.shape}"

        if i == ref_idx:
            title += " (ref)"
        elif metric:
            score = compare_images(ref, arr2d, metric=metric, norm=norm, **norm_kwargs)
            m = metric.lower()
            if m == "psnr":
                title += f" | PSNR vs {ref_idx}: {score:.2f} dB"
            else:
                title += f" | {metric} vs {ref_idx}: {score:.4g}"

        axes[i].set_title(title)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[vis_2d_arrays] saved to: {save_path}")

def vis_2d_arrays(arr_list, save_path, cmap="viridis"):
    """
    Visualize last two dims of arrays in a list as heatmaps.

    Args:
        arr_list (list): list of np.ndarray or torch.Tensor, each >= 2D
        save_path (str): output image path
        cmap (str): matplotlib colormap
    """
    n = len(arr_list)
    fig, axes = plt.subplots(n, 1, figsize=(5, 4 * n), squeeze=False)
    axes = axes[:,0]

    for i, arr in enumerate(arr_list):
        # torch -> numpy
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        im = axes[i].imshow(arr, cmap=cmap, aspect="auto")
        axes[i].set_title(f"Array {i}: {arr.shape}")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[vis_2d_arrays] saved to: {save_path}")

def min_max_norm(im):
    # Min-max normalization with percentiles
    im_min = np.percentile(im, 0.01)
    im = im - im_min
    im_max = np.percentile(im, 99)
    if im_max > 0:
        im /= im_max
    return im
def random_crop_arrays(arrays, h, w):
    # 用来处理sim emitter和lp，由于emitter和lp与sim的形状不同，故需要对齐形状
    scales = [int(x.shape[-1]/arrays[0].shape[-1]) for x in arrays]
    H, W = arrays[0].shape[-2:]
    assert h <= H and w <= W
    h0 = np.random.randint(0, H - h + 1)
    w0 = np.random.randint(0, W - w + 1)

    return [array[..., h0*scale:h0*scale+h*scale, w0*scale:w0*scale+w*scale] for (array,scale) in zip(arrays,scales)]

class dataset_2d_sim_supervised(torch.utils.data.Dataset):
    """
    2D supervised dataset for SIM-like input:
      - input: 9-channel/9-frame grayscale stack, same HxW as GT
      - gt:    1-channel super-res ground truth image
    目标是自监督学习，只需要输出SIM原图即可，但是为了能够衡量学习效果，增加输出gt的功能
    Returns:
      dict(img=(9,H,W), gt=(1,H,W)) as float32 torch tensors
    """

    def __init__(
        self,
        paths,  # glob pattern
        crop_size: Tuple[int, int] = (80,80),
        use_gt: bool = False, # gt_paths 双保险
        gt_paths = None,
        # normalize: bool = True,
        log_fn = print
    ):
        super().__init__()
        # accessory
        if use_gt and (gt_paths is None or len(gt_paths)!=len(paths)):
            w_save(str(paths),str(gt_paths))
            import sys; sys.exit()
        log_fn(f"num of files: {len(paths)}, first path: {paths[0]}")
        self.use_gt = use_gt
        self.crop_size = crop_size

        self.imgs,self.emitter_gts,self.lp_gts = [],[],[]
        for index in tqdm(range(len(paths)), desc="Indexing dataset"):
            img = min_max_norm(imread(paths[index]).astype(np.float32))
            self.imgs.append(img)
            if use_gt:
                self.emitter_gts.append(imread(gt_paths[index][0]).astype(np.float32)[None])
                self.lp_gts.append(imread(gt_paths[index][1]).astype(np.float32))
    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        data = [self.imgs[idx],self.emitter_gts[idx],self.lp_gts[idx]] if self.use_gt else [self.imgs[idx]]
        croped_data = random_crop_arrays(data, *self.crop_size)
        # croped_data = data # debug 用以检查crop前各张量的形状
        return croped_data
if __name__ == "__main__":
    import glob
    train_glob = "./data/SIM-simulation/*/*/*/train/*.tif"
    paths = glob.glob(train_glob)[:2]
    gt_paths = [(x.replace("/train/",'/train_gt/'),x.replace("/train/",'/train_gt/').replace(".tif","_lp.tif")) for x in paths]
    # dataset = dataset_2d_sim_supervised(paths)
    dataset_with_gt = dataset_2d_sim_supervised(paths,use_gt=True,gt_paths=gt_paths)
    data_item = dataset_with_gt[0]
    print(data_item[0].shape,data_item[1].shape,data_item[2].shape)
    vis_2d_arrays([data_item[0][0],data_item[1][0],data_item[2][0],],'tmp1.jpg')
