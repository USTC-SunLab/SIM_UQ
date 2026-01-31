# data_vis_sim_dataset_hr2.py
# -*- coding: utf-8 -*-
"""Visualize samples from simple_data_uq/sim_dataset_hr2."""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tifffile
    _HAS_TIFF = True
except Exception:
    _HAS_TIFF = False
from skimage.io import imread


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _read_tiff(path: str) -> np.ndarray:
    if _HAS_TIFF:
        return tifffile.imread(path)
    return imread(path)


def _to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        if arr.shape[0] in (1, 9) and arr.shape[1] > 8 and arr.shape[2] > 8:
            return arr
        if arr.shape[-1] in (1, 9):
            return np.transpose(arr, (2, 0, 1))
    return arr


def _percentile_norm(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> Tuple[float, float]:
    vmin = float(np.percentile(arr, p_low))
    vmax = float(np.percentile(arr, p_high))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_single(path: str, img: np.ndarray, title: str):
    _ensure_dir(os.path.dirname(path))
    vmin, vmax = _percentile_norm(img)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(img, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_pair(path: str, a: np.ndarray, b: np.ndarray, title_a: str, title_b: str):
    _ensure_dir(os.path.dirname(path))
    merged = np.concatenate([a.ravel(), b.ravel()])
    vmin, vmax = _percentile_norm(merged)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(a, vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title(title_a)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(b, vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title(title_b)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_frames_grid(path: str, frames: np.ndarray, title: str, ncols: int = 3):
    _ensure_dir(os.path.dirname(path))
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim == 4 and frames.shape[1] == 1:
        frames = frames[:, 0]
    n = int(frames.shape[0])
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    vmin, vmax = _percentile_norm(frames)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for i in range(nrows * ncols):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        if i < n:
            im = ax.imshow(frames[i], vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{title} {i}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def downsample_mean(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    if x.ndim == 2:
        x = x[None, ...]
        squeeze = True
    else:
        squeeze = False
    c, h, w = x.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    x = x[:, :h2, :w2]
    x = x.reshape(c, h2 // factor, factor, w2 // factor, factor).mean(axis=(2, 4))
    if squeeze:
        return x[0]
    return x


def main():
    p = argparse.ArgumentParser("Visualize sim_dataset_hr2 samples")
    p.add_argument("--data_dir", type=str, default="./simple_data_uq/sim_dataset_hr2")
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--index", type=int, default=0, help="index in sorted file list")
    p.add_argument("--out_dir", type=str, default="", help="output directory")
    args = p.parse_args()

    split_dir = os.path.join(args.data_dir, args.split)
    paths = sorted(glob(os.path.join(split_dir, "*.tif")))
    if not paths:
        raise RuntimeError(f"No tif found in {split_dir}")
    idx = int(np.clip(args.index, 0, len(paths) - 1))
    img_path = paths[idx]

    base = os.path.splitext(os.path.basename(img_path))[0]
    gt_dir = os.path.join(args.data_dir, f"{args.split}_gt")
    meta_dir = os.path.join(args.data_dir, f"{args.split}_meta")

    gt_path = os.path.join(gt_dir, f"{base}.tif")
    lp_path = os.path.join(gt_dir, f"{base}_lp.tif")
    meta_npz = os.path.join(meta_dir, f"{base}.npz")
    meta_json = os.path.join(meta_dir, f"{base}_config.json")

    out_dir = args.out_dir or os.path.join(args.data_dir, "vis", f"{args.split}_{base}")
    _ensure_dir(out_dir)

    x = _to_chw(_read_tiff(img_path).astype(np.float32))
    y = _to_chw(_read_tiff(gt_path).astype(np.float32)) if os.path.exists(gt_path) else None
    lp = _to_chw(_read_tiff(lp_path).astype(np.float32)) if os.path.exists(lp_path) else None

    sigma = None
    if os.path.exists(meta_npz):
        meta = np.load(meta_npz)
        if "noise_sigma" in meta:
            sigma = _to_chw(meta["noise_sigma"].astype(np.float32))

    # basic stats
    stats = {
        "img_path": img_path,
        "gt_path": gt_path,
        "lp_path": lp_path,
        "meta_npz": meta_npz,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape) if y is not None else None,
        "lp_shape": list(lp.shape) if lp is not None else None,
        "sigma_shape": list(sigma.shape) if sigma is not None else None,
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # visualize input frames
    save_frames_grid(os.path.join(out_dir, "input_frames.png"), x, "input")
    x_mean = x.mean(axis=0) if x.ndim == 3 else x
    save_single(os.path.join(out_dir, "input_mean.png"), x_mean, "Input mean")

    if y is not None:
        y_mean = y.mean(axis=0) if y.ndim == 3 else y[0]
        save_single(os.path.join(out_dir, "emitter_hr.png"), y_mean, "Emitter HR")
        # downsample for comparison
        hr_factor = int(round(y_mean.shape[-1] / x_mean.shape[-1])) if x_mean.shape[-1] > 0 else 1
        y_lr = downsample_mean(y_mean, max(1, hr_factor))
        save_pair(os.path.join(out_dir, "input_mean_vs_emitter_lr.png"), x_mean, y_lr, "Input mean", "Emitter LR")

    if lp is not None:
        save_frames_grid(os.path.join(out_dir, "lp_frames.png"), lp, "lp")
        lp_mean = lp.mean(axis=0) if lp.ndim == 3 else lp
        save_single(os.path.join(out_dir, "lp_mean.png"), lp_mean, "LP mean (HR)")

    if sigma is not None:
        sigma_mean = sigma.mean(axis=0) if sigma.ndim == 3 else sigma
        sigma_max = sigma.max(axis=0) if sigma.ndim == 3 else sigma
        save_single(os.path.join(out_dir, "sigma_mean.png"), sigma_mean, "Noise sigma mean")
        save_single(os.path.join(out_dir, "sigma_max.png"), sigma_max, "Noise sigma max")

    if os.path.exists(meta_json):
        try:
            with open(meta_json, "r") as f:
                cfg = json.load(f)
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
