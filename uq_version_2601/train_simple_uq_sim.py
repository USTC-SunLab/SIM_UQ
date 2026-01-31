# train_simple_uq_sim.py
# -*- coding: utf-8 -*-
"""Simple supervised SIM data UQ in JAX/Flax (predict emitter mean + logvar)."""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
from skimage.io import imread

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from flax import jax_utils

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow running from repo root or this subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from uq_data import min_max_norm, min_max_norm_full, random_crop_arrays


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        # already CHW
        if arr.shape[0] in (1, 9) and arr.shape[1] > 8 and arr.shape[2] > 8:
            return arr
        # HWC -> CHW
        if arr.shape[-1] in (1, 9):
            return np.transpose(arr, (2, 0, 1))
    return arr


def _center_crop(arr: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    if arr.ndim < 2:
        return arr
    h, w = int(arr.shape[-2]), int(arr.shape[-1])
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)
    h0 = max((h - crop_h) // 2, 0)
    w0 = max((w - crop_w) // 2, 0)
    return arr[..., h0 : h0 + crop_h, w0 : w0 + crop_w]


def _center_crop_pair_scaled(
    x: np.ndarray, gt: np.ndarray, crop_h: int, crop_w: int
) -> Tuple[np.ndarray, np.ndarray]:
    # x: (C,H,W), gt: (C,Hh,Wh) possibly HR
    x_h, x_w = int(x.shape[-2]), int(x.shape[-1])
    crop_h = min(crop_h, x_h)
    crop_w = min(crop_w, x_w)
    h0 = max((x_h - crop_h) // 2, 0)
    w0 = max((x_w - crop_w) // 2, 0)
    x_crop = x[..., h0 : h0 + crop_h, w0 : w0 + crop_w]

    g_h, g_w = int(gt.shape[-2]), int(gt.shape[-1])
    scale_h = float(g_h) / float(x_h) if x_h > 0 else 1.0
    scale_w = float(g_w) / float(x_w) if x_w > 0 else 1.0
    gh0 = int(round(h0 * scale_h))
    gw0 = int(round(w0 * scale_w))
    gh = int(round(crop_h * scale_h))
    gw = int(round(crop_w * scale_w))
    gh = max(1, min(gh, g_h - gh0))
    gw = max(1, min(gw, g_w - gw0))
    gt_crop = gt[..., gh0 : gh0 + gh, gw0 : gw0 + gw]
    return x_crop, gt_crop


def _normalize_gt(gt: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "none":
        return gt.astype(np.float32)
    if mode == "minmax_full":
        return min_max_norm_full(gt).astype(np.float32)
    if mode == "minmax":
        return min_max_norm(gt).astype(np.float32)
    return gt.astype(np.float32)


def _min_max_norm_with_scale(im: np.ndarray) -> Tuple[np.ndarray, float]:
    im = im.astype(np.float32)
    im_min = np.percentile(im, 0.01)
    im = im - im_min
    im_max = np.percentile(im, 99)
    if im_max > 0:
        im = im / im_max
    else:
        im = np.zeros_like(im, dtype=np.float32)
        im_max = 1.0
    return im.astype(np.float32), float(im_max)


def _replace_dir_token(path: str, token: str, repl: str, fallback_token: str, fallback_repl: str) -> str:
    if token in path:
        return path.replace(token, repl)
    if fallback_token in path:
        return path.replace(fallback_token, fallback_repl)
    return path


def derive_gt_paths(
    paths: List[str],
    gt_dir_token: str,
    gt_dir_repl: str,
    gt_emitter_suffix: str,
    gt_lp_suffix: str,
) -> List[Tuple[str, str]]:
    out = []
    for p in paths:
        base = _replace_dir_token(p, gt_dir_token, gt_dir_repl, "/val/", "/val_gt/")
        root, ext = os.path.splitext(base)
        emitter_path = root + gt_emitter_suffix + ext
        lp_path = root + gt_lp_suffix + ext
        out.append((emitter_path, lp_path))
    return out


def derive_meta_paths(paths: List[str], meta_dir_token: str, meta_dir_repl: str) -> List[str]:
    out = []
    for p in paths:
        base = _replace_dir_token(p, meta_dir_token, meta_dir_repl, "/val/", "/val_meta/")
        root, _ = os.path.splitext(base)
        out.append(root + ".npz")
    return out


def downsample_gt_np(gt: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if gt.ndim != 3:
        return gt
    th, tw = int(target_hw[0]), int(target_hw[1])
    gh, gw = int(gt.shape[-2]), int(gt.shape[-1])
    if (gh, gw) == (th, tw):
        return gt
    if gh % th == 0 and gw % tw == 0:
        kh = gh // th
        kw = gw // tw
        g = gt.reshape(gt.shape[0], th, kh, tw, kw)
        g = g.mean(axis=(2, 4))
        return g
    # fallback: resize with JAX
    g0 = jnp.asarray(gt[0], dtype=jnp.float32)
    g1 = jax.image.resize(g0, shape=(th, tw), method="linear")
    return np.asarray(g1)[None, ...]


def _corrcoef_flat(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    cov = jnp.mean(a * b)
    va = jnp.mean(a * a)
    vb = jnp.mean(b * b)
    return cov / (jnp.sqrt(va * vb) + float(eps))


def _to_nhwc(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.transpose(x, (0, 2, 3, 1))


def _to_nchw(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.transpose(x, (0, 3, 1, 2))


class ResBlock(nn.Module):
    features: int
    groups: int = 4

    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        h = nn.GroupNorm(num_groups=min(self.groups, self.features))(h)
        h = nn.relu(h)
        h = nn.Conv(self.features, (3, 3), padding="SAME")(h)
        h = nn.GroupNorm(num_groups=min(self.groups, self.features))(h)
        if x.shape[-1] != self.features:
            x = nn.Conv(self.features, (1, 1), padding="SAME")(x)
        return nn.relu(h + x)


class UNetUQ(nn.Module):
    base: int = 32
    in_ch: int = 9

    @nn.compact
    def __call__(self, x):
        # x: NHWC
        e1 = ResBlock(self.base)(x)
        e2 = ResBlock(self.base * 2)(nn.max_pool(e1, (2, 2), (2, 2), padding="SAME"))
        e3 = ResBlock(self.base * 4)(nn.max_pool(e2, (2, 2), (2, 2), padding="SAME"))
        e4 = ResBlock(self.base * 8)(nn.max_pool(e3, (2, 2), (2, 2), padding="SAME"))

        b = ResBlock(self.base * 16)(nn.max_pool(e4, (2, 2), (2, 2), padding="SAME"))

        d4 = nn.ConvTranspose(self.base * 8, (2, 2), strides=(2, 2))(b)
        d4 = ResBlock(self.base * 8)(jnp.concatenate([d4, e4], axis=-1))
        d3 = nn.ConvTranspose(self.base * 4, (2, 2), strides=(2, 2))(d4)
        d3 = ResBlock(self.base * 4)(jnp.concatenate([d3, e3], axis=-1))
        d2 = nn.ConvTranspose(self.base * 2, (2, 2), strides=(2, 2))(d3)
        d2 = ResBlock(self.base * 2)(jnp.concatenate([d2, e2], axis=-1))
        d1 = nn.ConvTranspose(self.base, (2, 2), strides=(2, 2))(d2)
        d1 = ResBlock(self.base)(jnp.concatenate([d1, e1], axis=-1))

        mu = nn.Conv(1, (1, 1), name="mu_head")(d1)
        logvar = nn.Conv(1, (1, 1), name="logvar_head")(d1)
        return mu, logvar


def gaussian_nll(mu: jnp.ndarray, logvar: jnp.ndarray, target: jnp.ndarray, clamp_min: float, clamp_max: float):
    logvar = jnp.clip(logvar, clamp_min, clamp_max)
    var = jnp.exp(logvar) + 1e-6
    diff = mu - target
    nll = 0.5 * (diff * diff / var + logvar)
    return jnp.mean(nll), var, diff


def save_pair(path: str, a: np.ndarray, b: np.ndarray, title_a: str, title_b: str):
    _ensure_dir(os.path.dirname(path))
    merged = np.concatenate([a.ravel(), b.ravel()])
    vmin, vmax = np.percentile(merged, [1, 99])
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


def save_pair_separate(path: str, a: np.ndarray, b: np.ndarray, title_a: str, title_b: str):
    _ensure_dir(os.path.dirname(path))
    vmin_a, vmax_a = np.percentile(a.ravel(), [1, 99])
    vmin_b, vmax_b = np.percentile(b.ravel(), [1, 99])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(a, vmin=vmin_a, vmax=vmax_a, aspect="auto")
    axes[0].set_title(title_a)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(b, vmin=vmin_b, vmax=vmax_b, aspect="auto")
    axes[1].set_title(title_b)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_triplet(path: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, titles: Tuple[str, str, str]):
    _ensure_dir(os.path.dirname(path))
    merged = np.concatenate([a.ravel(), b.ravel(), c.ravel()])
    vmin, vmax = np.percentile(merged, [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, arr in enumerate([a, b, c]):
        im = axes[i].imshow(arr, vmin=vmin, vmax=vmax, aspect="auto")
        axes[i].set_title(titles[i])
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_single(path: str, a: np.ndarray, title: str):
    _ensure_dir(os.path.dirname(path))
    vmin, vmax = np.percentile(a.ravel(), [1, 99])
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(a, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_frames_grid(path: str, frames: np.ndarray, ncols: int = 3):
    _ensure_dir(os.path.dirname(path))
    n = int(frames.shape[0])
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    vmin, vmax = np.percentile(frames.ravel(), [1, 99])
    for i in range(nrows * ncols):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        if i < n:
            im = ax.imshow(frames[i], vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"frame {i}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _reduce_sigma_map(sigma: np.ndarray, mode: str) -> np.ndarray:
    sigma = _to_chw(sigma)
    mode = str(mode).strip().lower()
    if sigma.ndim == 2:
        sigma = sigma[None, ...]
    if sigma.shape[0] <= 1:
        return sigma.astype(np.float32)
    if mode in ("max", "peak"):
        sigma = sigma.max(axis=0, keepdims=True)
    else:
        sigma = sigma.mean(axis=0, keepdims=True)
    return sigma.astype(np.float32)


def _normalize_sigma_map(sigma: np.ndarray, mode: str, input_scale: float) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "none":
        return sigma.astype(np.float32)
    if mode == "input_scale":
        return (sigma / (float(input_scale) + 1e-8)).astype(np.float32)
    if mode == "minmax_full":
        return min_max_norm_full(sigma).astype(np.float32)
    if mode == "minmax":
        return min_max_norm(sigma).astype(np.float32)
    return sigma.astype(np.float32)


def calibration_curve(pred_var: np.ndarray, err2: np.ndarray, bins: int = 10):
    pv = pred_var.reshape(-1)
    e2 = err2.reshape(-1)
    idx = np.argsort(pv)
    pv = pv[idx]
    e2 = e2[idx]
    n = pv.size
    edges = np.linspace(0, n, bins + 1, dtype=int)
    xs, ys = [], []
    for i in range(bins):
        s = edges[i]
        t = edges[i + 1]
        if t <= s:
            continue
        xs.append(float(np.mean(pv[s:t])))
        ys.append(float(np.mean(e2[s:t])))
    return np.asarray(xs), np.asarray(ys)


def plot_calibration(curve: Tuple[np.ndarray, np.ndarray], out_path: str, title: str):
    x, y = curve
    mx = max(float(np.max(x)), float(np.max(y))) if x.size and y.size else 1.0
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(x, y, marker="o", linewidth=1.2, label="empirical")
    ax.plot([0, mx], [0, mx], linestyle="--", color="gray", label="ideal")
    ax.set_title(title)
    ax.set_xlabel("predicted variance")
    ax.set_ylabel("empirical MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def csv_append(path: str, row: Dict[str, float], header: List[str]):
    _ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def plot_loss_curves(metrics_csv: str, out_path: str):
    if not os.path.exists(metrics_csv):
        return
    epochs: List[int] = []
    train_loss: List[float] = []
    val_loss: List[float] = []
    with open(metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                epochs.append(int(float(r.get("epoch", 0))))
                train_loss.append(float(r.get("train_loss", "nan")))
                val_loss.append(float(r.get("val_loss", "nan")))
            except Exception:
                continue
    if not epochs:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epochs, train_loss, label="train_loss")
    if any(np.isfinite(val_loss)):
        ax.plot(epochs, val_loss, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


class SimpleSimUQDataset:
    def __init__(
        self,
        paths: List[str],
        gt_paths: List[Tuple[str, str]],
        crop_size: Tuple[int, int],
        gt_norm: str,
        meta_paths: List[str] | None = None,
        sigma_reduce: str = "mean",
        sigma_norm: str = "input_scale",
        log_fn=print,
    ):
        if not paths:
            raise RuntimeError("No training files found")
        if len(paths) != len(gt_paths):
            raise RuntimeError("paths and gt_paths must have the same length")
        if meta_paths is not None and len(meta_paths) != len(paths):
            raise RuntimeError("meta_paths must match paths length")
        self.paths = paths
        self.gt_paths = gt_paths
        self.crop_size = tuple(int(x) for x in crop_size)
        self.gt_norm = gt_norm
        self.meta_paths = meta_paths
        self.sigma_reduce = sigma_reduce
        self.sigma_norm = sigma_norm
        self.has_sigma = False
        if meta_paths:
            missing = [p for p in meta_paths if not os.path.exists(p)]
            if missing:
                log_fn(f"[warn] {len(missing)} meta npz missing, sigma supervision disabled")
                self.meta_paths = None
            else:
                self.has_sigma = True
        log_fn(f"num of files: {len(paths)}, first path: {paths[0]}")

    def __len__(self) -> int:
        return len(self.paths)

    def _load_one(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.paths[idx]
        gt_em, _ = self.gt_paths[idx]
        x = imread(p).astype(np.float32)
        x = _to_chw(x)
        x, x_scale = _min_max_norm_with_scale(x)
        em = imread(gt_em).astype(np.float32)
        em = _to_chw(em)
        em = _normalize_gt(em, self.gt_norm)
        sigma = None
        if self.meta_paths:
            meta_path = self.meta_paths[idx]
            if meta_path and os.path.exists(meta_path):
                meta = np.load(meta_path)
                if "noise_sigma" in meta:
                    sigma = meta["noise_sigma"].astype(np.float32)
                    sigma = _reduce_sigma_map(sigma, self.sigma_reduce)
                    sigma = _normalize_sigma_map(sigma, self.sigma_norm, x_scale)
        arrays = [x, em]
        if sigma is not None:
            arrays.append(sigma)
        cropped = random_crop_arrays(arrays, *self.crop_size)
        if sigma is not None:
            x, em, sigma = cropped
        else:
            x, em = cropped
        em = downsample_gt_np(em, x.shape[-2:])
        if sigma is None:
            sigma = np.zeros((1, x.shape[-2], x.shape[-1]), dtype=np.float32)
        return x.astype(np.float32), em.astype(np.float32), sigma.astype(np.float32)

    def get_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, ss = [], [], []
        for i in indices:
            x, y, s = self._load_one(int(i))
            xs.append(x)
            ys.append(y)
            ss.append(s)
        return np.stack(xs, axis=0), np.stack(ys, axis=0), np.stack(ss, axis=0)


class TrainState(train_state.TrainState):
    pass


def _load_debug_sample(
    img_path: str,
    gt_path: str,
    meta_path: str,
    crop_size: Tuple[int, int],
    gt_norm: str,
    sigma_reduce: str,
    sigma_norm: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = imread(img_path).astype(np.float32)
    x = _to_chw(x)
    x, x_scale = _min_max_norm_with_scale(x)
    gt = imread(gt_path).astype(np.float32)
    gt = _to_chw(gt)
    gt = _normalize_gt(gt, gt_norm)
    x, gt = _center_crop_pair_scaled(x, gt, int(crop_size[0]), int(crop_size[1]))
    gt = downsample_gt_np(gt, x.shape[-2:])

    std_gt = None
    if meta_path and os.path.exists(meta_path):
        meta = np.load(meta_path)
        if "noise_sigma" in meta:
            sigma = meta["noise_sigma"].astype(np.float32)
            sigma = _reduce_sigma_map(sigma, sigma_reduce)
            sigma = _normalize_sigma_map(sigma, sigma_norm, x_scale)
            sigma = _center_crop(sigma, int(crop_size[0]), int(crop_size[1]))
            std_gt = sigma[0] if sigma.ndim == 3 else sigma
    return x.astype(np.float32), gt.astype(np.float32), std_gt


def main():
    p = argparse.ArgumentParser("Simple SIM data UQ (JAX)")
    p.add_argument("--train_glob", type=str, required=True)
    p.add_argument("--val_glob", type=str, default="")
    p.add_argument("--gt_dir_token", type=str, default="/train/")
    p.add_argument("--gt_dir_repl", type=str, default="/train_gt/")
    p.add_argument("--gt_emitter_suffix", type=str, default="")
    p.add_argument("--gt_lp_suffix", type=str, default="_lp")
    p.add_argument("--meta_dir_token", type=str, default="/train/")
    p.add_argument("--meta_dir_repl", type=str, default="/train_meta/")
    p.add_argument("--gt_norm", type=str, default="minmax_full", choices=["none", "minmax", "minmax_full"])
    p.add_argument("--crop_size", type=int, nargs=2, default=(80, 80))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--logvar_min", type=float, default=-8.0)
    p.add_argument("--logvar_max", type=float, default=3.0)
    p.add_argument("--var_reg", type=float, default=1e-4)
    p.add_argument("--sigma_weight", type=float, default=0.2, help="(deprecated) ignored; sigma is not supervised")
    p.add_argument("--sigma_weight_end", type=float, default=0.6, help="(deprecated) ignored")
    p.add_argument("--sigma_warmup_epochs", type=int, default=20, help="(deprecated) ignored")
    p.add_argument("--sigma_reduce", type=str, default="mean", choices=["mean", "max"])
    p.add_argument("--sigma_norm", type=str, default="input_scale", choices=["input_scale", "none", "minmax", "minmax_full"])
    p.add_argument("--ckpt_dir", type=str, default="./uq_version_2601/ckpts")
    p.add_argument("--debug_every", type=int, default=5)
    p.add_argument("--calib_bins", type=int, default=20)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--debug_batch_size", type=int, default=4)
    args = p.parse_args()

    set_seed(args.seed)

    devices = jax.devices()
    n_devices = jax.local_device_count()
    print(f"[jax] devices={devices}")
    print(f"[jax] local_device_count={n_devices}")

    if args.batch_size % n_devices != 0:
        raise ValueError(f"batch_size={args.batch_size} must be divisible by n_devices={n_devices}")

    train_paths = sorted(glob(args.train_glob))
    if not train_paths:
        raise RuntimeError(f"No files found for train_glob: {args.train_glob}")

    if args.val_glob:
        val_paths = sorted(glob(args.val_glob))
        if not val_paths:
            print(f"[warn] No files found for val_glob: {args.val_glob}. Falling back to split from train.")
            val_paths = []
    else:
        val_paths = []

    if not val_paths:
        idx = np.random.permutation(len(train_paths))
        n_val = int(len(train_paths) * float(args.val_ratio))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_paths = [train_paths[i] for i in val_idx]
        train_paths = [train_paths[i] for i in train_idx]

    train_gt_paths = derive_gt_paths(
        train_paths,
        args.gt_dir_token,
        args.gt_dir_repl,
        args.gt_emitter_suffix,
        args.gt_lp_suffix,
    )
    val_gt_paths = derive_gt_paths(
        val_paths,
        args.gt_dir_token,
        args.gt_dir_repl,
        args.gt_emitter_suffix,
        args.gt_lp_suffix,
    )

    train_meta_paths = derive_meta_paths(train_paths, args.meta_dir_token, args.meta_dir_repl)
    val_meta_paths = derive_meta_paths(val_paths, args.meta_dir_token, args.meta_dir_repl)

    train_ds = SimpleSimUQDataset(
        train_paths,
        train_gt_paths,
        args.crop_size,
        args.gt_norm,
        meta_paths=train_meta_paths,
        sigma_reduce=args.sigma_reduce,
        sigma_norm=args.sigma_norm,
    )
    val_ds = SimpleSimUQDataset(
        val_paths,
        val_gt_paths,
        args.crop_size,
        args.gt_norm,
        meta_paths=val_meta_paths,
        sigma_reduce=args.sigma_reduce,
        sigma_norm=args.sigma_norm,
    )

    use_sigma = bool(train_ds.has_sigma and val_ds.has_sigma)
    if not use_sigma:
        print("[warn] sigma eval disabled (no meta/noise_sigma found in both train/val)")
    if float(args.sigma_weight) != 0.0 or float(args.sigma_weight_end) != 0.0:
        print("[warn] sigma supervision is disabled; sigma_weight args are ignored")

    model = UNetUQ(base=args.base)
    rng = jax.random.PRNGKey(args.seed)
    dummy = jnp.zeros((1, int(args.crop_size[0]), int(args.crop_size[1]), 9), dtype=jnp.float32)
    params = model.init(rng, dummy)["params"]

    tx = optax.adam(args.lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state_rep = jax_utils.replicate(state)

    def loss_fn(params, x, y, sigma):
        mu, logvar = model.apply({"params": params}, x)
        nll, var, diff = gaussian_nll(mu, logvar, y, args.logvar_min, args.logvar_max)
        reg = jnp.mean(logvar * logvar)
        loss = nll + args.var_reg * reg

        mse = jnp.mean(diff * diff)
        var_mean = jnp.mean(var)
        calib_ratio = mse / (var_mean + 1e-8)
        corr = _corrcoef_flat(var, diff * diff)
        if use_sigma:
            std = jnp.sqrt(var)
            sigma_mae = jnp.mean(jnp.abs(std - sigma))
            sigma_corr = _corrcoef_flat(var, sigma * sigma)
        else:
            sigma_mae = jnp.nan
            sigma_corr = jnp.nan
        metrics = {
            "loss": loss,
            "mse": mse,
            "nll": nll,
            "var_mean": var_mean,
            "calib_ratio": calib_ratio,
            "corr": corr,
            "sigma_mae": sigma_mae,
            "sigma_corr": sigma_corr,
        }
        return loss, metrics

    def train_step(state, x, y, sigma):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, x, y, sigma)
        grads = jax.lax.pmean(grads, axis_name="devices")
        metrics = jax.lax.pmean(metrics, axis_name="devices")
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics

    def eval_step(params, x, y, sigma):
        _, metrics = loss_fn(params, x, y, sigma)
        metrics = jax.lax.pmean(metrics, axis_name="devices")
        return metrics

    p_train_step = jax.pmap(train_step, axis_name="devices", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, axis_name="devices")

    metrics_csv = os.path.join(args.ckpt_dir, "metrics.csv")
    header = [
        "time",
        "epoch",
        "train_loss",
        "train_mse",
        "train_nll",
        "train_var_mean",
        "train_calib_ratio",
        "train_corr",
        "train_sigma_mae",
        "train_sigma_corr",
        "val_loss",
        "val_mse",
        "val_nll",
        "val_var_mean",
        "val_calib_ratio",
        "val_corr",
        "val_sigma_mae",
        "val_sigma_corr",
    ]

    _ensure_dir(args.ckpt_dir)

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        idx = np.random.permutation(len(train_ds))
        n_batches = len(idx) // args.batch_size
        sums_train = {k: 0.0 for k in ("loss", "mse", "nll", "var_mean", "calib_ratio", "corr", "sigma_mae", "sigma_corr")}
        total_train = 0
        for bi in range(n_batches):
            batch_idx = idx[bi * args.batch_size : (bi + 1) * args.batch_size]
            x_np, y_np, s_np = train_ds.get_batch(batch_idx)
            x = jnp.asarray(x_np)
            y = jnp.asarray(y_np)
            sigma = jnp.asarray(s_np)
            x = _to_nhwc(x)
            y = _to_nhwc(y)
            sigma = _to_nhwc(sigma)
            x = x.reshape((n_devices, -1) + x.shape[1:])
            y = y.reshape((n_devices, -1) + y.shape[1:])
            sigma = sigma.reshape((n_devices, -1) + sigma.shape[1:])
            state_rep, metrics = p_train_step(state_rep, x, y, sigma)
            m = {k: float(jax.device_get(v)[0]) for k, v in metrics.items()}
            for k in sums_train:
                sums_train[k] += m[k] * x_np.shape[0]
            total_train += x_np.shape[0]

        train_metrics = {k: (sums_train[k] / max(total_train, 1)) for k in sums_train}

        # ---- val ----
        idx_val = np.arange(len(val_ds))
        n_val_batches = int(math.ceil(len(idx_val) / args.batch_size))
        sums_val = {k: 0.0 for k in ("loss", "mse", "nll", "var_mean", "calib_ratio", "corr", "sigma_mae", "sigma_corr")}
        total_val = 0
        for bi in range(n_val_batches):
            batch_idx = idx_val[bi * args.batch_size : (bi + 1) * args.batch_size]
            if batch_idx.size == 0:
                continue
            # pad to full batch for pmap
            if batch_idx.size < args.batch_size:
                pad = np.random.choice(idx_val, size=(args.batch_size - batch_idx.size), replace=True)
                batch_idx = np.concatenate([batch_idx, pad], axis=0)
            x_np, y_np, s_np = val_ds.get_batch(batch_idx)
            x = jnp.asarray(x_np)
            y = jnp.asarray(y_np)
            sigma = jnp.asarray(s_np)
            x = _to_nhwc(x)
            y = _to_nhwc(y)
            sigma = _to_nhwc(sigma)
            x = x.reshape((n_devices, -1) + x.shape[1:])
            y = y.reshape((n_devices, -1) + y.shape[1:])
            sigma = sigma.reshape((n_devices, -1) + sigma.shape[1:])
            metrics = p_eval_step(state_rep.params, x, y, sigma)
            m = {k: float(jax.device_get(v)[0]) for k, v in metrics.items()}
            for k in sums_val:
                sums_val[k] += m[k] * x_np.shape[0]
            total_val += x_np.shape[0]

        val_metrics = {k: (sums_val[k] / max(total_val, 1)) for k in sums_val}

        row = {
            "time": time.time(),
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_nll": train_metrics["nll"],
            "train_var_mean": train_metrics["var_mean"],
            "train_calib_ratio": train_metrics["calib_ratio"],
            "train_corr": train_metrics["corr"],
            "train_sigma_mae": train_metrics.get("sigma_mae"),
            "train_sigma_corr": train_metrics.get("sigma_corr"),
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_nll": val_metrics["nll"],
            "val_var_mean": val_metrics["var_mean"],
            "val_calib_ratio": val_metrics["calib_ratio"],
            "val_corr": val_metrics["corr"],
            "val_sigma_mae": val_metrics.get("sigma_mae"),
            "val_sigma_corr": val_metrics.get("sigma_corr"),
        }
        csv_append(metrics_csv, row, header)
        print(f"epoch {epoch:03d} | train_loss {train_metrics['loss']:.4f} | val_loss {val_metrics['loss']:.4f}")

        if args.debug_every > 0 and epoch % args.debug_every == 0:
            out_dir = os.path.join(args.ckpt_dir, "debug", f"epoch_{epoch:03d}")
            _ensure_dir(out_dir)
            # deterministic debug sample (center crop) for aligned std_gt
            dbg_path = val_paths[0]
            dbg_gt = derive_gt_paths(
                [dbg_path],
                args.gt_dir_token,
                args.gt_dir_repl,
                args.gt_emitter_suffix,
                args.gt_lp_suffix,
            )[0][0]
            dbg_meta = _replace_dir_token(
                dbg_path, args.meta_dir_token, args.meta_dir_repl, "/val/", "/val_meta/"
            )
            dbg_meta = os.path.splitext(dbg_meta)[0] + ".npz"
            x_dbg, y_dbg, std_gt = _load_debug_sample(
                dbg_path, dbg_gt, dbg_meta, args.crop_size, args.gt_norm, args.sigma_reduce, args.sigma_norm
            )
            x_np = x_dbg[None, ...]
            y_np = y_dbg[None, ...]
            x = jnp.asarray(x_np)
            y = jnp.asarray(y_np)
            x_n = _to_nhwc(x)
            y_n = _to_nhwc(y)
            state_single = jax_utils.unreplicate(state_rep)
            mu, logvar = model.apply({"params": state_single.params}, x_n)
            logvar = jnp.clip(logvar, args.logvar_min, args.logvar_max)
            var = jnp.exp(logvar) + 1e-6
            std = jnp.sqrt(var)
            mu_np = np.array(jax.device_get(mu))
            std_np = np.array(jax.device_get(std))

            x0 = x_np[0]
            y0 = y_np[0, 0]
            mu0 = mu_np[0, ..., 0]
            std0 = std_np[0, ..., 0]
            x_mean = np.mean(x0, axis=0)

            save_frames_grid(os.path.join(out_dir, "input_frames.png"), x0, ncols=3)
            save_triplet(
                os.path.join(out_dir, "input_pred_gt.png"),
                x_mean,
                mu0,
                y0,
                ("Input mean", "Pred mean", "GT emitter"),
            )
            save_pair(os.path.join(out_dir, "pred_gt.png"), mu0, y0, "Pred mean", "GT emitter")
            save_pair_separate(
                os.path.join(out_dir, "pred_gt_separate.png"),
                mu0,
                y0,
                "Pred mean (own scale)",
                "GT emitter (own scale)",
            )
            save_single(os.path.join(out_dir, "pred_std.png"), std0, "Pred std")
            if std_gt is not None:
                err0 = np.abs(mu0 - y0)
                save_single(os.path.join(out_dir, "std_gt.png"), std_gt, "GT std (noise_sigma mean)")
                save_pair(
                    os.path.join(out_dir, "std_pred_vs_gt.png"),
                    std0,
                    std_gt,
                    "Pred std",
                    "GT std",
                )
                save_pair(
                    os.path.join(out_dir, "error_vs_sigma.png"),
                    err0,
                    std_gt,
                    "Abs error",
                    "GT std",
                )

            # calibration curve on val
            all_var = []
            all_err2 = []
            for start in range(0, len(val_ds), args.batch_size):
                batch_idx = np.arange(start, min(start + args.batch_size, len(val_ds)))
                if batch_idx.size == 0:
                    continue
                x_np, y_np, _ = val_ds.get_batch(batch_idx)
                x_n = _to_nhwc(jnp.asarray(x_np))
                y_n = _to_nhwc(jnp.asarray(y_np))
                mu, logvar = model.apply({"params": state_single.params}, x_n)
                logvar = jnp.clip(logvar, args.logvar_min, args.logvar_max)
                var = jnp.exp(logvar) + 1e-6
                diff = mu - y_n
                all_var.append(np.array(jax.device_get(var)))
                all_err2.append(np.array(jax.device_get(diff * diff)))
            if all_var and all_err2:
                all_var = np.concatenate(all_var, axis=0)
                all_err2 = np.concatenate(all_err2, axis=0)
                curve = calibration_curve(all_var, all_err2, bins=int(args.calib_bins))
                plot_calibration(curve, os.path.join(out_dir, "calibration.png"), "Calibration (val)")

        plot_loss_curves(metrics_csv, os.path.join(args.ckpt_dir, "loss_curve.png"))

    print("[done] training finished")


if __name__ == "__main__":
    main()
