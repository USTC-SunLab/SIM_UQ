# gen_uq_sup_dataset.py
# -*- coding: utf-8 -*-
"""
Generate a synthetic SIM dataset for supervised UQ training.
- Input: 9-frame SIM stacks (LR)
- GT: emitter (HR) + light patterns (HR)
- Adds spatially varying (heteroscedastic) noise to the SIM inputs

Folder layout (default):
  out_dir/
    train/         sample_xxxxxx.tif            (9,H_lr,W_lr)
    train_gt/      sample_xxxxxx.tif            (H_hr,W_hr) emitter
    train_gt/      sample_xxxxxx_lp.tif         (9,H_hr,W_hr) light pattern
    train_meta/    sample_xxxxxx.npz            (noise maps, params)
"""

from __future__ import annotations

import os
import argparse
from typing import Tuple

import numpy as np
from skimage.io import imsave

try:
    import tifffile
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    r = size // 2
    ax = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= (np.sum(k) + 1e-8)
    return k


def _fft_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = img.shape
    k = np.zeros((h, w), dtype=np.float32)
    kh, kw = kernel.shape
    r0 = kh // 2
    r1 = kw // 2
    k[:kh, :kw] = kernel
    k = np.fft.ifftshift(k)
    out = np.fft.irfft2(np.fft.rfft2(img) * np.fft.rfft2(k)).astype(np.float32)
    return out


def _downsample_avg(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    img = img[:h2, :w2]
    img = img.reshape(h2 // factor, factor, w2 // factor, factor)
    return img.mean(axis=(1, 3)).astype(np.float32)


def _make_emitter(hr_h: int, hr_w: int, rng: np.random.Generator) -> np.ndarray:
    n_blobs = int(rng.integers(6, 18))
    emitter = np.zeros((hr_h, hr_w), dtype=np.float32)
    yy, xx = np.mgrid[0:hr_h, 0:hr_w]
    for _ in range(n_blobs):
        cy = float(rng.uniform(0, hr_h))
        cx = float(rng.uniform(0, hr_w))
        sigma = float(rng.uniform(3.0, 18.0))
        amp = float(rng.uniform(0.6, 1.2))
        blob = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
        emitter += blob
    emitter -= emitter.min()
    emitter /= (emitter.max() + 1e-8)
    return emitter


def _make_light_patterns(hr_h: int, hr_w: int, rng: np.random.Generator) -> np.ndarray:
    # 3 angles * 3 phases = 9 patterns
    angles = [0.0, 60.0, 120.0]
    phases = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
    yy, xx = np.mgrid[0:hr_h, 0:hr_w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    freq = float(rng.uniform(0.02, 0.06))  # cycles per pixel
    mod = float(rng.uniform(0.7, 0.95))

    patterns = []
    for ang in angles:
        theta = np.deg2rad(ang)
        kx = np.cos(theta) * freq
        ky = np.sin(theta) * freq
        phase0 = float(rng.uniform(0, 2 * np.pi))
        for ph in phases:
            phase = phase0 + ph
            pat = 1.0 + mod * np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phase)
            patterns.append(pat.astype(np.float32))

    lp = np.stack(patterns, axis=0)  # (9, H, W)
    lp -= lp.min()
    lp /= (lp.max() + 1e-8)
    return lp


def _make_noise_mask(lr_h: int, lr_w: int, rng: np.random.Generator) -> np.ndarray:
    n_blobs = int(rng.integers(2, 6))
    mask = np.zeros((lr_h, lr_w), dtype=np.float32)
    yy, xx = np.mgrid[0:lr_h, 0:lr_w]
    for _ in range(n_blobs):
        cy = float(rng.uniform(0, lr_h))
        cx = float(rng.uniform(0, lr_w))
        sigma = float(rng.uniform(4.0, 14.0))
        amp = float(rng.uniform(0.5, 1.2))
        blob = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
        mask += blob
    mask -= mask.min()
    mask /= (mask.max() + 1e-8)
    return mask


def _save_tif(path: str, arr: np.ndarray):
    if _HAS_TIFFFILE:
        tifffile.imwrite(path, arr, compression="zlib")
    else:
        imsave(path, arr)


def _to_uint16(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a -= a.min()
    a /= (a.max() + 1e-8)
    a = np.clip(a, 0.0, 1.0)
    return (a * 65535.0).astype(np.uint16)


def generate_dataset(
    out_dir: str,
    n_samples: int,
    lr_size: Tuple[int, int],
    hr_factor: int,
    seed: int,
    psf_sigma_range: Tuple[float, float],
    noise_base: float,
    noise_gain: float,
    noise_region_amp: float,
):
    rng = np.random.default_rng(seed)

    train_dir = os.path.join(out_dir, "train")
    gt_dir = os.path.join(out_dir, "train_gt")
    meta_dir = os.path.join(out_dir, "train_meta")
    _ensure_dir(train_dir)
    _ensure_dir(gt_dir)
    _ensure_dir(meta_dir)

    lr_h, lr_w = int(lr_size[0]), int(lr_size[1])
    hr_h, hr_w = lr_h * int(hr_factor), lr_w * int(hr_factor)

    for i in range(n_samples):
        emitter = _make_emitter(hr_h, hr_w, rng)
        lp = _make_light_patterns(hr_h, hr_w, rng)

        # PSF
        psf_sigma = float(rng.uniform(psf_sigma_range[0], psf_sigma_range[1]))
        psf_kernel = _gaussian_kernel(size=31, sigma=psf_sigma)

        # SIM forward
        sim_hr = []
        for p in lp:
            img = emitter * p
            img = _fft_convolve2d(img, psf_kernel)
            sim_hr.append(img)
        sim_hr = np.stack(sim_hr, axis=0).astype(np.float32)

        # downsample to LR
        sim_lr = np.stack([_downsample_avg(s, hr_factor) for s in sim_hr], axis=0)

        # heteroscedastic noise
        noise_mask = _make_noise_mask(lr_h, lr_w, rng)
        sigma_base = noise_base + noise_gain * sim_lr
        sigma_map = sigma_base + noise_region_amp * noise_mask[None, ...]
        # per-channel jitter
        ch_jitter = (1.0 + 0.1 * rng.standard_normal(sim_lr.shape[0])).reshape(-1, 1, 1)
        sigma_map = sigma_map * ch_jitter

        noise = rng.normal(0.0, 1.0, size=sim_lr.shape).astype(np.float32) * sigma_map
        sim_noisy = np.maximum(sim_lr + noise, 0.0)

        # save
        name = f"sample_{i:06d}"
        sim_path = os.path.join(train_dir, f"{name}.tif")
        em_path = os.path.join(gt_dir, f"{name}.tif")
        lp_path = os.path.join(gt_dir, f"{name}_lp.tif")
        meta_path = os.path.join(meta_dir, f"{name}.npz")

        _save_tif(sim_path, _to_uint16(sim_noisy))
        _save_tif(em_path, _to_uint16(emitter))
        _save_tif(lp_path, _to_uint16(lp))

        np.savez_compressed(
            meta_path,
            psf_sigma=np.array(psf_sigma, dtype=np.float32),
            noise_sigma=sigma_map.astype(np.float32),
            noise_mask=noise_mask.astype(np.float32),
        )

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{n_samples}] saved")


def main():
    p = argparse.ArgumentParser("Generate synthetic SIM dataset for supervised UQ")
    p.add_argument("--out_dir", type=str, default="./data_uq_sup_sim")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--lr_h", type=int, default=80)
    p.add_argument("--lr_w", type=int, default=80)
    p.add_argument("--hr_factor", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--psf_sigma_min", type=float, default=1.2)
    p.add_argument("--psf_sigma_max", type=float, default=2.6)
    p.add_argument("--noise_base", type=float, default=0.01)
    p.add_argument("--noise_gain", type=float, default=0.08)
    p.add_argument("--noise_region_amp", type=float, default=0.08)
    args = p.parse_args()

    generate_dataset(
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        lr_size=(args.lr_h, args.lr_w),
        hr_factor=args.hr_factor,
        seed=args.seed,
        psf_sigma_range=(args.psf_sigma_min, args.psf_sigma_max),
        noise_base=args.noise_base,
        noise_gain=args.noise_gain,
        noise_region_amp=args.noise_region_amp,
    )


if __name__ == "__main__":
    main()
