# generate_simple_dataset.py
# -*- coding: utf-8 -*-
"""
Generate a simple heteroscedastic dataset for UQ sanity check.
Each sample:
  - emitter GT: sum of Gaussian blobs, normalized to [0,1]
  - noise sigma map: base + regional blobs + gain * emitter
  - observation: x = emitter + sigma * eps
Saved as .npz with keys: x, y, sigma
"""
from __future__ import annotations

import os
import argparse
import numpy as np


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _minmax_full(im: np.ndarray) -> np.ndarray:
    im = im.astype(np.float32)
    mn = float(im.min())
    mx = float(im.max())
    if mx <= mn:
        return np.zeros_like(im, dtype=np.float32)
    return (im - mn) / (mx - mn)


def _make_emitter(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    n_blobs = int(rng.integers(6, 18))
    yy, xx = np.mgrid[0:h, 0:w]
    em = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_blobs):
        cy = float(rng.uniform(0, h))
        cx = float(rng.uniform(0, w))
        sigma = float(rng.uniform(2.0, 10.0))
        amp = float(rng.uniform(0.6, 1.2))
        blob = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
        em += blob
    return _minmax_full(em)


def _make_noise_map(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    n_blobs = int(rng.integers(2, 6))
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_blobs):
        cy = float(rng.uniform(0, h))
        cx = float(rng.uniform(0, w))
        sigma = float(rng.uniform(4.0, 14.0))
        amp = float(rng.uniform(0.5, 1.2))
        blob = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
        mask += blob
    return _minmax_full(mask)


def generate_split(out_dir: str, n: int, h: int, w: int, seed: int, base_sigma: float, gain: float, region_amp: float):
    rng = np.random.default_rng(seed)
    _ensure_dir(out_dir)
    for i in range(n):
        emitter = _make_emitter(h, w, rng)
        region = _make_noise_map(h, w, rng)
        sigma = base_sigma + gain * emitter + region_amp * region
        eps = rng.standard_normal((h, w)).astype(np.float32)
        x = emitter + sigma * eps
        x = np.clip(x, 0.0, 1.0)
        np.savez_compressed(
            os.path.join(out_dir, f"sample_{i:06d}.npz"),
            x=x.astype(np.float32),
            y=emitter.astype(np.float32),
            sigma=sigma.astype(np.float32),
        )
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{n}] saved")


def main():
    p = argparse.ArgumentParser("Generate simple UQ dataset")
    p.add_argument("--out_dir", type=str, default="./simple_data_uq/data")
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--h", type=int, default=80)
    p.add_argument("--w", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_sigma", type=float, default=0.02)
    p.add_argument("--gain", type=float, default=0.08)
    p.add_argument("--region_amp", type=float, default=0.08)
    args = p.parse_args()

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    generate_split(train_dir, args.n_train, args.h, args.w, args.seed, args.base_sigma, args.gain, args.region_amp)
    generate_split(val_dir, args.n_val, args.h, args.w, args.seed + 1234, args.base_sigma, args.gain, args.region_amp)


if __name__ == "__main__":
    main()
