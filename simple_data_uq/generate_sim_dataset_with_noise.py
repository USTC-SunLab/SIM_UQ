# generate_sim_dataset_with_noise.py
# -*- coding: utf-8 -*-
"""
Generate SIM dataset with explicit noise sigma labels.
Emitter types follow simulation_np.py: curve / tube / ring.
HR emitter resolution = 2x input (default).
Only standard data is generated (no parameter sweeps).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Iterable, List, Tuple

import numpy as np
import tqdm
from scipy.special import jv
from scipy.signal import fftconvolve
from scipy import ndimage
from skimage.transform import resize
from skimage.draw import disk

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import tifffile
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False
from skimage.io import imsave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_tiff(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if _HAS_TIFFFILE:
        tifffile.imwrite(path, arr.astype(np.float32), compression="zlib")
    else:
        imsave(path, arr.astype(np.float32))


def new_psf_2d(Lambda=488, size=49, step=None):
    _Nx = size
    _Ny = size
    _scale = 1.0
    _step = step or 62.6 / _scale
    _x = np.linspace(-_Nx / 2 * _step, _Nx / 2 * _step, _Nx) - 0.0001
    _y = np.linspace(-_Ny / 2 * _step, _Ny / 2 * _step, _Ny)
    xx, yy = np.meshgrid(_x, _y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    NA = 1.3
    mask = r > 1e-10
    psf = np.zeros_like(r)
    psf[mask] = (jv(1, 2 * np.pi * NA / Lambda * r[mask]) ** 2) / (np.pi * r[mask] ** 2)
    central_value = (NA ** 2) / (Lambda ** 2)
    psf[~mask] = central_value
    psf = psf / psf.sum()
    return psf


def convolve_fft(xin, k):
    x = np.asarray(xin)
    k = np.asarray(k)
    if k.ndim == 3 and k.shape[0] == 1:
        k = k[0]
    if x.ndim == 4 and x.shape[1] == 1:
        x = np.squeeze(x, axis=1)  # (C,H,W)
    if x.ndim == 3:  # (C,H,W)
        out = np.stack([fftconvolve(x[c], k, mode="same") for c in range(x.shape[0])], axis=0)
        return out
    return fftconvolve(x, k, mode="same")


def convolve_curve_with_psf(curve, psf):
    return convolve_fft(curve, psf)


def cosine_light_pattern(shape, Ks, phases, M):
    x = np.linspace(0, shape[-1] - 1, shape[-1])
    y = np.linspace(0, shape[-2] - 1, shape[-2])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    fields = []
    for i, K in enumerate(Ks):
        phase = phases[i]
        for j in range(3):
            kx, ky = K[0], K[1]
            field = 1 + M * np.cos(2 * np.pi / shape[-1] * (kx * xx + ky * yy) + phase + j * 2 * np.pi / 3)
            fields.append(field[np.newaxis, ...])
    return np.stack(fields, axis=0)


def get_Ks(theta_start, K_num, period, N):
    thetas = np.deg2rad(np.array([theta_start, theta_start + 120, theta_start + 240]))
    Ks = []
    for t in thetas:
        kx = N / period * np.cos(t)
        ky = N / period * np.sin(t)
        Ks.append([kx, ky])
    return Ks


def percentile_norm(img: np.ndarray, p_low: float = 0.1, p_high: float = 99.9) -> np.ndarray:
    vmin = np.percentile(img, p_low)
    vmax = np.percentile(img, p_high)
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - vmin) / (vmax - vmin)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def bezier_curve(control_points: np.ndarray, num_points: int = 100) -> np.ndarray:
    n = len(control_points)
    t = np.linspace(0, 1, num_points)
    curve_points = np.zeros((num_points, 2), dtype=np.float32)
    for i in range(n):
        binom = math.comb(n - 1, i)
        curve_points += np.outer(binom * (t ** i) * ((1 - t) ** (n - 1 - i)), control_points[i])
    return curve_points


def generate_bezier_curves_2d(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    if not _HAS_CV2:
        raise RuntimeError("cv2 is required for generate_bezier_curves_2d")
    upsampling_factor = 4
    n_up = upsampling_factor * n
    img = np.zeros((n_up, n_up), dtype=np.float32)
    m = max(1, int(m))
    for _ in range(m):
        num_points = int(rng.integers(3, 6))
        control_points = rng.random((num_points, 2)).astype(np.float32) * n_up
        curve_points = bezier_curve(control_points, num_points=n_up)
        xs = np.clip(curve_points[:, 0].astype(int), 0, n_up - 1)
        ys = np.clip(curve_points[:, 1].astype(int), 0, n_up - 1)
        img[ys, xs] = float(rng.integers(160, 255))
    img = cv2.GaussianBlur(img.astype(np.float32), (9, 9), 1)
    img = resize(img, (n, n), anti_aliasing=True)
    img /= 255.0
    return img.astype(np.float32)


def circular_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    if not _HAS_CV2:
        raise RuntimeError("cv2 is required for circular_blur")
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    for i in range(ksize):
        for j in range(ksize):
            if (i - center) ** 2 + (j - center) ** 2 <= center ** 2:
                kernel[i, j] = 1
    kernel /= np.sum(kernel)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def generate_bezier_curves_tube(rng: np.random.Generator, tube_width: int, n: int) -> np.ndarray:
    upsampling_factor = 4
    n_up = upsampling_factor * n
    img = np.zeros((n_up, n_up), dtype=np.float32)
    m = int(rng.integers(15, 25))
    for _ in range(m):
        num_points = int(rng.integers(3, 6))
        control_points = rng.random((num_points, 2)).astype(np.float32) * n_up
        curve_points = bezier_curve(control_points, num_points=n_up)
        xs = np.clip(curve_points[:, 0].astype(int), 0, n_up - 1)
        ys = np.clip(curve_points[:, 1].astype(int), 0, n_up - 1)
        for x, y in zip(xs, ys):
            rr, cc = disk((x, y), tube_width * upsampling_factor, shape=img.shape)
            img[rr, cc] = 255

    img /= 255.0
    sobel_x = ndimage.sobel(img, axis=0)
    sobel_y = ndimage.sobel(img, axis=1)
    edges = np.hypot(sobel_x, sobel_y)
    edges = resize(edges, (n, n), anti_aliasing=True)
    if edges.max() > 0:
        edges = (edges / edges.max() * 255).astype(np.uint8)
    else:
        edges = np.zeros_like(edges, dtype=np.uint8)
    img = circular_blur(edges.astype(np.float32), 2)
    return img.astype(np.float32)


def generate_ring(rng: np.random.Generator, circle_num: int, W: int, radius: Tuple[int, int]) -> np.ndarray:
    if not _HAS_CV2:
        raise RuntimeError("cv2 is required for generate_ring")
    img = np.zeros((W, W), dtype=np.uint8)
    r0, r1 = int(radius[0]), int(radius[1])
    for _ in range(int(circle_num)):
        x1 = int(rng.integers(r0, r1))
        x2 = int(rng.integers(max(r0, x1 - 1), min(r1, x1 + 1)))
        axes = (x1, x2)
        center = (int(rng.integers(axes[0], img.shape[0] - axes[0])), int(rng.integers(axes[1], img.shape[1] - axes[1])))
        angle = int(rng.integers(0, 360))

        segments = []
        for start_angle in range(0, 360, 60):
            end_angle = start_angle + 60
            segment = np.zeros_like(img)
            cv2.ellipse(segment, center, axes, angle, start_angle, end_angle, 255, 1)
            sigma = float(rng.uniform(0.5, 1.0))
            segment = cv2.GaussianBlur(segment, (0, 0), sigma)
            segment = np.clip(segment, 0, 255)
            segments.append(segment)
        ellipse = np.maximum.reduce(segments)
        img = np.maximum(img, ellipse)
    mx = float(img.max())
    if mx <= 0:
        return np.zeros_like(img, dtype=np.float32)
    return (img / mx).astype(np.float32)


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


def sim_raw_generator(
    period: int,
    emitter_type: str,
    W_hr: int,
    psf: np.ndarray,
    rng: np.random.Generator,
    curve_lines: int,
    tube_width: int,
    ring_radius: Tuple[int, int],
    magnitude: float,
    theta_start: float,
    phi: np.ndarray,
):
    if emitter_type == "curve":
        curve = generate_bezier_curves_2d(rng, curve_lines, W_hr)
    elif emitter_type == "tube":
        curve = generate_bezier_curves_tube(rng, tube_width, W_hr)
    elif emitter_type == "ring":
        circle_num = int(rng.integers(100, 200))
        curve = generate_ring(rng, circle_num, W_hr, ring_radius)
    else:
        raise ValueError(f"Unknown emitter_type: {emitter_type}")

    curve = curve - curve.min()
    curve = curve / (curve.max() + 1e-8)

    Ks = get_Ks(theta_start, 3, period, curve.shape[-1])
    cosine_patterns = cosine_light_pattern(curve.shape, Ks, phases=phi, M=magnitude)
    modified_curves = curve[np.newaxis, ...] * cosine_patterns
    res = convolve_curve_with_psf(modified_curves, psf)
    res = percentile_norm(res)
    return curve, modified_curves, res, cosine_patterns


def _percentile_bounds(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> Tuple[float, float]:
    vmin = float(np.percentile(arr, p_low))
    vmax = float(np.percentile(arr, p_high))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_vis_jpg(path: str, noisy: np.ndarray, emitter_hr: np.ndarray, lp_hr: np.ndarray, sigma: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    noisy = np.asarray(noisy, dtype=np.float32)
    emitter_hr = np.asarray(emitter_hr, dtype=np.float32)
    lp_hr = np.asarray(lp_hr, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)

    if lp_hr.ndim == 4 and lp_hr.shape[1] == 1:
        lp_hr = lp_hr[:, 0]
    if lp_hr.ndim == 4 and lp_hr.shape[0] == 1:
        lp_hr = lp_hr[0]

    # layout: 3x4 grid (9 frames + emitter + lp mean + sigma mean)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    vmin, vmax = _percentile_bounds(noisy)
    for i in range(9):
        r, c = divmod(i, 3)
        ax = axes[r, c]
        ax.imshow(noisy[i], vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"frame {i}")
        ax.axis("off")
    # emitter HR
    ax = axes[0, 3]
    vmin, vmax = _percentile_bounds(emitter_hr)
    ax.imshow(emitter_hr, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title("emitter HR")
    ax.axis("off")
    # LP frame 0 HR (avoid stripe cancellation from averaging phases)
    ax = axes[1, 3]
    lp_frame0 = lp_hr[0] if lp_hr.ndim == 3 else lp_hr
    vmin, vmax = _percentile_bounds(lp_frame0)
    ax.imshow(lp_frame0, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title("LP frame0 HR")
    ax.axis("off")
    # sigma mean LR
    ax = axes[2, 3]
    sigma_mean = sigma.mean(axis=0) if sigma.ndim == 3 else sigma
    vmin, vmax = _percentile_bounds(sigma_mean)
    ax.imshow(sigma_mean, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title("sigma mean LR")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def process_one(
    i: int,
    W: int,
    hr_factor: int,
    period: int,
    curve_lines: int,
    tube_width: int,
    ring_radius: Tuple[int, int],
    psf: np.ndarray,
    rng: np.random.Generator,
    noise: float,
    magnitude: float,
    ave_photon: float,
    emitter_type: str,
    data_path: str,
    gt_path: str,
    meta_path: str,
    vis_path: str,
):
    phi = (rng.random(3) - 0.5) * 2 * np.pi
    theta_start = float(rng.random() * 360.0)
    W_hr = int(W * hr_factor)

    curve, modified_curves, res_hr, cosine_patterns = sim_raw_generator(
        period * hr_factor,
        emitter_type,
        W_hr,
        psf,
        rng,
        curve_lines,
        tube_width,
        ring_radius,
        magnitude,
        theta_start,
        phi,
    )

    # downsample raw SIM to LR
    res_lr = downsample_mean(res_hr.astype(np.float32), hr_factor)

    # add noise (Poisson + Gaussian)
    res_std = res_lr / (res_lr.std() + 1e-8)
    no_empty_area = res_std > 1e-2
    res_lr = res_lr / (res_lr[no_empty_area].mean() + 1e-8) * ave_photon
    res_lr = np.clip(res_lr, 0.0, None)

    sigma_gauss = noise * (res_lr.std() + 1e-8)
    sigma_map = np.sqrt(np.maximum(res_lr, 0.0) + sigma_gauss ** 2).astype(np.float32)

    noisy = rng.poisson(res_lr).astype(np.float32)
    noisy = noisy + rng.normal(0.0, sigma_gauss, res_lr.shape).astype(np.float32)

    # save results
    save_tiff(f"{gt_path}/{i}.tif", curve.astype(np.float32))
    save_tiff(f"{gt_path}/{i}_lp.tif", cosine_patterns.astype(np.float32))
    save_tiff(f"{data_path}/{i}.tif", noisy.astype(np.float32))

    # meta
    config = {
        "period": int(period),
        "noise_level": float(noise),
        "M": float(magnitude),
        "theta_start": float(theta_start),
        "phi": phi.tolist(),
        "ave_photon": float(ave_photon),
        "hr_factor": int(hr_factor),
        "emitter_type": str(emitter_type),
        "curve_lines": int(curve_lines),
        "tube_width": int(tube_width),
        "ring_radius": [int(ring_radius[0]), int(ring_radius[1])],
    }
    os.makedirs(meta_path, exist_ok=True)
    with open(f"{meta_path}/{i}_config.json", "w") as f:
        json.dump(config, f)

    np.savez_compressed(
        f"{meta_path}/{i}.npz",
        noise_sigma=sigma_map.astype(np.float32),
        noise_level=float(noise),
        ave_photon=float(ave_photon),
        hr_factor=int(hr_factor),
        emitter_type=str(emitter_type),
    )

    # vis
    save_vis_jpg(os.path.join(vis_path, f"{i}.jpg"), noisy, curve, cosine_patterns, sigma_map)


def generate_split(
    out_dir: str,
    split: str,
    n: int,
    W: int,
    hr_factor: int,
    period: int,
    curve_lines: int,
    tube_width: int,
    ring_radius: Tuple[int, int],
    psf: np.ndarray,
    rng: np.random.Generator,
    noise_min: float,
    noise_max: float,
    magnitude: float,
    ave_photon: float,
    emitter_type: str,
):
    data_path = os.path.join(out_dir, split)
    gt_path = os.path.join(out_dir, f"{split}_gt")
    meta_path = os.path.join(out_dir, f"{split}_meta")
    vis_path = os.path.join(out_dir, f"{split}_vis")
    for pth in [data_path, gt_path, meta_path, vis_path]:
        os.makedirs(pth, exist_ok=True)

    for i in tqdm.tqdm(range(n), desc=f"{emitter_type}:{split}"):
        noise = float(rng.uniform(noise_min, noise_max))
        process_one(
            i,
            W,
            hr_factor,
            period,
            curve_lines,
            tube_width,
            ring_radius,
            psf,
            rng,
            noise,
            magnitude,
            ave_photon,
            emitter_type,
            data_path,
            gt_path,
            meta_path,
            vis_path,
        )


def _parse_types(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def main():
    p = argparse.ArgumentParser("Generate SIM dataset with noise sigma labels (types: curve/tube/ring)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--types", type=str, default="curve,tube,ring")
    p.add_argument("--n_train", type=int, default=10, help="per type")
    p.add_argument("--n_val", type=int, default=2, help="per type")
    p.add_argument("--W", type=int, default=80, help="LR size")
    p.add_argument("--hr_factor", type=int, default=2, help="HR factor for GT emitter")
    p.add_argument("--period", type=int, default=5)
    p.add_argument("--curve_lines", type=int, default=20)
    p.add_argument("--tube_width", type=int, default=3)
    p.add_argument("--ring_radius", type=int, nargs=2, default=(3, 4))
    p.add_argument("--magnitude", type=float, default=0.3)
    p.add_argument("--ave_photon", type=float, default=100.0)
    p.add_argument("--noise_min", type=float, default=0.05)
    p.add_argument("--noise_max", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--psf_size", type=int, default=0, help="override PSF size, 0=auto")
    args = p.parse_args()

    if not _HAS_CV2:
        raise RuntimeError("cv2 is required; please install opencv-python")

    rng = np.random.default_rng(args.seed)
    hr_factor = max(1, int(args.hr_factor))
    psf_size = int(args.psf_size) if int(args.psf_size) > 0 else max(9, int(49 * hr_factor / 3))
    psf = np.array(new_psf_2d(500, psf_size, 62.6 / hr_factor))[np.newaxis, ...]

    types = _parse_types(args.types)
    for emitter_type in types:
        type_dir = os.path.join(args.out_dir, emitter_type)
        generate_split(
            type_dir,
            "train",
            args.n_train,
            args.W,
            hr_factor,
            args.period,
            args.curve_lines,
            args.tube_width,
            tuple(args.ring_radius),
            psf,
            rng,
            args.noise_min,
            args.noise_max,
            args.magnitude,
            args.ave_photon,
            emitter_type,
        )
        if args.n_val > 0:
            generate_split(
                type_dir,
                "val",
                args.n_val,
                args.W,
                hr_factor,
                args.period,
                args.curve_lines,
                args.tube_width,
                tuple(args.ring_radius),
                psf,
                rng,
                args.noise_min,
                args.noise_max,
                args.magnitude,
                args.ave_photon,
                emitter_type,
            )


if __name__ == "__main__":
    main()
