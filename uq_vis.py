# uq_vis.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

# matplotlib（训练/服务器环境建议 Agg）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- basic helpers ----------------
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def to_chw(arr: np.ndarray) -> np.ndarray:
    """
    尽量把输入转成 CHW（用于画 9 通道网格 / frames）。
    支持：
      - (B,1,9,H,W) / (B,9,H,W) / (B,H,W,9)
      - (1,9,H,W) / (9,H,W) / (H,W,9)
      - (H,W) -> (1,H,W)
    """
    a = np.asarray(arr)

    # drop batch / extra dims
    if a.ndim == 5:  # (B,1,9,H,W) or (B,?, ?, ?, ?)
        a = a[0]
    if a.ndim == 4:  # (1,9,H,W) or (B,9,H,W) or (B,H,W,9)
        a = a[0]

    if a.ndim == 3:
        if a.shape[0] in (1, 9):         # (C,H,W)
            return a.astype(np.float32, copy=False)
        if a.shape[-1] in (1, 9):        # (H,W,C)
            return np.transpose(a, (2, 0, 1)).astype(np.float32, copy=False)
        return a.astype(np.float32, copy=False)

    if a.ndim == 2:
        return a[None, ...].astype(np.float32, copy=False)

    raise ValueError(f"to_chw: unsupported shape {a.shape}")


def to_hw(arr: np.ndarray) -> np.ndarray:
    """把各种形状尽量 squeeze 到 (H,W)。"""
    a = np.asarray(arr)
    while a.ndim > 2:
        a = a[0]
    return a.astype(np.float32, copy=False)


def save_npz(path: str, **arrays):
    _ensure_dir(path)
    np.savez_compressed(path, **arrays)


# ---------------- uncertainty helpers ----------------
def _squeeze_to_3d(vol: np.ndarray) -> np.ndarray:
    """
    将 (B,1,Z,H,W) / (B,Z,H,W) / (Z,H,W) / (H,W) 统一到 (Z,H,W)（Z=1 也允许）。
    """
    v = np.asarray(vol)
    if v.ndim == 5:
        v = v[0, 0]
    elif v.ndim == 4:
        v = v[0]
    elif v.ndim == 3:
        pass
    elif v.ndim == 2:
        v = v[None, ...]
    else:
        while v.ndim > 3:
            v = v[0]
        if v.ndim == 2:
            v = v[None, ...]
    return v.astype(np.float32, copy=False)


def vol_to_2d(vol: np.ndarray, mode: str = "zmid") -> np.ndarray:
    """
    输入可为 (B,1,Z,H,W)/(Z,H,W)/(H,W)，输出 (H,W)
    mode: zmid / mip / sum / z0
    """
    v = _squeeze_to_3d(vol)  # (Z,H,W)
    if v.shape[0] == 1:
        return v[0]
    if mode == "mip":
        return v.max(axis=0)
    if mode == "sum":
        return v.sum(axis=0)
    if mode == "z0":
        return v[0]
    # default zmid
    return v[v.shape[0] // 2]


def save_hist_png(path: str, arr: np.ndarray, title: str, bins: int = 200):
    _ensure_dir(path)
    a = np.asarray(arr).ravel()
    a = a[np.isfinite(a)]
    plt.figure(figsize=(6, 4))
    plt.hist(a, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------- PSF debug utils ----------------
def psf_mass_2d_np(psf_np: np.ndarray) -> np.ndarray:
    psf = np.asarray(psf_np).astype(np.float32)
    H, W = psf.shape[-2], psf.shape[-1]
    mass = psf.reshape(-1, H, W).sum(axis=0)
    mass = np.maximum(mass, 0.0)
    mass = mass / (mass.sum() + 1e-8)
    return mass.astype(np.float32)


def gaussian_2d_np(H: int, W: int, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy = (H - 1) * 0.5
    cx = (W - 1) * 0.5
    s2 = max(float(sigma), 1e-6) ** 2
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s2)).astype(np.float32)
    g /= (g.sum() + 1e-8)
    return g


def center_of_mass_np(mass_hw: np.ndarray) -> Tuple[float, float]:
    mass = np.asarray(mass_hw).astype(np.float32)
    mass = np.maximum(mass, 0)
    mass = mass / (mass.sum() + 1e-8)
    H, W = mass.shape
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    cy = float((mass * ys).sum())
    cx = float((mass * xs).sum())
    return cy, cx


def tv_np(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float32)
    dh = np.abs(a[..., 1:, :] - a[..., :-1, :]).mean()
    dw = np.abs(a[..., :, 1:] - a[..., :, :-1]).mean()
    return float(dh + dw)


# ---------------- plotting ----------------
def _imshow_train(ax, img2d: np.ndarray, title: str):
    im = ax.imshow(img2d, aspect="auto")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _imshow_pct(ax, img2d: np.ndarray, title: str, pmin=1.0, pmax=99.0):
    img2d = np.asarray(img2d, dtype=np.float32)
    vmin, vmax = np.percentile(img2d, [pmin, pmax])
    im = ax.imshow(img2d, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_title(f"{title}\n(p{pmin:.0f}-p{pmax:.0f})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_png_input_rec_trainstyle(path: str, inp_9hw: np.ndarray, rec_9hw: np.ndarray, max_frames: int = 3):
    _ensure_dir(path)
    n = min(max_frames, inp_9hw.shape[0])

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    for i in range(n):
        _imshow_train(axes[0, i], inp_9hw[i], f"Input[{i}] (train-style)")
        _imshow_train(axes[1, i], rec_9hw[i], f"Rec[{i}] (train-style)")
        _imshow_train(axes[2, i], np.abs(rec_9hw[i] - inp_9hw[i]), f"|Rec-Input|[{i}] (train-style)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_png_input_rec_pct(path: str, inp_9hw: np.ndarray, rec_9hw: np.ndarray, max_frames: int = 3, pmin=1.0, pmax=99.0):
    _ensure_dir(path)
    n = min(max_frames, inp_9hw.shape[0])

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    for i in range(n):
        merged = np.concatenate([inp_9hw[i].ravel(), rec_9hw[i].ravel()])
        vmin, vmax = np.percentile(merged, [pmin, pmax])

        im0 = axes[0, i].imshow(inp_9hw[i], vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        axes[0, i].set_title(f"Input[{i}] (pct)")
        axes[0, i].axis("off")
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

        im1 = axes[1, i].imshow(rec_9hw[i], vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        axes[1, i].set_title(f"Rec[{i}] (pct)")
        axes[1, i].axis("off")
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

        err = np.abs(rec_9hw[i] - inp_9hw[i])
        e_vmin, e_vmax = np.percentile(err, [pmin, pmax])
        im2 = axes[2, i].imshow(err, vmin=e_vmin, vmax=e_vmax, interpolation="nearest", aspect="equal")
        axes[2, i].set_title(f"|Rec-Input|[{i}] (pct)")
        axes[2, i].axis("off")
        plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_png_lp(path: str, lp_9hw: np.ndarray, max_frames: int = 3):
    _ensure_dir(path)
    n = min(max_frames, lp_9hw.shape[0])

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        _imshow_train(axes[i], lp_9hw[i], f"LightPattern[{i}]")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_png_emitter_psf_gauss(path: str, emitter_hw: np.ndarray, psf_mass_hw: np.ndarray, gauss_hw: np.ndarray):
    _ensure_dir(path)
    diff = np.abs(psf_mass_hw - gauss_hw)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    _imshow_train(axes[0], emitter_hw, "Emitter")
    _imshow_train(axes[1], psf_mass_hw, "PSF mass (2D)")
    _imshow_train(axes[2], gauss_hw, "Gaussian target")
    _imshow_train(axes[3], diff, "|PSF - Gauss|")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_png_single_map_pct(path: str, img_hw: np.ndarray, title: str, pmin: float, pmax: float):
    _ensure_dir(path)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _imshow_pct(ax, img_hw, title, pmin=pmin, pmax=pmax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_png_uncertainty_9hw(path: str, uq_9hw: np.ndarray, title_prefix: str, max_frames: int, pmin: float, pmax: float):
    """
    uq_9hw: (9,H,W)
    """
    _ensure_dir(path)
    n = min(max_frames, uq_9hw.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        _imshow_pct(axes[i], uq_9hw[i], f"{title_prefix}[{i}]", pmin=pmin, pmax=pmax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------- high-level debug saver ----------------
def save_debug_artifacts(
    out_dir: str,
    args: Any,
    debug_path: str,
    crop_record: Optional[Dict[str, Any]],
    x_np: np.ndarray,        # input (B,1,9,H,W) or (B,9,H,W) ...
    rec_np: np.ndarray,
    lp_np: np.ndarray,
    emitter_np: np.ndarray,
    psf_np: np.ndarray,

    # --- optional ---
    mask_np: Optional[np.ndarray] = None,
    deconv_np: Optional[np.ndarray] = None,

    # --- MC Dropout uncertainty (optional) ---
    mc_samples: int = 0,
    mc_device: str = "",
    rec_mc_mean_np: Optional[np.ndarray] = None,
    rec_mc_std_np: Optional[np.ndarray] = None,
    rec_mc_cv_np: Optional[np.ndarray] = None,
    deconv_mc_mean_np: Optional[np.ndarray] = None,
    deconv_mc_std_np: Optional[np.ndarray] = None,
    deconv_mc_cv_np: Optional[np.ndarray] = None,
    mc_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    统一保存 debug 输出（npz/png/json），并返回关键数值指标用于训练日志。
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- metrics on full tensors (shape-aligned) ----
    diff = np.asarray(rec_np, np.float32) - np.asarray(x_np, np.float32)
    rec_mse = float(np.mean(diff * diff))
    rec_mae = float(np.mean(np.abs(diff)))
    psnr = float(20.0 * np.log10(1.0) - 10.0 * np.log10(rec_mse + 1e-8))

    lp_tv = tv_np(lp_np)
    psf_tv = tv_np(psf_np)

    psf_mass = psf_mass_2d_np(psf_np)
    Hpsf, Wpsf = psf_mass.shape
    gauss = gaussian_2d_np(Hpsf, Wpsf, float(args.psf_sigma))
    psf_gauss = float(np.mean((psf_mass - gauss) ** 2))

    cy, cx = center_of_mass_np(psf_mass)
    ty, tx = (Hpsf - 1) * 0.5, (Wpsf - 1) * 0.5
    psf_center = float((cy - ty) ** 2 + (cx - tx) ** 2)

    # ---- extract first sample for visualization ----
    inp_9hw = to_chw(x_np)         # (9,H,W)
    rec_9hw = to_chw(rec_np)
    lp_9hw = to_chw(lp_np)
    emitter_hw = to_hw(emitter_np)

    # ---- save npz ----
    arrays_to_save: Dict[str, Any] = dict(
        debug_path=np.array([debug_path]),
        input_9hw=inp_9hw,
        rec_9hw=rec_9hw,
        light_pattern_9hw=lp_9hw,
        emitter_hw=emitter_hw,
        psf=psf_np,
        psf_mass_hw=psf_mass,
        psf_gauss_hw=gauss,
    )
    if mask_np is not None:
        arrays_to_save["mask"] = np.asarray(mask_np, np.float32)
    if deconv_np is not None:
        arrays_to_save["deconv"] = np.asarray(deconv_np, np.float32)

    # MC arrays（可选）
    if rec_mc_mean_np is not None:
        arrays_to_save["rec_mc_mean"] = np.asarray(rec_mc_mean_np, np.float32)
    if rec_mc_std_np is not None:
        arrays_to_save["rec_mc_std"] = np.asarray(rec_mc_std_np, np.float32)
    if rec_mc_cv_np is not None:
        arrays_to_save["rec_mc_cv"] = np.asarray(rec_mc_cv_np, np.float32)
    if deconv_mc_mean_np is not None:
        arrays_to_save["deconv_mc_mean"] = np.asarray(deconv_mc_mean_np, np.float32)
    if deconv_mc_std_np is not None:
        arrays_to_save["deconv_mc_std"] = np.asarray(deconv_mc_std_np, np.float32)
    if deconv_mc_cv_np is not None:
        arrays_to_save["deconv_mc_cv"] = np.asarray(deconv_mc_cv_np, np.float32)

    save_npz(os.path.join(out_dir, "sample.npz"), **arrays_to_save)

    saved_files = {
        "sample_npz": "sample.npz",
        "input_rec_train": "input_rec_train.png",
        "input_rec_pct": "input_rec_pct.png",
        "light_pattern": "light_pattern.png",
        "emitter_psf_gauss": "emitter_psf_gauss.png",
    }

    # ---- save PNGs ----
    save_png_input_rec_trainstyle(
        os.path.join(out_dir, "input_rec_train.png"),
        inp_9hw,
        rec_9hw,
        max_frames=int(args.debug_frames),
    )
    save_png_input_rec_pct(
        os.path.join(out_dir, "input_rec_pct.png"),
        inp_9hw,
        rec_9hw,
        max_frames=int(args.debug_frames),
        pmin=float(args.vis_pmin),
        pmax=float(args.vis_pmax),
    )
    save_png_lp(
        os.path.join(out_dir, "light_pattern.png"),
        lp_9hw,
        max_frames=int(args.debug_frames),
    )
    save_png_emitter_psf_gauss(
        os.path.join(out_dir, "emitter_psf_gauss.png"),
        emitter_hw,
        psf_mass,
        gauss,
    )

    # ---- optional: mask/deconv quicklook ----
    if mask_np is not None:
        try:
            mask_hw = vol_to_2d(mask_np, mode="zmid")
            save_png_single_map_pct(
                os.path.join(out_dir, "mask.png"),
                mask_hw,
                "Mask (zmid)",
                pmin=float(args.vis_pmin),
                pmax=float(args.vis_pmax),
            )
            saved_files["mask_png"] = "mask.png"
        except Exception:
            pass

    if deconv_np is not None:
        try:
            de_hw = vol_to_2d(deconv_np, mode="zmid")
            save_png_single_map_pct(
                os.path.join(out_dir, "deconv.png"),
                de_hw,
                "Deconv (zmid)",
                pmin=float(args.vis_pmin),
                pmax=float(args.vis_pmax),
            )
            saved_files["deconv_png"] = "deconv.png"
        except Exception:
            pass

    # ---- MC Dropout uncertainty visualization ----
    mc_info = None
    if int(mc_samples) > 1:
        mc_info = {
            "mc_samples": int(mc_samples),
            "mc_device": str(mc_device),
        }
        if isinstance(mc_summary, dict):
            mc_info.update(mc_summary)

        with open(os.path.join(out_dir, "mc_uncertainty_summary.json"), "w", encoding="utf-8") as f:
            json.dump(mc_info, f, ensure_ascii=False, indent=2)
        saved_files["mc_summary_json"] = "mc_uncertainty_summary.json"

        # rec uncertainty (9 frames)
        if rec_mc_std_np is not None:
            try:
                rec_std_9hw = to_chw(rec_mc_std_np)
                save_png_uncertainty_9hw(
                    os.path.join(out_dir, "rec_mc_std.png"),
                    rec_std_9hw,
                    "Rec MC std",
                    max_frames=int(args.debug_frames),
                    pmin=float(args.vis_pmin),
                    pmax=float(args.vis_pmax),
                )
                saved_files["rec_mc_std_png"] = "rec_mc_std.png"
                save_hist_png(
                    os.path.join(out_dir, "rec_mc_std_hist.png"),
                    rec_mc_std_np,
                    title="Rec MC std histogram",
                    bins=200,
                )
                saved_files["rec_mc_std_hist_png"] = "rec_mc_std_hist.png"
            except Exception:
                pass

        if rec_mc_cv_np is not None:
            try:
                rec_cv_9hw = to_chw(rec_mc_cv_np)
                save_png_uncertainty_9hw(
                    os.path.join(out_dir, "rec_mc_cv.png"),
                    rec_cv_9hw,
                    "Rec MC CV",
                    max_frames=int(args.debug_frames),
                    pmin=float(args.vis_pmin),
                    pmax=float(args.vis_pmax),
                )
                saved_files["rec_mc_cv_png"] = "rec_mc_cv.png"
            except Exception:
                pass

        # deconv uncertainty (2D views)
        if deconv_mc_std_np is not None:
            try:
                std_zmid = vol_to_2d(deconv_mc_std_np, mode="zmid")
                save_png_single_map_pct(
                    os.path.join(out_dir, "deconv_mc_std_zmid.png"),
                    std_zmid,
                    "Deconv MC std (zmid)",
                    pmin=float(args.vis_pmin),
                    pmax=float(args.vis_pmax),
                )
                saved_files["deconv_mc_std_zmid_png"] = "deconv_mc_std_zmid.png"

                # 如果是 3D，补一个 MIP
                v3 = _squeeze_to_3d(deconv_mc_std_np)
                if v3.shape[0] > 1:
                    std_mip = vol_to_2d(deconv_mc_std_np, mode="mip")
                    save_png_single_map_pct(
                        os.path.join(out_dir, "deconv_mc_std_mip.png"),
                        std_mip,
                        "Deconv MC std (MIP)",
                        pmin=float(args.vis_pmin),
                        pmax=float(args.vis_pmax),
                    )
                    saved_files["deconv_mc_std_mip_png"] = "deconv_mc_std_mip.png"

                save_hist_png(
                    os.path.join(out_dir, "deconv_mc_std_hist.png"),
                    deconv_mc_std_np,
                    title="Deconv MC std histogram",
                    bins=200,
                )
                saved_files["deconv_mc_std_hist_png"] = "deconv_mc_std_hist.png"
            except Exception:
                pass

        if deconv_mc_cv_np is not None:
            try:
                cv_zmid = vol_to_2d(deconv_mc_cv_np, mode="zmid")
                save_png_single_map_pct(
                    os.path.join(out_dir, "deconv_mc_cv_zmid.png"),
                    cv_zmid,
                    "Deconv MC CV (zmid)",
                    pmin=float(args.vis_pmin),
                    pmax=float(args.vis_pmax),
                )
                saved_files["deconv_mc_cv_zmid_png"] = "deconv_mc_cv_zmid.png"
            except Exception:
                pass

    # ---- info.json ----
    info = {
        "debug_path": debug_path,
        "crop_record": crop_record,
        "rec_vs_input": {"mse": rec_mse, "mae": rec_mae, "psnr": psnr},
        "psf_debug": {
            "tv": psf_tv,
            "center_loss": psf_center,
            "gauss_prior": psf_gauss,
            "center_of_mass": {"cy": cy, "cx": cx, "target_y": ty, "target_x": tx},
            "shape": list(np.asarray(psf_np).shape),
        },
        "lp_debug": {"tv": lp_tv, "shape": list(np.asarray(lp_np).shape)},
        "uncertainty_mc": mc_info,
        "saved_files": saved_files,
    }
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 返回给训练脚本写 CSV 的指标（保持兼容）
    ret = {
        "rec_mse": rec_mse,
        "rec_mae": rec_mae,
        "psnr": psnr,
        "lp_tv": lp_tv,
        "psf_tv": psf_tv,
        "psf_center": psf_center,
        "psf_gauss": psf_gauss,
    }

    # 额外返回一些 UQ 标量（不会破坏旧逻辑，train 脚本可选择写入）
    if isinstance(mc_info, dict):
        if "rec_std_mean" in mc_info and mc_info["rec_std_mean"] is not None:
            ret["rec_mc_std_mean"] = float(mc_info["rec_std_mean"])
        if "rec_std_p95" in mc_info and mc_info["rec_std_p95"] is not None:
            ret["rec_mc_std_p95"] = float(mc_info["rec_std_p95"])
        if "deconv_std_mean" in mc_info and mc_info["deconv_std_mean"] is not None:
            ret["deconv_mc_std_mean"] = float(mc_info["deconv_std_mean"])
        if "deconv_std_p95" in mc_info and mc_info["deconv_std_p95"] is not None:
            ret["deconv_mc_std_p95"] = float(mc_info["deconv_std_p95"])

    return ret
