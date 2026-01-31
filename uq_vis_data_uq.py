# uq_vis_data_uq.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_vis import (
    save_debug_artifacts,
    save_npz,
    to_chw,
    to_hw,
    vol_to_2d,
    save_png_uncertainty_9hw,
    save_png_single_map_pct,
    save_hist_png,
    psf_mass_2d_np,
    center_of_mass_np,
    tv_np,
)


def _safe_np(x):
    return np.asarray(x, dtype=np.float32)


def _stat_summary(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float32).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "p95": float("nan")}
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95))}


def _pair_vmin_vmax(a: np.ndarray, b: np.ndarray, pmin: float, pmax: float) -> tuple[float, float]:
    merged = np.concatenate([a.ravel(), b.ravel()])
    vmin, vmax = np.percentile(merged, [pmin, pmax])
    return float(vmin), float(vmax)


def save_pair_2d(path: str, a: np.ndarray, b: np.ndarray, title_a: str, title_b: str, pmin: float, pmax: float):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    vmin, vmax = _pair_vmin_vmax(a, b, pmin, pmax)
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


def save_pair_9hw(
    path: str,
    a_9hw: np.ndarray,
    b_9hw: np.ndarray,
    title_a: str,
    title_b: str,
    max_frames: int,
    pmin: float,
    pmax: float,
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    a = to_chw(a_9hw)
    b = to_chw(b_9hw)
    n = min(max_frames, a.shape[0], b.shape[0])
    if n <= 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i in range(n):
        vmin, vmax = _pair_vmin_vmax(a[i], b[i], pmin, pmax)
        im0 = axes[0, i].imshow(a[i], vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, i].set_title(f"{title_a}[{i}]")
        axes[0, i].axis("off")
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

        im1 = axes[1, i].imshow(b[i], vmin=vmin, vmax=vmax, aspect="auto")
        axes[1, i].set_title(f"{title_b}[{i}]")
        axes[1, i].axis("off")
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _collect_sup_npz(root_dir: str) -> list[str]:
    npz_list = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f == "sup_sample.npz":
                npz_list.append(os.path.join(root, f))
    return sorted(npz_list)


def _get_var_from_npz(z: dict, prefix: str) -> Optional[np.ndarray]:
    std_key = f"uq_{prefix}_std"
    logv_key = f"uq_{prefix}_logvar"
    if std_key in z:
        std = np.asarray(z[std_key], dtype=np.float32)
        return std * std
    if logv_key in z:
        return np.exp(np.asarray(z[logv_key], dtype=np.float32))
    return None


def _get_pred_gt_from_npz(z: dict, prefix: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    pred_key = f"{prefix}_pred_hw"
    gt_key = f"{prefix}_gt_hw"
    pred9_key = f"{prefix}_pred_9hw"
    gt9_key = f"{prefix}_gt_9hw"
    if pred_key in z and gt_key in z:
        return np.asarray(z[pred_key], dtype=np.float32), np.asarray(z[gt_key], dtype=np.float32)
    if pred9_key in z and gt9_key in z:
        return np.asarray(z[pred9_key], dtype=np.float32), np.asarray(z[gt9_key], dtype=np.float32)
    return None, None


def _calibration_curve(pred_var: np.ndarray, err2: np.ndarray, bins: int = 10):
    pv = np.asarray(pred_var, dtype=np.float32).reshape(-1)
    e2 = np.asarray(err2, dtype=np.float32).reshape(-1)
    if pv.size == 0 or e2.size == 0:
        return None, None
    idx = np.argsort(pv)
    pv = pv[idx]
    e2 = e2[idx]
    n = pv.size
    edges = np.linspace(0, n, bins + 1, dtype=int)
    xs = []
    ys = []
    for i in range(bins):
        s = edges[i]
        t = edges[i + 1]
        if t <= s:
            continue
        xs.append(float(np.mean(pv[s:t])))
        ys.append(float(np.mean(e2[s:t])))
    return np.asarray(xs), np.asarray(ys)


def _plot_calibration_curve(x: np.ndarray, y: np.ndarray, title: str, out_path: str):
    if x is None or y is None or x.size == 0 or y.size == 0:
        return
    mx = max(float(np.max(x)), float(np.max(y)))
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(x, y, marker="o", linewidth=1.2, label="empirical")
    ax.plot([0, mx], [0, mx], linestyle="--", color="gray", label="ideal")
    ax.set_title(title)
    ax.set_xlabel("predicted variance")
    ax.set_ylabel("empirical MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_calibration_curves_from_dir(root_dir: str, out_dir: Optional[str] = None, bins: int = 10):
    if not os.path.isdir(root_dir):
        return
    npz_list = _collect_sup_npz(root_dir)
    if not npz_list:
        return

    em_vars = []
    em_err2 = []
    lp_vars = []
    lp_err2 = []

    for pth in npz_list:
        try:
            with np.load(pth) as z:
                v = _get_var_from_npz(z, "emitter")
                pred, gt = _get_pred_gt_from_npz(z, "emitter")
                if v is not None and pred is not None and gt is not None:
                    em_vars.append(v)
                    em_err2.append((pred - gt) ** 2)

                v = _get_var_from_npz(z, "lp")
                pred, gt = _get_pred_gt_from_npz(z, "lp")
                if v is not None and pred is not None and gt is not None:
                    lp_vars.append(v)
                    lp_err2.append((pred - gt) ** 2)
        except Exception:
            continue

    if out_dir is None:
        out_dir = os.path.join(root_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)

    if em_vars and em_err2:
        x, y = _calibration_curve(np.concatenate(em_vars), np.concatenate(em_err2), bins=bins)
        _plot_calibration_curve(x, y, "Emitter calibration", os.path.join(out_dir, "calib_emitter.png"))

    if lp_vars and lp_err2:
        x, y = _calibration_curve(np.concatenate(lp_vars), np.concatenate(lp_err2), bins=bins)
        _plot_calibration_curve(x, y, "LP calibration", os.path.join(out_dir, "calib_lp.png"))


def save_debug_artifacts_data_uq(
    out_dir: str,
    args: Any,
    debug_path: str,
    crop_record: Optional[Dict[str, Any]],
    x_np: np.ndarray,
    rec_np: np.ndarray,
    lp_np: np.ndarray,
    emitter_np: np.ndarray,
    psf_np: np.ndarray,
    mask_np: Optional[np.ndarray] = None,
    deconv_np: Optional[np.ndarray] = None,
    uq_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Wrapper on uq_vis.save_debug_artifacts, plus data-UQ maps.
    uq_pack can contain: rec_std, emitter_std, lp_std, psf_std, deconv_std,
    rec_var, emitter_logvar, lp_logvar, psf_logvar, etc.
    """
    base = save_debug_artifacts(
        out_dir=out_dir,
        args=args,
        debug_path=debug_path,
        crop_record=crop_record,
        x_np=x_np,
        rec_np=rec_np,
        lp_np=lp_np,
        emitter_np=emitter_np,
        psf_np=psf_np,
        mask_np=mask_np,
        deconv_np=deconv_np,
    )

    if not uq_pack:
        return base

    os.makedirs(out_dir, exist_ok=True)

    # ---- save npz ----
    uq_arrays = {k: _safe_np(v) for k, v in uq_pack.items() if v is not None}
    save_npz(os.path.join(out_dir, "uq_uncertainty.npz"), **uq_arrays)

    # ---- save pngs ----
    pmin = float(getattr(args, "vis_pmin", 1.0))
    pmax = float(getattr(args, "vis_pmax", 99.0))
    max_frames = int(getattr(args, "debug_frames", 3))

    if "rec_std" in uq_arrays:
        rec_std_9hw = to_chw(uq_arrays["rec_std"])
        save_png_uncertainty_9hw(
            os.path.join(out_dir, "rec_std.png"),
            rec_std_9hw,
            "Rec std",
            max_frames=max_frames,
            pmin=pmin,
            pmax=pmax,
        )
        save_hist_png(os.path.join(out_dir, "rec_std_hist.png"), uq_arrays["rec_std"], "Rec std histogram")

    if "lp_std" in uq_arrays:
        lp_std_9hw = to_chw(uq_arrays["lp_std"])
        save_png_uncertainty_9hw(
            os.path.join(out_dir, "lp_std.png"),
            lp_std_9hw,
            "LightPattern std",
            max_frames=max_frames,
            pmin=pmin,
            pmax=pmax,
        )
        save_hist_png(os.path.join(out_dir, "lp_std_hist.png"), uq_arrays["lp_std"], "LightPattern std histogram")

    if "emitter_std" in uq_arrays:
        em_std = vol_to_2d(uq_arrays["emitter_std"], mode="zmid")
        save_png_single_map_pct(
            os.path.join(out_dir, "emitter_std.png"),
            em_std,
            "Emitter std (zmid)",
            pmin=pmin,
            pmax=pmax,
        )
        save_hist_png(os.path.join(out_dir, "emitter_std_hist.png"), uq_arrays["emitter_std"], "Emitter std histogram")

    if "emitter_nll" in uq_arrays:
        em_nll = vol_to_2d(uq_arrays["emitter_nll"], mode="zmid")
        save_png_single_map_pct(
            os.path.join(out_dir, "emitter_nll.png"),
            em_nll,
            "Emitter NLL (zmid)",
            pmin=pmin,
            pmax=pmax,
        )
        save_hist_png(os.path.join(out_dir, "emitter_nll_hist.png"), uq_arrays["emitter_nll"], "Emitter NLL histogram")

    if "deconv_std" in uq_arrays:
        de_std = vol_to_2d(uq_arrays["deconv_std"], mode="zmid")
        save_png_single_map_pct(
            os.path.join(out_dir, "deconv_std.png"),
            de_std,
            "Deconv std (zmid)",
            pmin=pmin,
            pmax=pmax,
        )
        save_hist_png(os.path.join(out_dir, "deconv_std_hist.png"), uq_arrays["deconv_std"], "Deconv std histogram")

    if "psf_std" in uq_arrays:
        psf_std = vol_to_2d(uq_arrays["psf_std"], mode="zmid")
        save_png_single_map_pct(
            os.path.join(out_dir, "psf_std.png"),
            psf_std,
            "PSF std (zmid)",
            pmin=pmin,
            pmax=pmax,
        )
        save_hist_png(os.path.join(out_dir, "psf_std_hist.png"), uq_arrays["psf_std"], "PSF std histogram")

    if "lp_nll" in uq_arrays:
        lp_nll_9hw = to_chw(uq_arrays["lp_nll"])
        save_png_uncertainty_9hw(
            os.path.join(out_dir, "lp_nll.png"),
            lp_nll_9hw,
            "LightPattern NLL",
            max_frames=max_frames,
            pmin=pmin,
            pmax=pmax,
        )
        save_hist_png(os.path.join(out_dir, "lp_nll_hist.png"), uq_arrays["lp_nll"], "LightPattern NLL histogram")

    # ---- summary json ----
    summary: Dict[str, Any] = {}
    if "rec_std" in uq_arrays:
        summary["rec_std"] = _stat_summary(uq_arrays["rec_std"])
    if "emitter_std" in uq_arrays:
        summary["emitter_std"] = _stat_summary(uq_arrays["emitter_std"])
    if "lp_std" in uq_arrays:
        summary["lp_std"] = _stat_summary(uq_arrays["lp_std"])
    if "psf_std" in uq_arrays:
        summary["psf_std"] = _stat_summary(uq_arrays["psf_std"])
    if "emitter_nll" in uq_arrays:
        summary["emitter_nll"] = _stat_summary(uq_arrays["emitter_nll"])
    if "lp_nll" in uq_arrays:
        summary["lp_nll"] = _stat_summary(uq_arrays["lp_nll"])

    # optional: rec_nll from rec_var
    if "rec_var" in uq_arrays:
        diff = _safe_np(rec_np) - _safe_np(x_np)
        var = np.maximum(uq_arrays["rec_var"], 1e-8)
        rec_nll = 0.5 * (diff * diff / var + np.log(var))
        summary["rec_nll"] = float(np.mean(rec_nll))

    with open(os.path.join(out_dir, "uq_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # expose key metrics to CSV
    if "rec_std" in uq_arrays:
        base["rec_std_mean"] = summary.get("rec_std", {}).get("mean")
    if "emitter_std" in uq_arrays:
        base["emitter_std_mean"] = summary.get("emitter_std", {}).get("mean")
    if "lp_std" in uq_arrays:
        base["lp_std_mean"] = summary.get("lp_std", {}).get("mean")
    if "psf_std" in uq_arrays:
        base["psf_std_mean"] = summary.get("psf_std", {}).get("mean")
    if "emitter_nll" in uq_arrays:
        base["emitter_nll_mean"] = summary.get("emitter_nll", {}).get("mean")
    if "lp_nll" in uq_arrays:
        base["lp_nll_mean"] = summary.get("lp_nll", {}).get("mean")
    if "rec_nll" in summary:
        base["rec_nll"] = summary["rec_nll"]

    return base


def save_debug_artifacts_data_uq_sup(
    out_dir: str,
    args: Any,
    debug_path: str,
    crop_record: Optional[Dict[str, Any]],
    x_np: np.ndarray,
    rec_np: np.ndarray,
    lp_np: np.ndarray,
    emitter_np: np.ndarray,
    psf_np: np.ndarray,
    mask_np: Optional[np.ndarray] = None,
    deconv_np: Optional[np.ndarray] = None,
    uq_pack: Optional[Dict[str, Any]] = None,
    gt_pack: Optional[Dict[str, Any]] = None,
    noise_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Supervised debug visualization:
      - input image
      - pred vs GT pairs for emitter / lp / psf
      - uncertainty vs true noise (if provided)
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

    cy, cx = center_of_mass_np(psf_mass)
    ty, tx = (Hpsf - 1) * 0.5, (Wpsf - 1) * 0.5
    psf_center = float((cy - ty) ** 2 + (cx - tx) ** 2)

    # ---- base arrays ----
    inp_9hw = to_chw(x_np)
    lp_pred_9hw = to_chw(lp_np)
    emitter_pred_hw = vol_to_2d(emitter_np, mode="zmid")
    psf_pred_hw = vol_to_2d(psf_np, mode="zmid")

    # ---- save npz (compact) ----
    arrays_to_save: Dict[str, Any] = dict(
        debug_path=np.array([debug_path]),
        input_9hw=inp_9hw,
        lp_pred_9hw=lp_pred_9hw,
        emitter_pred_hw=emitter_pred_hw,
        psf_pred_hw=psf_pred_hw,
    )
    if gt_pack:
        if "emitter_gt" in gt_pack and gt_pack["emitter_gt"] is not None:
            arrays_to_save["emitter_gt_hw"] = vol_to_2d(gt_pack["emitter_gt"], mode="zmid")
        if "lp_gt" in gt_pack and gt_pack["lp_gt"] is not None:
            arrays_to_save["lp_gt_9hw"] = to_chw(gt_pack["lp_gt"])
        if "psf_gt" in gt_pack and gt_pack["psf_gt"] is not None:
            arrays_to_save["psf_gt_hw"] = vol_to_2d(gt_pack["psf_gt"], mode="zmid")
    if uq_pack:
        for k, v in uq_pack.items():
            arrays_to_save[f"uq_{k}"] = _safe_np(v)
    if noise_pack:
        for k, v in noise_pack.items():
            arrays_to_save[f"noise_{k}"] = _safe_np(v)

    save_npz(os.path.join(out_dir, "sup_sample.npz"), **arrays_to_save)

    # ---- save PNGs ----
    pmin = float(getattr(args, "vis_pmin", 1.0))
    pmax = float(getattr(args, "vis_pmax", 99.0))
    max_frames = int(getattr(args, "debug_frames", 3))

    # input
    save_png_uncertainty_9hw(
        os.path.join(out_dir, "input.png"),
        inp_9hw,
        "Input",
        max_frames=max_frames,
        pmin=pmin,
        pmax=pmax,
    )

    # emitter pred vs gt
    if gt_pack and gt_pack.get("emitter_gt") is not None:
        emitter_gt_hw = vol_to_2d(gt_pack["emitter_gt"], mode="zmid")
        save_pair_2d(
            os.path.join(out_dir, "emitter_pred_gt.png"),
            emitter_pred_hw,
            emitter_gt_hw,
            "Emitter Pred",
            "Emitter GT",
            pmin=pmin,
            pmax=pmax,
        )

    # lp pred vs gt (9 frames)
    if gt_pack and gt_pack.get("lp_gt") is not None:
        lp_gt_9hw = to_chw(gt_pack["lp_gt"])
        save_pair_9hw(
            os.path.join(out_dir, "lp_pred_gt.png"),
            lp_pred_9hw,
            lp_gt_9hw,
            "LP Pred",
            "LP GT",
            max_frames=max_frames,
            pmin=pmin,
            pmax=pmax,
        )

    # psf pred vs gt
    if gt_pack and gt_pack.get("psf_gt") is not None:
        psf_gt_hw = vol_to_2d(gt_pack["psf_gt"], mode="zmid")
        save_pair_2d(
            os.path.join(out_dir, "psf_pred_gt.png"),
            psf_pred_hw,
            psf_gt_hw,
            "PSF Pred (zmid)",
            "PSF GT (zmid)",
            pmin=pmin,
            pmax=pmax,
        )

    # uncertainty vs noise level (prefer rec_std vs noise_sigma)
    if uq_pack and noise_pack:
        uq_std = None
        noise_sigma = None
        if "rec_std" in uq_pack:
            uq_std = to_chw(uq_pack["rec_std"])
        if "noise_sigma" in noise_pack:
            noise_sigma = to_chw(noise_pack["noise_sigma"])
        if uq_std is not None and noise_sigma is not None:
            save_pair_9hw(
                os.path.join(out_dir, "uq_vs_noise.png"),
                uq_std,
                noise_sigma,
                "Pred UQ (rec_std)",
                "True noise sigma",
                max_frames=max_frames,
                pmin=pmin,
                pmax=pmax,
            )

    info = {
        "debug_path": debug_path,
        "crop_record": crop_record,
        "rec_vs_input": {"mse": rec_mse, "mae": rec_mae, "psnr": psnr},
        "psf_debug": {
            "tv": psf_tv,
            "center_loss": psf_center,
            "center_of_mass": {"cy": cy, "cx": cx, "target_y": ty, "target_x": tx},
            "shape": list(np.asarray(psf_np).shape),
        },
        "lp_debug": {"tv": lp_tv, "shape": list(np.asarray(lp_np).shape)},
        "saved_files": [
            "input.png",
            "emitter_pred_gt.png",
            "lp_pred_gt.png",
            "psf_pred_gt.png",
            "uq_vs_noise.png",
            "sup_sample.npz",
        ],
    }
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return {
        "rec_mse": rec_mse,
        "rec_mae": rec_mae,
        "psnr": psnr,
        "lp_tv": lp_tv,
        "psf_tv": psf_tv,
        "psf_center": psf_center,
    }
