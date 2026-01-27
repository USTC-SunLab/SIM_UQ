# uq_vis_data_uq.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import numpy as np

from uq_vis import (
    save_debug_artifacts,
    save_npz,
    to_chw,
    to_hw,
    vol_to_2d,
    save_png_uncertainty_9hw,
    save_png_single_map_pct,
    save_hist_png,
)


def _safe_np(x):
    return np.asarray(x, dtype=np.float32)


def _stat_summary(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float32).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "p95": float("nan")}
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95))}


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
