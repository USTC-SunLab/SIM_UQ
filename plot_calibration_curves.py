# plot_calibration_curves.py
# -*- coding: utf-8 -*-
"""Plot calibration curves from supervised debug outputs (sup_sample.npz)."""
from __future__ import annotations

import os
import argparse
from glob import glob
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _collect_npz(root: str) -> list[str]:
    if os.path.isdir(root):
        return sorted(glob(os.path.join(root, "**", "sup_sample.npz"), recursive=True))
    if root.endswith(".npz"):
        return [root]
    return []


def _flatten(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    return a.reshape(-1)


def _get_var(z: dict, prefix: str) -> np.ndarray:
    # prefer std, fallback to logvar
    std_key = f"uq_{prefix}_std"
    logv_key = f"uq_{prefix}_logvar"
    if std_key in z:
        std = np.asarray(z[std_key], dtype=np.float32)
        return std * std
    if logv_key in z:
        return np.exp(np.asarray(z[logv_key], dtype=np.float32))
    return None


def _get_pred_gt(z: dict, prefix: str) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
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
    pv = _flatten(pred_var)
    e2 = _flatten(err2)
    if pv.size == 0 or e2.size == 0:
        return None
    # sort by pred var
    idx = np.argsort(pv)
    pv = pv[idx]
    e2 = e2[idx]
    n = pv.size
    bin_edges = np.linspace(0, n, bins + 1, dtype=int)
    x = []
    y = []
    for i in range(bins):
        s = bin_edges[i]
        t = bin_edges[i + 1]
        if t <= s:
            continue
        x.append(float(np.mean(pv[s:t])))
        y.append(float(np.mean(e2[s:t])))
    return np.asarray(x), np.asarray(y)


def _plot_curve(x, y, title: str, out_path: str):
    if x is None or y is None:
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


def main():
    p = argparse.ArgumentParser("Plot calibration curves from sup_sample.npz")
    p.add_argument("--root", type=str, required=True, help="debug_samples dir or a sup_sample.npz")
    p.add_argument("--out_dir", type=str, default="", help="default: <root>/calibration")
    p.add_argument("--bins", type=int, default=10)
    args = p.parse_args()

    npz_list = _collect_npz(args.root)
    if not npz_list:
        raise RuntimeError(f"No sup_sample.npz found under: {args.root}")

    emitter_vars = []
    emitter_err2 = []
    lp_vars = []
    lp_err2 = []

    for pth in npz_list:
        with np.load(pth) as z:
            # emitter
            v = _get_var(z, "emitter")
            pred, gt = _get_pred_gt(z, "emitter")
            if v is not None and pred is not None and gt is not None:
                emitter_vars.append(v)
                emitter_err2.append((pred - gt) ** 2)
            # lp
            v = _get_var(z, "lp")
            pred, gt = _get_pred_gt(z, "lp")
            if v is not None and pred is not None and gt is not None:
                lp_vars.append(v)
                lp_err2.append((pred - gt) ** 2)

    out_dir = args.out_dir
    if not out_dir:
        if os.path.isdir(args.root):
            out_dir = os.path.join(args.root, "calibration")
        else:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(args.root)), "calibration")

    if emitter_vars and emitter_err2:
        x, y = _calibration_curve(np.concatenate(emitter_vars), np.concatenate(emitter_err2), bins=args.bins)
        _plot_curve(x, y, "Emitter calibration", os.path.join(out_dir, "calib_emitter.png"))

    if lp_vars and lp_err2:
        x, y = _calibration_curve(np.concatenate(lp_vars), np.concatenate(lp_err2), bins=args.bins)
        _plot_curve(x, y, "LP calibration", os.path.join(out_dir, "calib_lp.png"))

    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
