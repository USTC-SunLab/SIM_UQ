# plot_metrics_curves.py
# -*- coding: utf-8 -*-
"""Plot loss/metric curves from metrics.csv."""
from __future__ import annotations

import os
import csv
import argparse
from math import ceil
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_metrics(path: str) -> Dict[str, List[float]]:
    rows: Dict[str, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # skip empty rows
            if not r:
                continue
            for k, v in r.items():
                if v is None or str(v).strip() == "":
                    continue
                try:
                    val = float(v)
                except Exception:
                    continue
                rows.setdefault(k, []).append(val)
    return rows


def plot_metrics(metrics_csv: str, out_path: str, x_key: str = "step"):
    data = _read_metrics(metrics_csv)
    if not data:
        raise RuntimeError(f"No valid numeric data in {metrics_csv}")

    if x_key not in data:
        # fallback to epoch or index
        if "epoch" in data:
            x = np.asarray(data["epoch"], dtype=np.float32)
        else:
            n = len(next(iter(data.values())))
            x = np.arange(n, dtype=np.float32)
    else:
        x = np.asarray(data[x_key], dtype=np.float32)

    prefer = [
        "loss",
        "emitter_nll",
        "lp_nll",
        "psf_nll",
        "emitter_calib_ratio",
        "lp_calib_ratio",
        "emitter_corr",
        "lp_corr",
    ]

    keys = [k for k in prefer if k in data]
    if not keys:
        keys = [k for k in data.keys() if k not in ("time", "note")]

    n = len(keys)
    ncols = 2
    nrows = int(ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    for i, k in enumerate(keys):
        r = i // ncols
        c = i % ncols
        y = np.asarray(data[k], dtype=np.float32)
        axes[r, c].plot(x[: len(y)], y, linewidth=1.2)
        axes[r, c].set_title(k)
        axes[r, c].set_xlabel(x_key)
        axes[r, c].grid(True, alpha=0.3)

    # hide unused axes
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser("Plot metrics curves")
    p.add_argument("--metrics_csv", type=str, required=True)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--x_key", type=str, default="step")
    args = p.parse_args()

    out_path = args.out
    if not out_path:
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.metrics_csv)), "metrics_curves.png")

    plot_metrics(args.metrics_csv, out_path, x_key=args.x_key)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
