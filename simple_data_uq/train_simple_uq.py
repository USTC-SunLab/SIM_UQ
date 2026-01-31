# train_simple_uq.py
# -*- coding: utf-8 -*-
"""Train a stronger CNN (U-Net) to predict mean + logvar (heteroscedastic NLL)."""
from __future__ import annotations

import os
import csv
import time
import argparse
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _corrcoef_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.view(-1)
    b = b.view(-1)
    a = a - a.mean()
    b = b - b.mean()
    cov = (a * b).mean()
    va = (a * a).mean()
    vb = (b * b).mean()
    return cov / (torch.sqrt(va * vb) + eps)


class SimpleUQDataset(Dataset):
    def __init__(self, root: str):
        self.paths = sorted(glob(os.path.join(root, "*.npz")))
        if not self.paths:
            raise RuntimeError(f"No npz found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        z = np.load(self.paths[idx])
        x = z["x"].astype(np.float32)
        y = z["y"].astype(np.float32)
        sigma = z["sigma"].astype(np.float32)
        # add channel dim
        return (
            torch.from_numpy(x[None, ...]),
            torch.from_numpy(y[None, ...]),
            torch.from_numpy(sigma[None, ...]),
        )


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(4, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(4, out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.dropout(h)
        h = self.gn2(self.conv2(h))
        return self.act(h + self.skip(x))


class SimpleUQUNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 48, dropout: float = 0.05):
        super().__init__()
        self.enc1 = ResBlock(in_ch, base, dropout)
        self.enc2 = ResBlock(base, base * 2, dropout)
        self.enc3 = ResBlock(base * 2, base * 4, dropout)
        self.enc4 = ResBlock(base * 4, base * 8, dropout)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResBlock(base * 8, base * 16, dropout)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ResBlock(base * 16, base * 8, dropout)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ResBlock(base * 8, base * 4, dropout)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ResBlock(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ResBlock(base * 2, base, dropout)

        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out(d1)
        mu = out[:, :1]
        logvar = out[:, 1:]
        return mu, logvar


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, clamp_min: float, clamp_max: float):
    logvar = torch.clamp(logvar, clamp_min, clamp_max)
    var = torch.exp(logvar) + 1e-6
    diff = mu - target
    nll = 0.5 * (diff * diff / var + logvar)
    return nll.mean(), var, diff


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


def plot_metrics_curves(metrics_csv: str, out_path: str):
    if not os.path.exists(metrics_csv):
        return
    data: Dict[str, List[float]] = {}
    with open(metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k, v in r.items():
                if v is None or str(v).strip() == "":
                    continue
                try:
                    val = float(v)
                except Exception:
                    continue
                data.setdefault(k, []).append(val)
    if not data:
        return

    x = np.asarray(data.get("step", list(range(len(next(iter(data.values())))))), dtype=np.float32)
    keys = [k for k in ("loss", "mse", "nll", "calib_ratio", "corr", "sigma_mae", "sigma_corr") if k in data]
    if not keys:
        keys = [k for k in data.keys() if k not in ("time",)]

    n = len(keys)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    for i, k in enumerate(keys):
        r = i // ncols
        c = i % ncols
        y = np.asarray(data[k], dtype=np.float32)
        axes[r, c].plot(x[: len(y)], y, linewidth=1.2)
        axes[r, c].set_title(k)
        axes[r, c].set_xlabel("step")
        axes[r, c].grid(True, alpha=0.3)
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser("Train simple UQ model")
    p.add_argument("--data_dir", type=str, default="./simple_data_uq/data")
    p.add_argument("--ckpt_dir", type=str, default="./simple_data_uq/ckpts")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--logvar_min", type=float, default=-8.0)
    p.add_argument("--logvar_max", type=float, default=3.0)
    p.add_argument("--var_reg", type=float, default=1e-4)
    p.add_argument("--sigma_weight", type=float, default=0.2, help="supervise std with true sigma")
    p.add_argument("--debug_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", type=int, default=48, help="base channels for UNet")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_dataparallel", action="store_true", help="use DataParallel if multiple GPUs visible")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = SimpleUQDataset(os.path.join(args.data_dir, "train"))
    val_ds = SimpleUQDataset(os.path.join(args.data_dir, "val"))
    pin = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[device] {device} | n_gpu={n_gpu} | use_dataparallel={args.use_dataparallel}")

    model = SimpleUQUNet(base=args.base, dropout=args.dropout).to(device)
    if args.use_dataparallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_csv = os.path.join(args.ckpt_dir, "metrics.csv")
    header = [
        "time",
        "epoch",
        "step",
        "loss",
        "mse",
        "mae",
        "nll",
        "var_mean",
        "calib_ratio",
        "corr",
        "sigma_mae",
        "sigma_corr",
    ]

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, sigma in train_loader:
            x = x.to(device)
            y = y.to(device)
            sigma = sigma.to(device)
            opt.zero_grad()
            mu, logvar = model(x)
            nll, var, diff = gaussian_nll(mu, logvar, y, args.logvar_min, args.logvar_max)
            std = torch.sqrt(var)
            sigma_loss = F.l1_loss(std, sigma)
            loss = nll + args.var_reg * (logvar * logvar).mean() + args.sigma_weight * sigma_loss
            loss.backward()
            opt.step()

            mse = (diff * diff).mean()
            mae = diff.abs().mean()
            var_mean = var.mean()
            calib_ratio = mse / (var_mean + 1e-8)
            corr = _corrcoef_flat(var, diff * diff)
            sigma_mae = F.l1_loss(std, sigma)
            sigma_corr = _corrcoef_flat(var, sigma * sigma)

            row = {
                "time": time.time(),
                "epoch": epoch,
                "step": step,
                "loss": float(loss.detach().cpu()),
                "mse": float(mse.detach().cpu()),
                "mae": float(mae.detach().cpu()),
                "nll": float(nll.detach().cpu()),
                "var_mean": float(var_mean.detach().cpu()),
                "calib_ratio": float(calib_ratio.detach().cpu()),
                "corr": float(corr.detach().cpu()),
                "sigma_mae": float(sigma_mae.detach().cpu()),
                "sigma_corr": float(sigma_corr.detach().cpu()),
            }
            csv_append(metrics_csv, row, header)
            step += 1

        # debug visualization
        if args.debug_every > 0 and epoch % args.debug_every == 0:
            model.eval()
            with torch.no_grad():
                x, y, sigma = next(iter(val_loader))
                x = x.to(device)
                y = y.to(device)
                sigma = sigma.to(device)
                mu, logvar = model(x)
                logvar = torch.clamp(logvar, args.logvar_min, args.logvar_max)
                var = torch.exp(logvar) + 1e-6
                std = torch.sqrt(var)

                # take first sample
                x0 = x[0, 0].cpu().numpy()
                y0 = y[0, 0].cpu().numpy()
                mu0 = mu[0, 0].cpu().numpy()
                std0 = std[0, 0].cpu().numpy()
                sig0 = sigma[0, 0].cpu().numpy()
                err0 = np.abs(mu0 - y0)

                out_dir = os.path.join(args.ckpt_dir, "debug", f"epoch_{epoch:03d}")
                _ensure_dir(out_dir)
                save_pair(os.path.join(out_dir, "pred_gt.png"), mu0, y0, "Pred mean", "GT emitter")
                save_pair(os.path.join(out_dir, "uq_vs_sigma.png"), std0, sig0, "Pred std", "True sigma")
                save_triplet(os.path.join(out_dir, "input_pred_gt.png"), x0, mu0, y0, ("Input", "Pred", "GT"))
                save_pair(os.path.join(out_dir, "error_vs_sigma.png"), err0, sig0, "Abs error", "True sigma")

                # calibration curve on val set
                all_var = []
                all_err2 = []
                for xv, yv, _ in val_loader:
                    xv = xv.to(device)
                    yv = yv.to(device)
                    muv, logv = model(xv)
                    logv = torch.clamp(logv, args.logvar_min, args.logvar_max)
                    varv = torch.exp(logv) + 1e-6
                    diffv = muv - yv
                    all_var.append(varv.cpu().numpy())
                    all_err2.append((diffv * diffv).cpu().numpy())
                all_var = np.concatenate(all_var, axis=0)
                all_err2 = np.concatenate(all_err2, axis=0)
                curve = calibration_curve(all_var, all_err2, bins=10)
                plot_calibration(curve, os.path.join(out_dir, "calibration.png"), "Calibration (val)")

        # save checkpoint + curves
        _ensure_dir(args.ckpt_dir)
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, os.path.join(args.ckpt_dir, "model.pt"))
        plot_metrics_curves(metrics_csv, os.path.join(args.ckpt_dir, "metrics_curves.png"))

    print("[done] training finished")


if __name__ == "__main__":
    main()
