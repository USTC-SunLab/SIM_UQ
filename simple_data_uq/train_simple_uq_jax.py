# train_simple_uq_jax.py
# -*- coding: utf-8 -*-
"""Train a stronger UNet in JAX/Flax to predict mean + logvar (heteroscedastic NLL)."""
from __future__ import annotations

import os
import csv
import time
import argparse
from glob import glob
from typing import Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from flax import jax_utils
from flax.traverse_util import flatten_dict, unflatten_dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
    base: int = 48

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


class SimpleUQDataset:
    def __init__(self, root: str):
        self.paths = sorted(glob(os.path.join(root, "*.npz")))
        if not self.paths:
            raise RuntimeError(f"No npz found in {root}")
        xs, ys, sigmas = [], [], []
        for p in self.paths:
            z = np.load(p)
            xs.append(z["x"].astype(np.float32)[None])
            ys.append(z["y"].astype(np.float32)[None])
            sigmas.append(z["sigma"].astype(np.float32)[None])
        self.x = np.stack(xs, axis=0)  # (N,1,H,W)
        self.y = np.stack(ys, axis=0)
        self.sigma = np.stack(sigmas, axis=0)

    def __len__(self):
        return self.x.shape[0]

    def get_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x[indices], self.y[indices], self.sigma[indices]


class TrainState(train_state.TrainState):
    pass


def _mask_by_name(params, allow_substr: str):
    flat = flatten_dict(params)
    mask_flat = {}
    for k in flat.keys():
        key_str = "/".join([str(p) for p in k])
        mask_flat[k] = allow_substr in key_str
    return unflatten_dict(mask_flat)


def main():
    p = argparse.ArgumentParser("Train simple UQ model (JAX)")
    p.add_argument("--data_dir", type=str, default="./simple_data_uq/data")
    p.add_argument("--ckpt_dir", type=str, default="./simple_data_uq/ckpts_jax")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--logvar_min", type=float, default=-8.0)
    p.add_argument("--logvar_max", type=float, default=3.0)
    p.add_argument("--var_reg", type=float, default=1e-4)
    p.add_argument("--sigma_weight", type=float, default=0.2, help="initial sigma supervision weight")
    p.add_argument("--sigma_weight_end", type=float, default=0.6, help="final sigma supervision weight")
    p.add_argument("--sigma_warmup_epochs", type=int, default=20, help="epochs to ramp sigma_weight")
    p.add_argument("--debug_every", type=int, default=5)
    p.add_argument("--calib_bins", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", type=int, default=48)
    args = p.parse_args()

    np.random.seed(args.seed)

    devices = jax.devices()
    n_devices = jax.local_device_count()
    print(f"[jax] devices={devices}")
    print(f"[jax] local_device_count={n_devices}")

    if args.batch_size % n_devices != 0:
        raise ValueError(f"batch_size={args.batch_size} must be divisible by n_devices={n_devices}")

    train_ds = SimpleUQDataset(os.path.join(args.data_dir, "train"))
    val_ds = SimpleUQDataset(os.path.join(args.data_dir, "val"))

    model = UNetUQ(base=args.base)
    rng = jax.random.PRNGKey(args.seed)
    dummy = jnp.zeros((1, 80, 80, 1), dtype=jnp.float32)
    params = model.init(rng, dummy)["params"]

    tx = optax.adam(args.lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state_rep = jax_utils.replicate(state)

    def loss_fn(params, x, y, sigma, sigma_w):
        mu, logvar = model.apply({"params": params}, x)
        nll, var, diff = gaussian_nll(mu, logvar, y, args.logvar_min, args.logvar_max)
        std = jnp.sqrt(var)
        sigma_loss = jnp.mean(jnp.abs(std - sigma))
        reg = jnp.mean(logvar * logvar)
        loss = nll + args.var_reg * reg + sigma_w * sigma_loss

        mse = jnp.mean(diff * diff)
        mae = jnp.mean(jnp.abs(diff))
        var_mean = jnp.mean(var)
        calib_ratio = mse / (var_mean + 1e-8)
        corr = _corrcoef_flat(var, diff * diff)
        sigma_mae = jnp.mean(jnp.abs(std - sigma))
        sigma_corr = _corrcoef_flat(var, sigma * sigma)
        metrics = {
            "loss": loss,
            "mse": mse,
            "mae": mae,
            "nll": nll,
            "var_mean": var_mean,
            "calib_ratio": calib_ratio,
            "corr": corr,
            "sigma_mae": sigma_mae,
            "sigma_corr": sigma_corr,
        }
        return loss, metrics

    def train_step(state, x, y, sigma, sigma_w):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, x, y, sigma, sigma_w)
        grads = jax.lax.pmean(grads, axis_name="devices")
        metrics = jax.lax.pmean(metrics, axis_name="devices")
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics

    p_train_step = jax.pmap(train_step, axis_name="devices", donate_argnums=(0,))

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
        "sigma_weight",
    ]

    step = 0
    for epoch in range(1, args.epochs + 1):
        # sigma weight schedule (linear ramp)
        if args.sigma_warmup_epochs <= 0:
            sigma_w = args.sigma_weight_end
        else:
            t = min(max(epoch - 1, 0), args.sigma_warmup_epochs)
            sigma_w = args.sigma_weight + (args.sigma_weight_end - args.sigma_weight) * (t / args.sigma_warmup_epochs)
        # shuffle
        idx = np.random.permutation(len(train_ds))
        n_batches = len(idx) // args.batch_size
        for bi in range(n_batches):
            batch_idx = idx[bi * args.batch_size : (bi + 1) * args.batch_size]
            x_np, y_np, s_np = train_ds.get_batch(batch_idx)
            x = jnp.asarray(x_np)
            y = jnp.asarray(y_np)
            sigma = jnp.asarray(s_np)

            # NCHW -> NHWC
            x = _to_nhwc(x)
            y = _to_nhwc(y)
            sigma = _to_nhwc(sigma)

            # shard
            x = x.reshape((n_devices, -1) + x.shape[1:])
            y = y.reshape((n_devices, -1) + y.shape[1:])
            sigma = sigma.reshape((n_devices, -1) + sigma.shape[1:])

            sigma_w_b = jnp.asarray(sigma_w, dtype=jnp.float32)
            sigma_w_b = jnp.broadcast_to(sigma_w_b, (n_devices,))
            state_rep, metrics = p_train_step(state_rep, x, y, sigma, sigma_w_b)
            m = {k: float(jax.device_get(v)[0]) for k, v in metrics.items()}

            row = {
                "time": time.time(),
                "epoch": epoch,
                "step": step,
                "loss": m["loss"],
                "mse": m["mse"],
                "mae": m["mae"],
                "nll": m["nll"],
                "var_mean": m["var_mean"],
                "calib_ratio": m["calib_ratio"],
                "corr": m["corr"],
                "sigma_mae": m["sigma_mae"],
                "sigma_corr": m["sigma_corr"],
                "sigma_weight": float(sigma_w),
            }
            csv_append(metrics_csv, row, header)
            step += 1

        # debug visualization
        if args.debug_every > 0 and epoch % args.debug_every == 0:
            out_dir = os.path.join(args.ckpt_dir, "debug", f"epoch_{epoch:03d}")
            _ensure_dir(out_dir)

            # take first val batch
            batch_idx = np.arange(min(args.batch_size, len(val_ds)))
            x_np, y_np, s_np = val_ds.get_batch(batch_idx)
            x = jnp.asarray(x_np)
            y = jnp.asarray(y_np)
            sigma = jnp.asarray(s_np)

            x_n = _to_nhwc(x)
            y_n = _to_nhwc(y)
            sigma_n = _to_nhwc(sigma)

            state_single = jax_utils.unreplicate(state_rep)
            mu, logvar = model.apply({"params": state_single.params}, x_n)
            logvar = jnp.clip(logvar, args.logvar_min, args.logvar_max)
            var = jnp.exp(logvar) + 1e-6
            std = jnp.sqrt(var)

            mu_np = np.array(jax.device_get(mu))
            std_np = np.array(jax.device_get(std))

            # first sample
            x0 = x_np[0, 0]
            y0 = y_np[0, 0]
            mu0 = mu_np[0, ..., 0]
            std0 = std_np[0, ..., 0]
            sig0 = s_np[0, 0]
            err0 = np.abs(mu0 - y0)

            save_pair(os.path.join(out_dir, "pred_gt.png"), mu0, y0, "Pred mean", "GT emitter")
            save_pair(os.path.join(out_dir, "uq_vs_sigma.png"), std0, sig0, "Pred std", "True sigma")
            save_triplet(os.path.join(out_dir, "input_pred_gt.png"), x0, mu0, y0, ("Input", "Pred", "GT"))
            save_pair(os.path.join(out_dir, "error_vs_sigma.png"), err0, sig0, "Abs error", "True sigma")

            # calibration curve on val
            all_var = []
            all_err2 = []
            for start in range(0, len(val_ds), args.batch_size):
                batch_idx = np.arange(start, min(start + args.batch_size, len(val_ds)))
                x_np, y_np, _ = val_ds.get_batch(batch_idx)
                x_n = _to_nhwc(jnp.asarray(x_np))
                y_n = _to_nhwc(jnp.asarray(y_np))
                mu, logvar = model.apply({"params": state_single.params}, x_n)
                logvar = jnp.clip(logvar, args.logvar_min, args.logvar_max)
                var = jnp.exp(logvar) + 1e-6
                diff = mu - y_n
                all_var.append(np.array(jax.device_get(var)))
                all_err2.append(np.array(jax.device_get(diff * diff)))
            all_var = np.concatenate(all_var, axis=0)
            all_err2 = np.concatenate(all_err2, axis=0)
            curve = calibration_curve(all_var, all_err2, bins=int(args.calib_bins))
            plot_calibration(curve, os.path.join(out_dir, "calibration.png"), "Calibration (val)")

        # save curves
        plot_metrics_curves(metrics_csv, os.path.join(args.ckpt_dir, "metrics_curves.png"))

    print("[done] training finished")


if __name__ == "__main__":
    main()
'''
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
  uv run python simple_data_uq/train_simple_uq_jax.py \
  --data_dir "./simple_data_uq/data" \
  --ckpt_dir "./simple_data_uq/ckpts_jax" \
  --epochs 50 --batch_size 28 --lr 1e-3 \
  --debug_every 5 \
  --sigma_weight 0.2 --base 48
'''
