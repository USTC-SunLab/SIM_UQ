# train_selfsup_uq.py
# -*- coding: utf-8 -*-
"""
核心训练脚本（pmap 多卡）：
- 基于 uq_train.py 的新脚本，不修改原文件
- 用异方差回归（heteroscedastic regression）做**数据不确定度**量化
- 自监督：仅有 SIM 原图作为观测，使用重建误差的 NLL 训练

模型输出约定：
  rec, light_pattern, emitter, psf, mask, deconv = model(x, ...)
新增输出：
  emitter_logvar, lp_logvar, psf_logvar （用于不确定度）

Loss:
  rec_nll（异方差高斯 NLL，来自 rec vs input）
 + TV / center 等正则
 + 可选 rec_aux（L1/MS-SSIM）稳定训练
"""

from __future__ import annotations

import os
import sys

import pickle
import csv
import glob
import time
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Mapping

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils_metrics import ms_ssim_3d

import numpy as np
import torch
from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
from jax import lax
import optax
from flax.training import train_state, checkpoints
from flax import jax_utils
from flax import linen as nn
from flax.core import freeze, unfreeze

from skimage.io import imread
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import uq_data
from uq_data import dataset_2d_sim_supervised

import uq_args
from uq_data_uq_utils import (
    clip_logvar,
    logvar_to_var,
    gaussian_nll_from_var,
    product_variance,
    propagate_rec_variance,
    downsample_variance,
)
from uq_vis import to_chw, vol_to_2d


# =============================================================================
# 0) training crop: 静默版本（避免刷屏）
# =============================================================================
def _norm_5d_leading_dims(t):
    """把 (1,B,...) 纠正成 (B,1,...)；其它情况原样返回。"""
    if t is None or t.ndim != 5:
        return t
    if (t.shape[0] == 1) and (t.shape[1] != 1):
        t = jnp.transpose(t, (1, 0, 2, 3, 4))
    return t


def _to_b1_5d(t):
    """
    统一把输入变成 (B,1,...) 的 5D：
    - 4D: (B,Z,Y,X) -> (B,1,Z,Y,X)
    - 5D: 修正 (1,B,...) -> (B,1,...)；其余不动
    - 3D: (Z,Y,X) -> (1,1,Z,Y,X) 兜底
    """
    if t is None:
        return None
    if t.ndim == 5:
        return _norm_5d_leading_dims(t)
    if t.ndim == 4:
        return t[:, None, ...]
    if t.ndim == 3:
        return t[None, None, ...]
    return t


def _align_like(target, t):
    """
    把 t 的维度/前两维布局尽量对齐到 target（只处理你描述的 4D/5D + 单通道情形）。
    - target 5D: 期望 t 也变成 5D (B,1,Z,Y,X)
    - target 4D: 期望 t 也变成 4D (B,Z,Y,X)
    """
    if t is None:
        return None

    target = _norm_5d_leading_dims(target)
    t = _norm_5d_leading_dims(t)

    if target.ndim == t.ndim:
        return t

    # target 是 5D，而 t 是 4D -> 给 t 加单通道维
    if target.ndim == 5 and t.ndim == 4:
        return t[:, None, ...]  # (B,Z,Y,X) -> (B,1,Z,Y,X)

    # target 是 4D，而 t 是 5D -> 去掉单通道维（优先 squeeze axis=1）
    if target.ndim == 4 and t.ndim == 5:
        if t.shape[1] == 1:
            return jnp.squeeze(t, axis=1)  # (B,1,Z,Y,X) -> (B,Z,Y,X)
        if t.shape[0] == 1:
            return jnp.squeeze(t, axis=0)  # (1,B,Z,Y,X) -> (B,Z,Y,X)
        return t

    # 额外兜底：偶尔可能是 3D (Z,Y,X)
    if target.ndim == 5 and t.ndim == 3:
        return t[None, None, ...]  # -> (1,1,Z,Y,X)
    if target.ndim == 4 and t.ndim == 3:
        return t[None, ...]  # -> (1,Z,Y,X)

    return t


def _replace_dir_token(path: str, token: str, repl: str, fallback_token: str, fallback_repl: str) -> str:
    if token in path:
        return path.replace(token, repl)
    if fallback_token in path:
        return path.replace(fallback_token, fallback_repl)
    return path


def derive_gt_paths(
    paths: List[str],
    gt_dir_token: str,
    gt_dir_repl: str,
    gt_emitter_suffix: str,
    gt_lp_suffix: str,
) -> List[Tuple[str, str]]:
    out = []
    for p in paths:
        base = _replace_dir_token(p, gt_dir_token, gt_dir_repl, "/val/", "/val_gt/")
        root, ext = os.path.splitext(base)
        emitter_path = root + gt_emitter_suffix + ext
        lp_path = root + gt_lp_suffix + ext
        out.append((emitter_path, lp_path))
    return out


def _downsample_hw_like(gt: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    Downsample GT spatially to match target (expects 5D (B,1,Z,H,W)).
    Uses avg_pool for integer factors, else resize.
    """
    gt = _to_b1_5d(gt)
    target = _to_b1_5d(target)
    th, tw = int(target.shape[-2]), int(target.shape[-1])
    gh, gw = int(gt.shape[-2]), int(gt.shape[-1])
    if (gh, gw) == (th, tw):
        return gt

    if gh % th == 0 and gw % tw == 0:
        s0 = gh // th
        s1 = gw // tw
        g = jnp.transpose(gt, (0, 2, 3, 4, 1))  # (B,Z,H,W,1)
        g = nn.avg_pool(g, window_shape=(1, s0, s1), strides=(1, s0, s1), padding="VALID")
        g = jnp.transpose(g, (0, 4, 1, 2, 3))
        return g

    g0 = jnp.asarray(gt[0, 0], dtype=jnp.float32)
    g1 = jax.image.resize(g0, shape=(int(gt.shape[2]), th, tw), method="linear")
    return g1[None, None, ...]


def _calibration_curve_np(pred_var: np.ndarray, err2: np.ndarray, bins: int = 10):
    pv = np.asarray(pred_var, dtype=np.float32).reshape(-1)
    e2 = np.asarray(err2, dtype=np.float32).reshape(-1)
    if pv.size == 0 or e2.size == 0:
        return None, None
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


def _format_param_paths(paths, max_items: int = 5) -> str:
    if not paths:
        return ""
    show = ["/".join([str(p) for p in path]) for path in paths[:max_items]]
    more = f" (+{len(paths) - max_items} more)" if len(paths) > max_items else ""
    return ", ".join(show) + more


def _merge_pretrained_params(init_params, loaded_params):
    if loaded_params is None:
        return init_params, {"missing": [], "unexpected": [], "mismatched": []}

    init = unfreeze(init_params)
    loaded = unfreeze(loaded_params)
    missing = []
    unexpected = []
    mismatched = []

    def _merge(i, l, path):
        if isinstance(i, Mapping):
            if not isinstance(l, Mapping):
                mismatched.append(path)
                return i
            out = {}
            for k, v in i.items():
                if k in l:
                    out[k] = _merge(v, l[k], path + (k,))
                else:
                    missing.append(path + (k,))
                    out[k] = v
            for k in l.keys():
                if k not in i:
                    unexpected.append(path + (k,))
            return out

        if isinstance(l, Mapping):
            mismatched.append(path)
            return i

        try:
            if hasattr(i, "shape") and hasattr(l, "shape") and i.shape != l.shape:
                mismatched.append(path)
                return i
        except Exception:
            mismatched.append(path)
            return i
        return l

    merged = _merge(init, loaded, ())
    report = {"missing": missing, "unexpected": unexpected, "mismatched": mismatched}
    return freeze(merged), report


def _normalize_pretrain_tree_for_data_uq(tree, label: str):
    if tree is None:
        return tree
    if isinstance(tree, Mapping):
        if "base" not in tree and any(k in tree for k in ("pt_predictor", "PSF_predictor", "psf_seed")):
            print(f"[pretrain] wrapping {label} under 'base' for DataUQ model")
            return {"base": tree}
    return tree


# =============================================================================
# args (extend uq_args)
# =============================================================================
def parse_args(argv: Optional[List[str]] = None):
    p = uq_args.build_parser()
    p.add_argument("--val_glob", type=str, default="", help="optional val glob for evaluation")
    p.add_argument("--use_gt_metrics", action="store_true", help="use GT for debug/metrics if available")
    p.add_argument("--init_from_ckpt", type=str, default="", help="init params/batch_stats from a plain ckpt dir")
    p.add_argument("--freeze_base", action="store_true", help="freeze base model; train logvar heads only")
    p.add_argument(
        "--loss_mode",
        type=str,
        default="selfsup_uq",
        choices=["selfsup_uq", "emitter_uq", "plain"],
        help="training loss mode: selfsup_uq | emitter_uq | plain",
    )
    p.add_argument(
        "--emitter_sup_weight",
        type=float,
        default=0.0,
        help="supervised emitter NLL weight (requires GT + uq_emitter)",
    )
    p.add_argument(
        "--rec_loss_weight",
        type=float,
        default=1.0,
        help="weight for recon NLL (selfsup_uq only; ignored in plain/emitter_uq)",
    )
    p.add_argument("--ckpt_dir_fixed", action="store_true", help="do not auto-append timestamp to ckpt_dir")
    p.add_argument("--gt_dir_token", type=str, default="/train/")
    p.add_argument("--gt_dir_repl", type=str, default="/train_gt/")
    p.add_argument("--gt_emitter_suffix", type=str, default="")
    p.add_argument("--gt_lp_suffix", type=str, default="_lp")
    p.add_argument("--meta_dir_token", type=str, default="/train/")
    p.add_argument("--meta_dir_repl", type=str, default="/train_meta/")

    # ---- data uncertainty / heteroscedastic ----
    p.add_argument("--uq_logvar_min", type=float, default=-10.0, help="min clamp for log-variance")
    p.add_argument("--uq_logvar_max", type=float, default=3.0, help="max clamp for log-variance")
    p.add_argument("--uq_logvar_init", type=float, default=-4.0, help="init bias for logvar heads")
    p.add_argument("--uq_var_eps", type=float, default=1e-6, help="epsilon added to variance")
    p.add_argument("--uq_var_floor", type=float, default=1e-6, help="minimum variance after propagation")
    p.add_argument("--uq_var_reg", type=float, default=0.0, help="L2 reg weight for logvar maps")
    p.add_argument("--uq_psf_var_weight", type=float, default=1.0, help="weight for psf variance terms")
    p.add_argument("--uq_aux_rec_weight", type=float, default=0.0, help="aux rec loss weight (L1/MS-SSIM)")
    p.add_argument(
        "--uq_targets",
        type=str,
        default="emitter,lp,psf",
        help="comma-separated subset of {emitter,lp,psf} to enable UQ (use 'all' or 'none')",
    )
    p.add_argument(
        "--uq_aux_rec_mode",
        type=str,
        default="l1+ms-ssim",
        choices=["l1", "ms-ssim", "l1+ms-ssim"],
        help="aux rec loss type",
    )

    # default for HR=2x datasets
    p.set_defaults(rescale=(2, 2))
    args = p.parse_args(argv)

    # normalize tuples
    args.rescale = tuple(args.rescale)
    args.crop_size = tuple(args.crop_size)

    # parse UQ targets
    raw = str(args.uq_targets).strip().lower()
    allowed = {"emitter", "lp", "psf"}
    if raw in ("", "all", "*"):
        targets = allowed
    elif raw in ("none", "off", "0"):
        targets = set()
    else:
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        targets = set(parts)
        invalid = targets - allowed
        if invalid:
            raise ValueError(f"Invalid --uq_targets: {sorted(invalid)} (allowed: {sorted(allowed)})")
    args.uq_emitter = "emitter" in targets
    args.uq_lp = "lp" in targets
    args.uq_psf = "psf" in targets

    # normalize loss mode + adjust flags
    args.loss_mode = str(args.loss_mode).strip().lower()
    if args.loss_mode == "plain":
        if float(args.emitter_sup_weight) > 0:
            print("[warn] loss_mode=plain ignores emitter_sup_weight; set to 0")
            args.emitter_sup_weight = 0.0
        if float(args.rec_loss_weight) <= 0:
            print("[warn] loss_mode=plain uses rec_loss; forcing rec_loss_weight=1.0")
            args.rec_loss_weight = 1.0
        if float(args.uq_aux_rec_weight) > 0:
            print("[warn] loss_mode=plain ignores uq_aux_rec_weight; set to 0")
            args.uq_aux_rec_weight = 0.0
        args.uq_emitter = False
        args.uq_lp = False
        args.uq_psf = False
    elif args.loss_mode == "emitter_uq":
        if not args.uq_emitter:
            raise ValueError("loss_mode=emitter_uq requires uq_targets to include 'emitter'")
        if float(args.emitter_sup_weight) <= 0:
            print("[warn] loss_mode=emitter_uq sets emitter_sup_weight=1.0")
            args.emitter_sup_weight = 1.0
        if float(args.rec_loss_weight) != 0:
            print("[warn] loss_mode=emitter_uq sets rec_loss_weight=0")
            args.rec_loss_weight = 0.0
        if not bool(args.use_gt_metrics):
            print("[warn] loss_mode=emitter_uq enables --use_gt_metrics for debug/metrics")
            args.use_gt_metrics = True

    if float(args.emitter_sup_weight) > 0 and not args.uq_emitter:
        raise ValueError("--emitter_sup_weight requires uq_targets to include 'emitter'")

    # auto ckpt dir (timestamped)
    if not bool(args.ckpt_dir_fixed):
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.ckpt_dir = os.path.join(str(args.ckpt_dir), ts)

    # abs paths + debug_dir default
    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not str(args.debug_dir).strip():
        args.debug_dir = os.path.join(args.ckpt_dir, "debug_samples")
    args.debug_dir = os.path.abspath(args.debug_dir)

    if str(args.init_from_ckpt).strip() and args.resume_pickle is not None:
        print("[warn] init_from_ckpt set; ignore resume_pickle pretrain")
        args.resume_pickle = None
    return args


def rec_loss(x, rec, mask=None):
    if mask is None:
        mask = jnp.ones_like(x)
    l1_loss = jnp.abs((rec - x))
    l1_loss = (l1_loss * mask).sum() / mask.sum()
    x_norm = (x - x.min()) / (x.max() - x.min())
    rec_norm = (rec - x.min()) / (x.max() - x.min())
    ms_ssim_loss = jnp.mean(1 - ms_ssim_3d(x_norm, rec_norm, win_size=5))
    loss = 0.875 * l1_loss + 0.125 * ms_ssim_loss
    return loss


def _make_logvar_mask(params, args) -> Any:
    allowed = set()
    if bool(getattr(args, "uq_emitter", False)):
        allowed.add("emitter_logvar_decoder")
    if bool(getattr(args, "uq_lp", False)):
        allowed.add("lp_logvar_head")
    if bool(getattr(args, "uq_psf", False)):
        allowed.add("psf_logvar_head")

    def match(path) -> bool:
        for k in path:
            if isinstance(k, str) and k in allowed:
                return True
        return False

    mask = jax.tree_util.tree_map_with_path(lambda path, _: match(path), params)
    return freeze(unfreeze(mask))


def TV_Loss(img):
    img = img.reshape([-1, img.shape[-3], img.shape[-2], img.shape[-1]])
    img = img / img.mean()
    batch_size = img.shape[0]
    z, y, x = img.shape[1:4]

    def _tensor_size(t):
        return t.shape[-3] * t.shape[-2] * t.shape[-1]

    cz = _tensor_size(img[:, 1:, :, :])
    cy = _tensor_size(img[:, :, 1:, :])
    cx = _tensor_size(img[:, :, :, 1:])

    hz = lax.pow(jnp.abs(img[:, 1:, :, :] - img[:, : z - 1, :, :]), 2.0).sum()
    hy = lax.pow(jnp.abs(img[:, :, 1:, :] - img[:, :, : y - 1, :]), 2.0).sum()
    hx = lax.pow(jnp.abs(img[:, :, :, 1:] - img[:, :, :, : x - 1]), 2.0).sum()

    if cz == 0:
        return (hy / cy + hx / cx) / batch_size
    else:
        return (hz / cz + hy / cy + hx / cx) / batch_size


# =============================================================================
# 1) small utils
# =============================================================================
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_bchw_9(x: np.ndarray) -> np.ndarray:
    """
    确保输入是 (B,9,H,W)
    支持：
      - (B,9,H,W)
      - (B,H,W,9)
      - (9,H,W)
      - (H,W,9)
      - (B,1,9,H,W) -> squeeze
    """
    x = np.asarray(x)
    if x.ndim == 5:
        # (B,1,9,H,W) -> (B,9,H,W)
        if x.shape[1] == 1:
            x = x[:, 0]
        else:
            x = x[:, 0]

    if x.ndim == 4:
        if x.shape[1] == 9:
            return x
        if x.shape[-1] == 9:
            return np.transpose(x, (0, 3, 1, 2))
        raise ValueError(f"Expected 9 channels, got shape {x.shape}")

    if x.ndim == 3:
        if x.shape[0] == 9:
            return x[None, ...]
        if x.shape[-1] == 9:
            return np.transpose(x[None, ...], (0, 3, 1, 2))
        raise ValueError(f"Expected 9 channels, got shape {x.shape}")

    raise ValueError(f"Unexpected input ndim={x.ndim}, shape={x.shape}")


def shard_batch(x: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    b = x.shape[0]
    if b % n_devices != 0:
        raise ValueError(f"batch_size={b} must be divisible by n_devices={n_devices}")
    return x.reshape((n_devices, b // n_devices) + x.shape[1:])


def psnr_from_mse(mse: jnp.ndarray, max_val: float = 1.0, eps: float = 1e-8) -> jnp.ndarray:
    return 20.0 * jnp.log10(max_val) - 10.0 * jnp.log10(mse + eps)


def csv_append(path: str, row: Dict[str, Any], header: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# =============================================================================
# 1.5) debug visualization helpers
# =============================================================================
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _pct_vmin_vmax(arr: np.ndarray, pmin: float, pmax: float):
    a = np.asarray(arr).astype(np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, None
    vmin = float(np.percentile(a, pmin))
    vmax = float(np.percentile(a, pmax))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def _safe_float(v) -> float:
    try:
        if v is None or v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _plot_metric_curves(csv_path: str, out_dir: str):
    if not os.path.exists(csv_path):
        return
    metrics = ["loss", "rec_nll", "rec_mse", "psnr", "emitter_nll"]
    data = {"train": {m: [] for m in metrics}, "val": {m: [] for m in metrics}}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = str(row.get("note", "")).strip()
            if note not in ("train_epoch", "val_epoch"):
                continue
            split = "train" if note == "train_epoch" else "val"
            epoch = _safe_float(row.get("epoch", ""))
            if not np.isfinite(epoch):
                continue
            for m in metrics:
                val = _safe_float(row.get(m, ""))
                if np.isfinite(val):
                    data[split][m].append((epoch, val))
    os.makedirs(out_dir, exist_ok=True)
    for m in metrics:
        t = data["train"][m]
        v = data["val"][m]
        if not t and not v:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        if t:
            xs, ys = zip(*t)
            ax.plot(xs, ys, label="train")
        if v:
            xs, ys = zip(*v)
            ax.plot(xs, ys, label="val")
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{m}.png"), dpi=150)
        plt.close(fig)

    # compact grid for quick glance
    grid_metrics = ["loss", "rec_nll", "psnr", "emitter_nll"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.reshape(-1)
    for i, m in enumerate(grid_metrics):
        ax = axes[i]
        t = data["train"][m]
        v = data["val"][m]
        if t:
            xs, ys = zip(*t)
            ax.plot(xs, ys, label="train")
        if v:
            xs, ys = zip(*v)
            ax.plot(xs, ys, label="val")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_grid.png"), dpi=150)
    plt.close(fig)


def _sym_vmin_vmax(arr: np.ndarray, p: float):
    a = np.asarray(arr).astype(np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(a), p))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6
    return -vmax, vmax


def _save_map_png(
    path: str,
    img2d: np.ndarray,
    title: str,
    pmin: float,
    pmax: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
):
    _ensure_dir(path)
    if vmin is None or vmax is None:
        vmin, vmax = _pct_vmin_vmax(img2d, pmin, pmax)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(img2d, vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_frames_grid(
    path: str,
    frames_any: np.ndarray,
    title: str,
    max_frames: int,
    pmin: float,
    pmax: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
):
    _ensure_dir(path)
    frames = to_chw(frames_any)  # (C,H,W)
    n = int(min(frames.shape[0], max_frames))
    if n <= 0:
        return
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    if vmin is None or vmax is None:
        vmin, vmax = _pct_vmin_vmax(frames[:n], pmin, pmax)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(frames[i], vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap)
            ax.set_title(f"{title}[{i}]")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_debug_visuals(
    out_dir: str,
    args: Any,
    x_np: np.ndarray,
    rec_np: np.ndarray,
    lp_np: np.ndarray,
    emitter_np: np.ndarray,
    psf_np: np.ndarray,
    deconv_np: Optional[np.ndarray],
    mask_np: Optional[np.ndarray],
    emitter_std_np: Optional[np.ndarray],
    rec_std_np: Optional[np.ndarray],
    gt_pack: Optional[Dict[str, Any]],
):
    pmin = float(getattr(args, "vis_pmin", 1.0))
    pmax = float(getattr(args, "vis_pmax", 99.0))
    max_frames = int(getattr(args, "debug_frames", 3))

    os.makedirs(out_dir, exist_ok=True)

    # core inputs/outputs
    _save_frames_grid(os.path.join(out_dir, "input.png"), x_np, "input", max_frames, pmin, pmax)
    try:
        x_mean = np.asarray(x_np)
        if x_mean.ndim >= 5:
            x_mean = x_mean[0, 0]
        if x_mean.ndim == 3:
            x_mean = x_mean.mean(axis=0)
        _save_map_png(os.path.join(out_dir, "input_mean.png"), x_mean, "input_mean", pmin, pmax)
    except Exception:
        pass
    _save_frames_grid(os.path.join(out_dir, "recon.png"), rec_np, "recon", max_frames, pmin, pmax)
    try:
        rec_mean = np.asarray(rec_np)
        if rec_mean.ndim >= 5:
            rec_mean = rec_mean[0, 0]
        if rec_mean.ndim == 3:
            rec_mean = rec_mean.mean(axis=0)
        _save_map_png(os.path.join(out_dir, "recon_mean.png"), rec_mean, "recon_mean", pmin, pmax)
    except Exception:
        pass
    _save_frames_grid(
        os.path.join(out_dir, "recon_err.png"),
        (rec_np - x_np),
        "recon_err",
        max_frames,
        pmin,
        pmax,
        vmin=_sym_vmin_vmax(rec_np - x_np, pmax)[0],
        vmax=_sym_vmin_vmax(rec_np - x_np, pmax)[1],
        cmap="seismic",
    )
    try:
        err_mean = np.asarray(rec_np - x_np)
        if err_mean.ndim >= 5:
            err_mean = err_mean[0, 0]
        if err_mean.ndim == 3:
            err_mean = err_mean.mean(axis=0)
        _save_map_png(
            os.path.join(out_dir, "recon_err_mean.png"),
            err_mean,
            "recon_err_mean",
            pmin,
            pmax,
            vmin=_sym_vmin_vmax(err_mean, pmax)[0],
            vmax=_sym_vmin_vmax(err_mean, pmax)[1],
            cmap="seismic",
        )
    except Exception:
        pass

    # predictions
    _save_frames_grid(os.path.join(out_dir, "lp_pred.png"), lp_np, "lp_pred", max_frames, pmin, pmax)
    try:
        lp_mean = np.asarray(lp_np)
        if lp_mean.ndim >= 5:
            lp_mean = lp_mean[0, 0]
        if lp_mean.ndim == 3:
            lp_mean = lp_mean.mean(axis=0)
        _save_map_png(os.path.join(out_dir, "lp_mean.png"), lp_mean, "lp_mean", pmin, pmax)
    except Exception:
        pass
    _save_map_png(os.path.join(out_dir, "emitter_mu_pred.png"), vol_to_2d(emitter_np, mode="zmid"), "emitter_mu_pred", pmin, pmax)
    _save_map_png(os.path.join(out_dir, "psf.png"), vol_to_2d(psf_np, mode="zmid"), "psf", pmin, pmax)
    if deconv_np is not None:
        _save_frames_grid(
            os.path.join(out_dir, "lightfield_pred.png"),
            deconv_np,
            "lightfield_pred",
            max_frames,
            pmin,
            pmax,
        )
    elif lp_np is not None:
        try:
            emitter_real = np.asarray(emitter_np) * np.asarray(lp_np)
            _save_frames_grid(
                os.path.join(out_dir, "lightfield_pred.png"),
                emitter_real,
                "lightfield_pred",
                max_frames,
                pmin,
                pmax,
            )
        except Exception:
            pass
    if mask_np is not None:
        _save_frames_grid(os.path.join(out_dir, "mask.png"), mask_np, "mask", max_frames, pmin, pmax)

    # labels (use GT when available; skip if identical)
    gt_em = gt_pack.get("emitter_gt") if gt_pack is not None else None
    gt_lp = gt_pack.get("lp_gt") if gt_pack is not None else None
    if gt_em is not None:
        _save_map_png(os.path.join(out_dir, "emitter_real_label.png"), vol_to_2d(gt_em, mode="zmid"), "emitter_real_label", pmin, pmax)
        _save_map_png(
            os.path.join(out_dir, "emitter_diff.png"),
            vol_to_2d(emitter_np - gt_em, mode="zmid"),
            "emitter_diff",
            pmin,
            pmax,
            vmin=_sym_vmin_vmax(emitter_np - gt_em, pmax)[0],
            vmax=_sym_vmin_vmax(emitter_np - gt_em, pmax)[1],
            cmap="seismic",
        )
    if gt_lp is not None:
        _save_frames_grid(os.path.join(out_dir, "lp_label.png"), gt_lp, "lp_label", max_frames, pmin, pmax)
        _save_frames_grid(
            os.path.join(out_dir, "lp_diff.png"),
            (lp_np - gt_lp),
            "lp_diff",
            max_frames,
            pmin,
            pmax,
            vmin=_sym_vmin_vmax(lp_np - gt_lp, pmax)[0],
            vmax=_sym_vmin_vmax(lp_np - gt_lp, pmax)[1],
            cmap="seismic",
        )
    if gt_em is not None and gt_lp is not None:
        try:
            lightfield_gt = np.asarray(gt_em) * np.asarray(gt_lp)
            _save_frames_grid(
                os.path.join(out_dir, "lightfield_label.png"),
                lightfield_gt,
                "lightfield_label",
                max_frames,
                pmin,
                pmax,
            )
        except Exception:
            pass

    # UQ maps
    if emitter_std_np is not None:
        _save_map_png(os.path.join(out_dir, "emitter_std.png"), vol_to_2d(emitter_std_np, mode="zmid"), "emitter_std", pmin, pmax)
    if rec_std_np is not None:
        _save_frames_grid(os.path.join(out_dir, "recon_std.png"), rec_std_np, "recon_std", max_frames, pmin, pmax)

# =============================================================================
# 2) PSF regularizers (JAX)
# =============================================================================
def psf_mass_2d(psf: jnp.ndarray) -> jnp.ndarray:
    """把 psf 投影到 2D mass map，并归一化到 sum=1，输出 (B,H,W)。"""
    psf = jnp.asarray(psf).astype(jnp.float32)

    if psf.ndim == 2:
        mass = psf[None, ...]
    elif psf.ndim == 3:
        mass = jnp.sum(psf, axis=0, keepdims=True)
    elif psf.ndim == 4:
        mass = jnp.sum(psf, axis=1)
    elif psf.ndim == 5:
        mass = jnp.sum(psf, axis=(1, 2))
    else:
        axes = tuple(range(psf.ndim - 2))
        mass2d = jnp.sum(psf, axis=axes)
        mass = mass2d[None, ...]

    mass = jnp.maximum(mass, 0.0)
    mass = mass / (jnp.sum(mass, axis=(-2, -1), keepdims=True) + 1e-8)
    if mass.ndim == 2:
        mass = mass[None, ...]
    return mass


def tv_loss(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x)
    dh = jnp.abs(x[..., 1:, :] - x[..., :-1, :])
    dw = jnp.abs(x[..., :, 1:] - x[..., :, :-1])
    return jnp.mean(dh) + jnp.mean(dw)


def center_loss(psf: jnp.ndarray) -> jnp.ndarray:
    mass = psf_mass_2d(psf)  # (B,H,W)
    H, W = mass.shape[-2], mass.shape[-1]

    ys = jnp.arange(H, dtype=jnp.float32)[None, :, None]
    xs = jnp.arange(W, dtype=jnp.float32)[None, None, :]

    cy = jnp.sum(mass * ys, axis=(-2, -1))
    cx = jnp.sum(mass * xs, axis=(-2, -1))

    ty = (H - 1) * 0.5
    tx = (W - 1) * 0.5
    return jnp.mean((cy - ty) ** 2 + (cx - tx) ** 2)


def gaussian_prior_loss(psf: jnp.ndarray, sigma: float = 3.0) -> jnp.ndarray:
    mass = psf_mass_2d(psf)  # (B,H,W)
    H, W = mass.shape[-2], mass.shape[-1]

    yy = jnp.arange(H, dtype=jnp.float32)[:, None]
    xx = jnp.arange(W, dtype=jnp.float32)[None, :]
    cy = (H - 1) * 0.5
    cx = (W - 1) * 0.5

    s2 = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float32), 1e-6) ** 2
    g = jnp.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s2))
    g = g / (jnp.sum(g) + 1e-8)
    g = g[None, ...]

    return jnp.mean((mass - g) ** 2)


# =============================================================================
# 3) Data-UQ model wrapper (heteroscedastic heads)
# =============================================================================
class LogVarHead(nn.Module):
    init_logvar: float = -4.0

    @nn.compact
    def __call__(self, x):
        # x: (B,1,Z,H,W) -> (B,Z,H,W,1) -> conv -> (B,1,Z,H,W)
        x_t = jnp.transpose(x, (0, 2, 3, 4, 1))
        y = nn.Conv(
            features=1,
            kernel_size=(1, 1, 1),
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(float(self.init_logvar)),
        )(x_t)
        return jnp.transpose(y, (0, 4, 1, 2, 3))


class DataUQPiMAE(nn.Module):
    img_size: tuple[int, int]
    patch_size: tuple[int, int, int]
    psf_size: tuple[int, int]
    rank: int
    logvar_init: float = -4.0

    def setup(self):
        from network import PiMAE, Decoder

        self.base = PiMAE(self.img_size, self.patch_size, self.psf_size, self.rank)
        # emitter logvar branches from intermediate features (not from emitter output)
        self.emitter_logvar_decoder = Decoder(patch_size=self.patch_size, out_p=1)
        self.lp_logvar_head = LogVarHead(self.logvar_init, name="lp_logvar")
        self.psf_logvar_head = LogVarHead(self.logvar_init, name="psf_logvar")

    def __call__(self, x, args, training):
        base_training = training
        if bool(getattr(args, "freeze_base", False)):
            base_training = False
        rec, light_pattern, emitter, psf, mask, deconv, feat_em = self.base(
            x, args, base_training, return_features=True
        )
        emitter_logvar = self.emitter_logvar_decoder(feat_em).transpose([0, 4, 1, 2, 3]) + self.logvar_init
        lp_logvar = self.lp_logvar_head(light_pattern)
        psf_logvar = self.psf_logvar_head(psf)
        return rec, light_pattern, emitter, psf, mask, deconv, emitter_logvar, lp_logvar, psf_logvar


# =============================================================================
# 4) TrainState
# =============================================================================
class TrainState(train_state.TrainState):
    batch_stats: Any
    rng: Any


def _pmean_tree(tree):
    return jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="devices"), tree)


# =============================================================================
# 4) debug crop record (only for debug_loader num_workers=0)
# =============================================================================
DEBUG_CROP_RECORDS: List[Dict[str, Any]] = []


def _random_crop_arrays_record(arrays, h, w):
    scales = [int(x.shape[-1] / arrays[0].shape[-1]) for x in arrays]
    H, W = arrays[0].shape[-2:]
    assert h <= H and w <= W
    h0 = int(np.random.randint(0, H - h + 1))
    w0 = int(np.random.randint(0, W - w + 1))
    DEBUG_CROP_RECORDS.append(
        {
            "h0": h0,
            "w0": w0,
            "h": int(h),
            "w": int(w),
            "ref_shape": list(np.asarray(arrays[0]).shape),
            "scales": [int(s) for s in scales],
        }
    )
    out = []
    for array, scale in zip(arrays, scales):
        out.append(array[..., h0 * scale : h0 * scale + h * scale, w0 * scale : w0 * scale + w * scale])
    return out


# =============================================================================
# 5) state init + forward
# =============================================================================
def create_state(rng: jax.Array, model, args, steps_per_epoch: int) -> TrainState:
    rng, k_params, k_do, k_dp, k_rm = jax.random.split(rng, 5)

    dummy = jnp.zeros((1, 1, 9, args.crop_h, args.crop_w), dtype=jnp.float32)
    variables = model.init(
        {"params": k_params, "dropout": k_do, "drop_path": k_dp, "random_masking": k_rm},
        dummy,
        args,
        True,
    )
    params = variables["params"]
    batch_stats = variables.get("batch_stats", None)

    warmup_steps = max(1, int(7 * steps_per_epoch))
    lr_schedule = optax.linear_schedule(init_value=0.0, end_value=float(args.lr), transition_steps=warmup_steps)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=float(args.weight_decay)),
    )
    if bool(getattr(args, "freeze_base", False)):
        tx = optax.masked(tx, lambda p: _make_logvar_mask(p, args))
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats, rng=rng)


def debug_forward_single_device(state: TrainState, x: jnp.ndarray, args, debug_train: bool):
    variables = {"params": state.params}
    if state.batch_stats is not None:
        variables["batch_stats"] = state.batch_stats

    step = int(jax.device_get(state.step))
    rng = jax.random.fold_in(state.rng, step)
    rng, k_do, k_dp, k_rm = jax.random.split(rng, 4)

    outs, _ = state.apply_fn(
        variables,
        x,
        args,
        training=debug_train,
        rngs={"dropout": k_do, "drop_path": k_dp, "random_masking": k_rm},
        mutable=["batch_stats"] if state.batch_stats is not None else [],
    )
    (
        rec,
        light_pattern,
        emitter,
        psf,
        mask,
        deconv,
        emitter_logvar,
        lp_logvar,
        psf_logvar,
    ) = outs
    return rec, light_pattern, emitter, psf, mask, deconv, emitter_logvar, lp_logvar, psf_logvar


# =============================================================================
# 6) main
# =============================================================================
def main():
    args = parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    devices = jax.devices()
    n_devices = jax.local_device_count()
    device0 = devices[0]
    print(f"[jax] local_device_count={n_devices}")
    print(f"[jax] devices={devices}")

    if args.batch_size % n_devices != 0:
        raise ValueError(f"--batch_size={args.batch_size} must be divisible by n_devices={n_devices}")

    # paths
    train_paths = sorted(glob.glob(args.train_glob))
    if len(train_paths) == 0:
        raise RuntimeError(f"No train files matched: {args.train_glob}")

    # dataset/loader
    train_gt_paths = None
    train_use_gt = False
    if float(args.emitter_sup_weight) > 0 or bool(args.use_gt_metrics):
        train_gt_paths = derive_gt_paths(
            train_paths,
            args.gt_dir_token,
            args.gt_dir_repl,
            args.gt_emitter_suffix,
            args.gt_lp_suffix,
        )
        missing = [p for pair in train_gt_paths for p in pair if not os.path.exists(p)]
        if missing:
            raise RuntimeError(f"GT missing for training ({len(missing)} files).")
        train_use_gt = True

    train_ds = dataset_2d_sim_supervised(
        train_paths,
        crop_size=(args.crop_h, args.crop_w),
        use_gt=bool(train_use_gt),
        gt_paths=train_gt_paths,
    )
    pin = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # optional val loader
    val_loader = None
    val_use_gt = False
    if str(args.val_glob).strip():
        val_paths = sorted(glob.glob(args.val_glob))
        if len(val_paths) == 0:
            raise RuntimeError(f"No val files matched: {args.val_glob}")
        val_gt_paths = None
        if bool(args.use_gt_metrics):
            val_gt_paths = derive_gt_paths(
                val_paths,
                args.gt_dir_token,
                args.gt_dir_repl,
                args.gt_emitter_suffix,
                args.gt_lp_suffix,
            )
            missing = [p for pair in val_gt_paths for p in pair if not os.path.exists(p)]
            if missing:
                print(f"[warn] GT missing for val ({len(missing)} files), disable GT metrics")
                val_gt_paths = None
            else:
                val_use_gt = True

        val_ds = dataset_2d_sim_supervised(
            val_paths,
            crop_size=(args.crop_h, args.crop_w),
            use_gt=bool(val_use_gt),
            gt_paths=val_gt_paths,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )

    # optional debug loader (single file)
    debug_loader = None
    debug_path = None
    if args.debug_every_epochs and args.debug_every_epochs > 0:
        if not (0 <= args.debug_index < len(train_paths)):
            raise ValueError(f"--debug_index {args.debug_index} out of range [0, {len(train_paths)-1}]")
        debug_path = train_paths[args.debug_index]
        debug_ds = dataset_2d_sim_supervised(
            [debug_path],
            crop_size=(args.crop_h, args.crop_w),
            use_gt=False,
            gt_paths=None,
        )
        debug_loader = DataLoader(
            debug_ds,
            batch_size=args.debug_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,  # 必须 0，才能稳定记录 crop 坐标
            pin_memory=False,
        )

    # model (data-UQ wrapper)
    model = DataUQPiMAE(
        (9, args.crop_size[0] * args.rescale[0], args.crop_size[1] * args.rescale[1]),
        (3, 16, 16),
        (49, 49),
        args.lrc9_rank,
        logvar_init=float(args.uq_logvar_init),
    )

    rng = jax.random.PRNGKey(args.seed)
    steps_per_epoch = len(train_loader)
    print(f"[opt] warmup_steps={int(7 * steps_per_epoch)} (7 * steps_per_epoch)")
    state = create_state(rng, model, args, steps_per_epoch)
    print("[init] success")

    if str(args.init_from_ckpt).strip():
        init_ckpt = os.path.abspath(str(args.init_from_ckpt))
        print(f"[init_from_ckpt] loading params/batch_stats from: {init_ckpt}")
        try:
            raw = checkpoints.restore_checkpoint(init_ckpt, target=None)
        except Exception as e2:
            raise RuntimeError(f"[init_from_ckpt] failed to load checkpoint: {e2}") from e2

        if isinstance(raw, TrainState):
            loaded_params = raw.params
            loaded_bs = raw.batch_stats
        elif isinstance(raw, dict):
            loaded_params = raw.get("params", None)
            loaded_bs = raw.get("batch_stats", None)
        else:
            loaded_params = None
            loaded_bs = None

        if loaded_params is None:
            raise RuntimeError("[init_from_ckpt] checkpoint has no params to load")

        merged_params, report = _merge_pretrained_params(state.params, loaded_params)
        state = state.replace(params=merged_params)
        if loaded_bs is not None and state.batch_stats is not None:
            merged_bs, report_bs = _merge_pretrained_params(state.batch_stats, loaded_bs)
            state = state.replace(batch_stats=merged_bs)
        else:
            report_bs = {"missing": [], "unexpected": [], "mismatched": []}

        print(
            "[init_from_ckpt] partial merge done. "
            f"params missing={len(report['missing'])}, mismatched={len(report['mismatched'])}, "
            f"batch_stats missing={len(report_bs['missing'])}, mismatched={len(report_bs['mismatched'])}"
        )
        state = state.replace(params=freeze(unfreeze(state.params)))
        if state.batch_stats is not None:
            state = state.replace(batch_stats=freeze(unfreeze(state.batch_stats)))
        state = state.replace(opt_state=state.tx.init(state.params))
        print("[init_from_ckpt] optimizer state reset")

    if args.resume:
        state = checkpoints.restore_checkpoint(args.ckpt_dir, state)
        print(f"[resume] restored from {args.ckpt_dir}, step={int(jax.device_get(state.step))}")
    if args.resume_pickle is not None and not args.resume:
        print(f"\033[94mLoading BioSR pretrained checkpoint from: {args.resume_pickle}\033[0m")
        with open(args.resume_pickle, "rb") as f:
            biosr_pretrain = pickle.load(f)

        # BioSR pretrained checkpoints have 'params' and optionally 'batch_stats'
        if "params" not in biosr_pretrain:
            raise ValueError(
                f"Invalid BioSR pretrained checkpoint format. Expected 'params' key, got: {list(biosr_pretrain.keys())}"
            )

        loaded_params = _normalize_pretrain_tree_for_data_uq(biosr_pretrain["params"], "params")
        # Load params (merge to keep new params like psf_seed/logvar heads)
        merged_params, report = _merge_pretrained_params(state.params, loaded_params)
        state = state.replace(params=merged_params)

        if report["missing"]:
            print(
                f"[pretrain] missing params (kept init): {len(report['missing'])} :: "
                f"{_format_param_paths(report['missing'])}"
            )
        if report["unexpected"]:
            print(
                f"[pretrain] unexpected params (ignored): {len(report['unexpected'])} :: "
                f"{_format_param_paths(report['unexpected'])}"
            )
        if report["mismatched"]:
            print(
                f"[pretrain] shape/structure mismatch (kept init): {len(report['mismatched'])} :: "
                f"{_format_param_paths(report['mismatched'])}"
            )

        # Load batch_stats if available
        if "batch_stats" in biosr_pretrain:
            loaded_bs = _normalize_pretrain_tree_for_data_uq(biosr_pretrain["batch_stats"], "batch_stats")
            state = state.replace(batch_stats=loaded_bs)

        # ensure FrozenDict structure before initializing masked opt_state
        state = state.replace(params=freeze(unfreeze(state.params)))
        if state.batch_stats is not None:
            state = state.replace(batch_stats=freeze(unfreeze(state.batch_stats)))

        # Reset optimizer state to match (possibly updated) params tree
        state = state.replace(opt_state=state.tx.init(state.params))
        print("[pretrain] optimizer state reset")

        print("\033[92mSuccessfully loaded pretrained model\033[0m")

    state_rep = jax_utils.replicate(state)

    # pmap train step
    tv_w = float(args.tv_loss)
    lp_w = float(args.lp_tv)
    psfc_w = float(args.psfc_loss)
    psfg_w = float(args.psfg_loss)
    psf_sigma = float(args.psf_sigma)

    def train_step_pmap(state: TrainState, x: jnp.ndarray, gt_em: jnp.ndarray):
        rng = jax.random.fold_in(state.rng, state.step)
        rng = jax.random.fold_in(rng, jax.lax.axis_index("devices"))
        rng, k_do, k_dp, k_rm, new_rng = jax.random.split(rng, 5)

        def loss_fn(params, train: bool = True):
            variables = {"params": params}
            if state.batch_stats is not None:
                variables["batch_stats"] = state.batch_stats

            outs, new_state = state.apply_fn(
                variables,
                x,
                args,
                train,
                rngs={"dropout": k_do, "drop_path": k_dp, "random_masking": k_rm},
                mutable=["batch_stats"] if state.batch_stats is not None else [],
            )

            # 你现在的输出 tuple：
            (
                rec,
                light_pattern,
                emitter,
                psf,
                mask,
                deconv,
                emitter_logvar,
                lp_logvar,
                psf_logvar,
            ) = outs

            # ---- 统一成 (B,1,...) 5D，解决 4D/5D 混用 + (1,B,...) 问题 ----
            x5 = _to_b1_5d(x)
            rec = _to_b1_5d(rec)
            light_pattern = _to_b1_5d(light_pattern)
            emitter = _to_b1_5d(emitter)
            psf = _to_b1_5d(psf)
            mask = _to_b1_5d(mask)
            deconv = _to_b1_5d(deconv)
            emitter_logvar = _to_b1_5d(emitter_logvar)
            lp_logvar = _to_b1_5d(lp_logvar)
            psf_logvar = _to_b1_5d(psf_logvar)

            new_batch_stats = new_state["batch_stats"] if (state.batch_stats is not None) else None
            if bool(getattr(args, "freeze_base", False)):
                new_batch_stats = state.batch_stats

            # ---- 1) loss modes ----
            loss_mode = str(getattr(args, "loss_mode", "selfsup_uq"))
            use_plain = loss_mode == "plain"
            use_selfsup_uq = loss_mode == "selfsup_uq"
            use_emitter_uq = loss_mode == "emitter_uq"

            use_mask = train and getattr(args, "mask_ratio", 0.0) > 0.3 and (mask is not None)

            # ---- 2) 正则项：TV / center ----
            psf_tv = TV_Loss(psf)
            lp_tv = TV_Loss(light_pattern)
            psf_center = center_loss(psf)

            # 仅当 metric
            deconv_tv = TV_Loss(deconv) if (deconv is not None) else jnp.zeros((), dtype=rec.dtype)

            # base metrics from recon
            diff = rec - x5
            rec_mse = jnp.mean(diff * diff)
            rec_mae = jnp.mean(jnp.abs(diff))
            psnr = psnr_from_mse(rec_mse, max_val=1.0)

            # init
            rec_nll = jnp.nan
            rec_aux = jnp.zeros((), dtype=rec.dtype)
            logvar_reg = jnp.zeros((), dtype=rec.dtype)
            emitter_nll = jnp.nan
            var_emitter = None
            var_lp = None
            var_psf = None
            var_rec = None

            if use_plain:
                # ---- plain self-supervised loss (ref: model.py compute_metrics) ----
                if use_mask:
                    rec_plain = rec_loss(x5, rec, mask)
                else:
                    rec_plain = rec_loss(x5, rec)
                rec_nll = rec_plain
                loss = rec_plain + args.tv_loss * psf_tv + args.psfc_loss * psf_center + args.lp_tv * lp_tv
            elif use_selfsup_uq:
                # ---- self-supervised UQ (heteroscedastic NLL) ----
                lv_min = float(args.uq_logvar_min)
                lv_max = float(args.uq_logvar_max)

                if bool(getattr(args, "uq_emitter", True)):
                    emitter_logvar = clip_logvar(emitter_logvar, lv_min, lv_max)
                    var_emitter = logvar_to_var(emitter_logvar, float(args.uq_var_eps))
                else:
                    emitter_logvar = None
                    var_emitter = jnp.zeros_like(emitter)

                if bool(getattr(args, "uq_lp", True)):
                    lp_logvar = clip_logvar(lp_logvar, lv_min, lv_max)
                    var_lp = logvar_to_var(lp_logvar, float(args.uq_var_eps))
                else:
                    lp_logvar = None
                    var_lp = jnp.zeros_like(light_pattern)

                if bool(getattr(args, "uq_psf", True)):
                    psf_logvar = clip_logvar(psf_logvar, lv_min, lv_max)
                    var_psf = logvar_to_var(psf_logvar, float(args.uq_var_eps))
                else:
                    psf_logvar = None
                    var_psf = jnp.zeros_like(psf)

                # S = emitter * light_pattern
                mu_s = deconv if deconv is not None else (emitter * light_pattern)
                var_s = product_variance(emitter, var_emitter, light_pattern, var_lp)

                # rec variance: propagate through convolution + downsample
                psf_var_weight = float(args.uq_psf_var_weight) if bool(getattr(args, "uq_psf", True)) else 0.0
                var_rec_hr = propagate_rec_variance(
                    mu_s,
                    var_s,
                    psf,
                    var_psf,
                    psf_var_weight=psf_var_weight,
                )
                var_rec = downsample_variance(var_rec_hr, args.rescale)
                var_rec = jnp.maximum(var_rec, float(args.uq_var_floor))

                if use_mask:
                    rec_nll = gaussian_nll_from_var(diff, var_rec, mask=mask)
                else:
                    rec_nll = gaussian_nll_from_var(diff, var_rec, mask=None)
                rec_loss_w = float(args.rec_loss_weight)

                if float(args.uq_aux_rec_weight) > 0 and rec_loss_w > 0:
                    rec_aux = rec_loss(x5, rec, mask if use_mask else None)

                # logvar L2 regularization (optional)
                if float(args.uq_var_reg) > 0:
                    terms = []
                    if emitter_logvar is not None:
                        terms.append(jnp.mean(emitter_logvar * emitter_logvar))
                    if lp_logvar is not None:
                        terms.append(jnp.mean(lp_logvar * lp_logvar))
                    if psf_logvar is not None:
                        terms.append(jnp.mean(psf_logvar * psf_logvar))
                    if terms:
                        logvar_reg = sum(terms)

                # optional emitter supervised NLL
                if float(args.emitter_sup_weight) > 0:
                    gt_em_t = _to_b1_5d(gt_em)
                    gt_em_t = _downsample_hw_like(gt_em_t, emitter)
                    gt_em_t = _align_like(emitter, gt_em_t)
                    emitter_diff = emitter - gt_em_t
                    emitter_nll = gaussian_nll_from_var(emitter_diff, var_emitter, mask=None)

                loss = (
                    rec_loss_w * rec_nll
                    + rec_loss_w * float(args.uq_aux_rec_weight) * rec_aux
                    + args.tv_loss * psf_tv
                    + args.psfc_loss * psf_center
                    + args.lp_tv * lp_tv
                    + float(args.uq_var_reg) * logvar_reg
                    + float(args.emitter_sup_weight) * emitter_nll
                )
            else:
                # ---- supervised emitter UQ ----
                lv_min = float(args.uq_logvar_min)
                lv_max = float(args.uq_logvar_max)
                emitter_logvar = clip_logvar(emitter_logvar, lv_min, lv_max)
                var_emitter = logvar_to_var(emitter_logvar, float(args.uq_var_eps))

                if float(args.uq_var_reg) > 0:
                    logvar_reg = jnp.mean(emitter_logvar * emitter_logvar)

                gt_em_t = _to_b1_5d(gt_em)
                gt_em_t = _downsample_hw_like(gt_em_t, emitter)
                gt_em_t = _align_like(emitter, gt_em_t)
                emitter_diff = emitter - gt_em_t
                emitter_nll = gaussian_nll_from_var(emitter_diff, var_emitter, mask=None)

                loss = (
                    float(args.emitter_sup_weight) * emitter_nll
                    + args.tv_loss * psf_tv
                    + args.psfc_loss * psf_center
                    + args.lp_tv * lp_tv
                    + float(args.uq_var_reg) * logvar_reg
                )

            # ---- 3) 监控指标 ----
            rec_std_mean = jnp.nan
            emitter_std_mean = jnp.nan
            lp_std_mean = jnp.nan
            psf_std_mean = jnp.nan
            if use_selfsup_uq:
                rec_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_rec, 0.0)))
                emitter_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_emitter, 0.0)))
                lp_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_lp, 0.0)))
                psf_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_psf, 0.0)))
            elif use_emitter_uq and (var_emitter is not None):
                emitter_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_emitter, 0.0)))

            emitter_mse = jnp.nan
            emitter_mae = jnp.nan
            emitter_nll_metric = jnp.nan
            train_has_gt = bool(args.use_gt_metrics) or float(args.emitter_sup_weight) > 0 or use_emitter_uq
            if train_has_gt:
                gt_em_t = _to_b1_5d(gt_em)
                gt_em_t = _downsample_hw_like(gt_em_t, emitter)
                gt_em_t = _align_like(emitter, gt_em_t)
                emitter_diff = emitter - gt_em_t
                emitter_mse = jnp.mean(emitter_diff * emitter_diff)
                emitter_mae = jnp.mean(jnp.abs(emitter_diff))
                if (var_emitter is not None) and bool(getattr(args, "uq_emitter", False)):
                    emitter_nll_metric = gaussian_nll_from_var(emitter_diff, var_emitter, mask=None)

            metrics = {
                "loss": loss,
                "rec_nll": rec_nll,
                "rec_mse": rec_mse,
                "rec_mae": rec_mae,
                "psnr": psnr,
                "psf_tv": psf_tv,
                "lp_tv": lp_tv,
                "psf_center": psf_center,
                "rec_aux": rec_aux,
                "logvar_reg": logvar_reg,
                "deconv_tv": deconv_tv,
                "rec_std_mean": rec_std_mean,
                "emitter_std_mean": emitter_std_mean,
                "lp_std_mean": lp_std_mean,
                "psf_std_mean": psf_std_mean,
                "emitter_nll": emitter_nll_metric,
                "emitter_mse": emitter_mse,
                "emitter_mae": emitter_mae,
            }

            return loss, (metrics, new_batch_stats)

        (loss, (metrics, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        grads = jax.lax.pmean(grads, axis_name="devices")
        metrics = jax.lax.pmean(metrics, axis_name="devices")

        if new_batch_stats is not None:
            new_batch_stats = _pmean_tree(new_batch_stats)

        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(
            batch_stats=new_batch_stats if new_batch_stats is not None else state.batch_stats,
            rng=new_rng,
        )
        return new_state, metrics

    def eval_step_pmap(state: TrainState, x: jnp.ndarray, gt_em: jnp.ndarray, gt_lp: jnp.ndarray):
        rng = jax.random.fold_in(state.rng, state.step)
        rng = jax.random.fold_in(rng, jax.lax.axis_index("devices"))
        rng, k_do, k_dp, k_rm = jax.random.split(rng, 4)

        def forward(params):
            variables = {"params": params}
            if state.batch_stats is not None:
                variables["batch_stats"] = state.batch_stats

            outs, _ = state.apply_fn(
                variables,
                x,
                args,
                False,
                rngs={"dropout": k_do, "drop_path": k_dp, "random_masking": k_rm},
                mutable=["batch_stats"] if state.batch_stats is not None else [],
            )

            (
                rec,
                light_pattern,
                emitter,
                psf,
                mask,
                deconv,
                emitter_logvar,
                lp_logvar,
                psf_logvar,
            ) = outs

            x5 = _to_b1_5d(x)
            rec = _to_b1_5d(rec)
            light_pattern = _to_b1_5d(light_pattern)
            emitter = _to_b1_5d(emitter)
            psf = _to_b1_5d(psf)
            mask = _to_b1_5d(mask)
            deconv = _to_b1_5d(deconv)
            emitter_logvar = _to_b1_5d(emitter_logvar)
            lp_logvar = _to_b1_5d(lp_logvar)
            psf_logvar = _to_b1_5d(psf_logvar)

            loss_mode = str(getattr(args, "loss_mode", "selfsup_uq"))
            use_plain = loss_mode == "plain"
            use_selfsup_uq = loss_mode == "selfsup_uq"
            use_emitter_uq = loss_mode == "emitter_uq"

            # ---- regularizers ----
            psf_tv = TV_Loss(psf)
            lp_tv = TV_Loss(light_pattern)
            psf_center = center_loss(psf)

            # base metrics from recon
            diff = rec - x5
            rec_mse = jnp.mean(diff * diff)
            rec_mae = jnp.mean(jnp.abs(diff))
            psnr = psnr_from_mse(rec_mse, max_val=1.0)

            # init
            rec_nll = jnp.nan
            rec_aux = jnp.zeros((), dtype=rec.dtype)
            logvar_reg = jnp.zeros((), dtype=rec.dtype)
            emitter_nll = jnp.nan
            var_emitter = None
            var_lp = None
            var_psf = None
            var_rec = None

            if use_plain:
                rec_plain = rec_loss(x5, rec)
                rec_nll = rec_plain
                loss = rec_plain + args.tv_loss * psf_tv + args.psfc_loss * psf_center + args.lp_tv * lp_tv
            elif use_selfsup_uq:
                lv_min = float(args.uq_logvar_min)
                lv_max = float(args.uq_logvar_max)
                if bool(getattr(args, "uq_emitter", True)):
                    emitter_logvar = clip_logvar(emitter_logvar, lv_min, lv_max)
                    var_emitter = logvar_to_var(emitter_logvar, float(args.uq_var_eps))
                else:
                    emitter_logvar = None
                    var_emitter = jnp.zeros_like(emitter)

                if bool(getattr(args, "uq_lp", True)):
                    lp_logvar = clip_logvar(lp_logvar, lv_min, lv_max)
                    var_lp = logvar_to_var(lp_logvar, float(args.uq_var_eps))
                else:
                    lp_logvar = None
                    var_lp = jnp.zeros_like(light_pattern)

                if bool(getattr(args, "uq_psf", True)):
                    psf_logvar = clip_logvar(psf_logvar, lv_min, lv_max)
                    var_psf = logvar_to_var(psf_logvar, float(args.uq_var_eps))
                else:
                    psf_logvar = None
                    var_psf = jnp.zeros_like(psf)

                mu_s = deconv if deconv is not None else (emitter * light_pattern)
                var_s = product_variance(emitter, var_emitter, light_pattern, var_lp)

                psf_var_weight = float(args.uq_psf_var_weight) if bool(getattr(args, "uq_psf", True)) else 0.0
                var_rec_hr = propagate_rec_variance(
                    mu_s,
                    var_s,
                    psf,
                    var_psf,
                    psf_var_weight=psf_var_weight,
                )
                var_rec = downsample_variance(var_rec_hr, args.rescale)
                var_rec = jnp.maximum(var_rec, float(args.uq_var_floor))

                rec_nll = gaussian_nll_from_var(diff, var_rec, mask=None)
                rec_loss_w = float(args.rec_loss_weight)

                if float(args.uq_aux_rec_weight) > 0 and rec_loss_w > 0:
                    rec_aux = rec_loss(x5, rec)

                if float(args.uq_var_reg) > 0:
                    terms = []
                    if emitter_logvar is not None:
                        terms.append(jnp.mean(emitter_logvar * emitter_logvar))
                    if lp_logvar is not None:
                        terms.append(jnp.mean(lp_logvar * lp_logvar))
                    if psf_logvar is not None:
                        terms.append(jnp.mean(psf_logvar * psf_logvar))
                    if terms:
                        logvar_reg = sum(terms)

                loss = (
                    rec_loss_w * rec_nll
                    + rec_loss_w * float(args.uq_aux_rec_weight) * rec_aux
                    + args.tv_loss * psf_tv
                    + args.psfc_loss * psf_center
                    + args.lp_tv * lp_tv
                    + float(args.uq_var_reg) * logvar_reg
                )
            else:
                lv_min = float(args.uq_logvar_min)
                lv_max = float(args.uq_logvar_max)
                emitter_logvar = clip_logvar(emitter_logvar, lv_min, lv_max)
                var_emitter = logvar_to_var(emitter_logvar, float(args.uq_var_eps))
                if float(args.uq_var_reg) > 0:
                    logvar_reg = jnp.mean(emitter_logvar * emitter_logvar)
                gt_em_t = _to_b1_5d(gt_em)
                gt_em_t = _downsample_hw_like(gt_em_t, emitter)
                gt_em_t = _align_like(emitter, gt_em_t)
                emitter_nll = gaussian_nll_from_var(emitter - gt_em_t, var_emitter, mask=None)
                loss = (
                    float(args.emitter_sup_weight) * emitter_nll
                    + args.tv_loss * psf_tv
                    + args.psfc_loss * psf_center
                    + args.lp_tv * lp_tv
                    + float(args.uq_var_reg) * logvar_reg
                )

            rec_std_mean = jnp.nan
            emitter_std_mean = jnp.nan
            lp_std_mean = jnp.nan
            psf_std_mean = jnp.nan
            if use_selfsup_uq:
                rec_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_rec, 0.0)))
                emitter_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_emitter, 0.0)))
                lp_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_lp, 0.0)))
                psf_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_psf, 0.0)))
            elif use_emitter_uq and (var_emitter is not None):
                emitter_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_emitter, 0.0)))

            emitter_mse = jnp.nan
            emitter_mae = jnp.nan
            lp_mse = jnp.nan
            lp_mae = jnp.nan
            emitter_nll_metric = jnp.nan
            if val_use_gt:
                gt_em_t = _to_b1_5d(gt_em)
                gt_lp_t = _to_b1_5d(gt_lp)
                gt_em_t = _downsample_hw_like(gt_em_t, emitter)
                gt_em_t = _align_like(emitter, gt_em_t)
                gt_lp_t = _downsample_hw_like(gt_lp_t, light_pattern)
                gt_lp_t = _align_like(light_pattern, gt_lp_t)
                emitter_mse = jnp.mean((emitter - gt_em_t) ** 2)
                emitter_mae = jnp.mean(jnp.abs(emitter - gt_em_t))
                lp_mse = jnp.mean((light_pattern - gt_lp_t) ** 2)
                lp_mae = jnp.mean(jnp.abs(light_pattern - gt_lp_t))
                if (var_emitter is not None) and bool(getattr(args, "uq_emitter", False)):
                    emitter_nll_metric = gaussian_nll_from_var(emitter - gt_em_t, var_emitter, mask=None)

            metrics = {
                "loss": loss,
                "rec_nll": rec_nll,
                "rec_mse": rec_mse,
                "rec_mae": rec_mae,
                "psnr": psnr,
                "psf_tv": psf_tv,
                "lp_tv": lp_tv,
                "psf_center": psf_center,
                "rec_aux": rec_aux,
                "logvar_reg": logvar_reg,
                "rec_std_mean": rec_std_mean,
                "emitter_std_mean": emitter_std_mean,
                "lp_std_mean": lp_std_mean,
                "psf_std_mean": psf_std_mean,
                "emitter_mse": emitter_mse,
                "emitter_mae": emitter_mae,
                "lp_mse": lp_mse,
                "lp_mae": lp_mae,
                "emitter_nll": emitter_nll_metric,
            }
            return metrics

        metrics = forward(state.params)
        metrics = jax.lax.pmean(metrics, axis_name="devices")
        return metrics

    P_TRAIN_STEP = jax.pmap(train_step_pmap, axis_name="devices", donate_argnums=(0,))
    P_EVAL_STEP = jax.pmap(eval_step_pmap, axis_name="devices")

    # logs
    metrics_csv = os.path.join(args.ckpt_dir, "metrics.csv")
    csv_header = [
        "time",
        "epoch",
        "step",
        "loss",
        "rec_nll",
        "rec_mse",
        "rec_mae",
        "psnr",
        "emitter_nll",
        "rec_aux",
        "logvar_reg",
        "psf_tv",
        "lp_tv",
        "psf_center",
        "rec_std_mean",
        "emitter_std_mean",
        "lp_std_mean",
        "psf_std_mean",
        "emitter_mse",
        "emitter_mae",
        "lp_mse",
        "lp_mae",
        "note",
    ]
    if not os.path.exists(metrics_csv):
        csv_append(
            metrics_csv,
            {
                "time": time.time(),
                "epoch": 0,
                "step": int(jax.device_get(state.step)),
                "loss": "",
                "rec_nll": "",
                "rec_mse": "",
                "rec_mae": "",
                "psnr": "",
                "emitter_nll": "",
                "rec_aux": "",
                "logvar_reg": "",
                "psf_tv": "",
                "lp_tv": "",
                "psf_center": "",
                "rec_std_mean": "",
                "emitter_std_mean": "",
                "lp_std_mean": "",
                "psf_std_mean": "",
                "emitter_mse": "",
                "emitter_mae": "",
                "lp_mse": "",
                "lp_mae": "",
                "note": f"init n_devices={n_devices}",
            },
            header=csv_header,
        )

    # train loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        interval_buf: List[Dict[str, float]] = []
        epoch_buf: List[Dict[str, float]] = []

        for it, batch in enumerate(train_loader, start=1):
            if isinstance(batch, (list, tuple)):
                torch_x = batch[0]
                torch_em = batch[1] if (len(batch) > 1) else None
            else:
                torch_x = batch
                torch_em = None
            dl_np = _to_numpy(torch_x).astype(np.float32)  # (B,...) maybe include extra dims
            np_bchw = ensure_bchw_9(dl_np).astype(np.float32)  # (B,9,H,W)

            x = jnp.asarray(np_bchw)[:, None, ...]  # (B,1,9,H,W)
            x = shard_batch(x, n_devices)  # (n_devices, per_dev, 1,9,H,W)
            if torch_em is not None:
                gt_em_np = _to_numpy(torch_em).astype(np.float32)
                if gt_em_np.ndim == 3:
                    gt_em_np = gt_em_np[:, None, ...]
            else:
                gt_em_np = np.zeros((np_bchw.shape[0], 1, 1, 1), dtype=np.float32)
            gt_em = jnp.asarray(gt_em_np)
            gt_em = shard_batch(gt_em, n_devices)

            state_rep, metrics = P_TRAIN_STEP(state_rep, x, gt_em)

            m = {k: float(jax.device_get(v)[0]) for k, v in metrics.items()}
            interval_buf.append(m)
            epoch_buf.append(m)

            step = int(jax.device_get(state_rep.step)[0])

            if (it % args.log_every) == 0:
                avg = {k: float(np.mean([d[k] for d in interval_buf])) for k in interval_buf[0].keys()}
                mode = str(getattr(args, "loss_mode", "selfsup_uq"))
                if mode == "plain":
                    main_name = "rec_loss"
                    main_val = avg["rec_nll"]
                elif mode == "emitter_uq":
                    main_name = "emitter_nll"
                    main_val = avg.get("emitter_nll", float("nan"))
                else:
                    main_name = "nll"
                    main_val = avg["rec_nll"]
                print(
                    f"[epoch {epoch:03d}][iter {it:05d}] "
                    f"loss={avg['loss']:.6f} {main_name}={main_val:.6f} rec_mse={avg['rec_mse']:.6f} psnr={avg['psnr']:.2f} "
                    f"psf_tv={avg['psf_tv']:.6f} lp_tv={avg['lp_tv']:.6f} "
                    f"psf_center={avg['psf_center']:.6f} "
                    f"(step={step})"
                )
                csv_append(
                    metrics_csv,
                    {
                        "time": time.time(),
                        "epoch": epoch,
                        "step": step,
                        "loss": avg["loss"],
                        "rec_nll": avg["rec_nll"],
                        "rec_mse": avg["rec_mse"],
                        "rec_mae": avg["rec_mae"],
                        "psnr": avg["psnr"],
                        "emitter_nll": avg.get("emitter_nll", ""),
                        "rec_aux": avg["rec_aux"],
                        "logvar_reg": avg["logvar_reg"],
                        "psf_tv": avg["psf_tv"],
                        "lp_tv": avg["lp_tv"],
                        "psf_center": avg["psf_center"],
                        "rec_std_mean": avg["rec_std_mean"],
                        "emitter_std_mean": avg["emitter_std_mean"],
                        "lp_std_mean": avg["lp_std_mean"],
                        "psf_std_mean": avg["psf_std_mean"],
                        "emitter_mse": avg.get("emitter_mse", ""),
                        "emitter_mae": avg.get("emitter_mae", ""),
                        "lp_mse": avg.get("lp_mse", ""),
                        "lp_mae": avg.get("lp_mae", ""),
                        "note": "train_interval",
                    },
                    header=csv_header,
                )
                interval_buf = []

        epoch_avg = {k: float(np.mean([d[k] for d in epoch_buf])) for k in epoch_buf[0].keys()}

        # ---- optional val ----
        if val_loader is not None:
            val_sums = None
            val_count = 0
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    torch_x = batch[0]
                    torch_em = batch[1] if (len(batch) > 1) else None
                    torch_lp = batch[2] if (len(batch) > 2) else None
                else:
                    torch_x = batch
                    torch_em = None
                    torch_lp = None
                dl_np = _to_numpy(torch_x).astype(np.float32)
                np_bchw = ensure_bchw_9(dl_np).astype(np.float32)
                if val_use_gt and (torch_em is not None) and (torch_lp is not None):
                    gt_em_np = _to_numpy(torch_em).astype(np.float32)
                    gt_lp_np = _to_numpy(torch_lp).astype(np.float32)
                else:
                    gt_em_np = None
                    gt_lp_np = None
                actual = int(np_bchw.shape[0])
                if actual == 0:
                    continue
                # pad to be divisible by n_devices
                rem = actual % n_devices
                if rem != 0:
                    pad = n_devices - rem
                    pad_block = np.repeat(np_bchw[:1], pad, axis=0)
                    np_bchw = np.concatenate([np_bchw, pad_block], axis=0)
                    if gt_em_np is not None:
                        gt_em_np = np.concatenate([gt_em_np, np.repeat(gt_em_np[:1], pad, axis=0)], axis=0)
                    if gt_lp_np is not None:
                        gt_lp_np = np.concatenate([gt_lp_np, np.repeat(gt_lp_np[:1], pad, axis=0)], axis=0)
                x = jnp.asarray(np_bchw)[:, None, ...]
                x = shard_batch(x, n_devices)
                if gt_em_np is None:
                    gt_em_np = np.zeros((np_bchw.shape[0], 1, 1, 1), dtype=np.float32)
                if gt_lp_np is None:
                    gt_lp_np = np.zeros((np_bchw.shape[0], 1, 1, 1), dtype=np.float32)
                gt_em = jnp.asarray(gt_em_np)
                gt_lp = jnp.asarray(gt_lp_np)
                gt_em = shard_batch(gt_em, n_devices)
                gt_lp = shard_batch(gt_lp, n_devices)
                metrics = P_EVAL_STEP(state_rep, x, gt_em, gt_lp)
                m = {k: float(jax.device_get(v)[0]) for k, v in metrics.items()}
                if val_sums is None:
                    val_sums = {k: 0.0 for k in m.keys()}
                for k in val_sums:
                    val_sums[k] += m[k] * actual
                val_count += actual
            if val_sums is not None and val_count > 0:
                val_avg = {k: (val_sums[k] / float(val_count)) for k in val_sums}
                mode = str(getattr(args, "loss_mode", "selfsup_uq"))
                if mode == "plain":
                    main_name = "rec_loss"
                    main_val = val_avg["rec_nll"]
                elif mode == "emitter_uq":
                    main_name = "emitter_nll"
                    main_val = val_avg.get("emitter_nll", float("nan"))
                else:
                    main_name = "nll"
                    main_val = val_avg["rec_nll"]
                print(
                    f"[val {epoch:03d}] "
                    f"loss={val_avg['loss']:.6f} {main_name}={main_val:.6f} rec_mse={val_avg['rec_mse']:.6f} "
                    f"psnr={val_avg['psnr']:.2f}"
                )
                csv_append(
                    metrics_csv,
                    {
                        "time": time.time(),
                        "epoch": epoch,
                        "step": int(jax.device_get(state_rep.step)[0]),
                        "loss": val_avg["loss"],
                        "rec_nll": val_avg["rec_nll"],
                        "rec_mse": val_avg["rec_mse"],
                        "rec_mae": val_avg["rec_mae"],
                        "psnr": val_avg["psnr"],
                        "emitter_nll": val_avg.get("emitter_nll", ""),
                        "rec_aux": val_avg["rec_aux"],
                        "logvar_reg": val_avg["logvar_reg"],
                        "psf_tv": val_avg["psf_tv"],
                        "lp_tv": val_avg["lp_tv"],
                        "psf_center": val_avg["psf_center"],
                        "rec_std_mean": val_avg["rec_std_mean"],
                        "emitter_std_mean": val_avg["emitter_std_mean"],
                        "lp_std_mean": val_avg["lp_std_mean"],
                        "psf_std_mean": val_avg["psf_std_mean"],
                        "emitter_mse": val_avg.get("emitter_mse", ""),
                        "emitter_mae": val_avg.get("emitter_mae", ""),
                        "lp_mse": val_avg.get("lp_mse", ""),
                        "lp_mae": val_avg.get("lp_mae", ""),
                        "note": "val_epoch",
                    },
                    header=csv_header,
                )

        # save ckpt
        to_save = jax_utils.unreplicate(state_rep)
        step = int(jax.device_get(to_save.step))
        checkpoints.save_checkpoint(
            ckpt_dir=args.ckpt_dir,
            target=to_save,
            step=step,
            keep=args.keep_ckpts,
            overwrite=True,
        )

        dt = time.time() - t0
        mode = str(getattr(args, "loss_mode", "selfsup_uq"))
        if mode == "plain":
            main_name = "rec_loss"
            main_val = epoch_avg["rec_nll"]
        elif mode == "emitter_uq":
            main_name = "emitter_nll"
            main_val = epoch_avg.get("emitter_nll", float("nan"))
        else:
            main_name = "nll"
            main_val = epoch_avg["rec_nll"]
        print(
            f"[epoch {epoch:03d}] "
            f"loss={epoch_avg['loss']:.6f} {main_name}={main_val:.6f} rec_mse={epoch_avg['rec_mse']:.6f} psnr={epoch_avg['psnr']:.2f} "
            f"psf_tv={epoch_avg['psf_tv']:.6f} lp_tv={epoch_avg['lp_tv']:.6f} "
            f"psf_center={epoch_avg['psf_center']:.6f} "
            f"| time={dt:.1f}s step={step}"
        )
        csv_append(
            metrics_csv,
            {
                "time": time.time(),
                "epoch": epoch,
                "step": step,
                "loss": epoch_avg["loss"],
                "rec_nll": epoch_avg["rec_nll"],
                "rec_mse": epoch_avg["rec_mse"],
                "rec_mae": epoch_avg["rec_mae"],
                "psnr": epoch_avg["psnr"],
                "emitter_nll": epoch_avg.get("emitter_nll", ""),
                "rec_aux": epoch_avg["rec_aux"],
                "logvar_reg": epoch_avg["logvar_reg"],
                "psf_tv": epoch_avg["psf_tv"],
                "lp_tv": epoch_avg["lp_tv"],
                "psf_center": epoch_avg["psf_center"],
                "rec_std_mean": epoch_avg["rec_std_mean"],
                "emitter_std_mean": epoch_avg["emitter_std_mean"],
                "lp_std_mean": epoch_avg["lp_std_mean"],
                "psf_std_mean": epoch_avg["psf_std_mean"],
                "emitter_mse": epoch_avg.get("emitter_mse", ""),
                "emitter_mae": epoch_avg.get("emitter_mae", ""),
                "lp_mse": epoch_avg.get("lp_mse", ""),
                "lp_mae": epoch_avg.get("lp_mae", ""),
                "note": "train_epoch",
            },
            header=csv_header,
        )
        _plot_metric_curves(metrics_csv, os.path.join(args.ckpt_dir, "curves"))

        # ---------------- optional debug saving ----------------
        if debug_loader is not None and args.debug_every_epochs > 0 and (epoch % args.debug_every_epochs == 0):
            out_dir = os.path.join(args.debug_dir, f"epoch_{epoch:03d}_step_{step:07d}")
            os.makedirs(out_dir, exist_ok=True)

            # 固定 debug seed（不影响训练）：只包住 debug 取样 + 记录 crop
            old_crop_fn = uq_data.random_crop_arrays
            uq_data.random_crop_arrays = _random_crop_arrays_record
            DEBUG_CROP_RECORDS.clear()

            rng_state = np.random.get_state()
            np.random.seed(args.debug_seed)
            debug_batch = next(iter(debug_loader))
            np.random.set_state(rng_state)

            uq_data.random_crop_arrays = old_crop_fn
            crop_record = DEBUG_CROP_RECORDS[0] if len(DEBUG_CROP_RECORDS) > 0 else None

            # batch -> numpy -> ensure -> add axis -> device0
            torch_dbg = debug_batch[0] if isinstance(debug_batch, (list, tuple)) else debug_batch
            dl_np = _to_numpy(torch_dbg).astype(np.float32)
            ensured = ensure_bchw_9(dl_np).astype(np.float32)  # (B,9,H,W)
            x_in = ensured[:, None, ...]  # (B,1,9,H,W)
            x_dbg = jax.device_put(jnp.asarray(x_in), device0)

            # single-device state（避免多卡参数 device mismatch）
            state_single = jax_utils.unreplicate(state_rep)
            state_single = jax.device_put(jax.device_get(state_single), device0)

            # forward
            (
                rec,
                lp,
                emitter,
                psf,
                mask,
                deconv,
                emitter_logvar,
                lp_logvar,
                psf_logvar,
            ) = debug_forward_single_device(
                state_single,
                x_dbg,
                args,
                debug_train=bool(getattr(args, "debug_train", True)),
            )

            # device_get
            x_np = np.array(jax.device_get(x_dbg)).astype(np.float32)
            rec_np = np.array(jax.device_get(rec)).astype(np.float32)
            lp_np = np.array(jax.device_get(lp)).astype(np.float32)
            emitter_np = np.array(jax.device_get(emitter)).astype(np.float32)
            psf_np = np.array(jax.device_get(psf)).astype(np.float32)

            deconv_np = None
            mask_np = None
            try:
                if deconv is not None:
                    deconv_np = np.array(jax.device_get(deconv)).astype(np.float32)
                if mask is not None:
                    mask_np = np.array(jax.device_get(mask)).astype(np.float32)
            except Exception:
                pass

            # -------- loss-mode-specific uncertainty (debug) --------
            loss_mode = str(getattr(args, "loss_mode", "selfsup_uq"))
            use_plain = loss_mode == "plain"
            use_selfsup_uq = loss_mode == "selfsup_uq"
            use_emitter_uq = loss_mode == "emitter_uq"
            use_emitter_uq_flag = (not use_plain) and bool(getattr(args, "uq_emitter", False))
            use_lp_uq_flag = use_selfsup_uq and bool(getattr(args, "uq_lp", False))
            use_psf_uq_flag = use_selfsup_uq and bool(getattr(args, "uq_psf", False))

            lv_min = float(args.uq_logvar_min)
            lv_max = float(args.uq_logvar_max)

            emitter_j = _to_b1_5d(emitter)
            lp_j = _to_b1_5d(lp)
            psf_j = _to_b1_5d(psf)
            deconv_j = _to_b1_5d(deconv) if deconv is not None else None
            x_j = _to_b1_5d(x_dbg)
            rec_j = _to_b1_5d(rec)

            var_emitter = None
            var_lp = None
            var_psf = None
            var_rec = None
            emitter_std_np = None
            rec_std_np = None

            if use_selfsup_uq:
                if use_emitter_uq_flag:
                    emitter_logvar = _to_b1_5d(emitter_logvar)
                    emitter_logvar_c = clip_logvar(emitter_logvar, lv_min, lv_max)
                    var_emitter = logvar_to_var(emitter_logvar_c, float(args.uq_var_eps))
                else:
                    var_emitter = jnp.zeros_like(emitter_j)

                if use_lp_uq_flag:
                    lp_logvar = _to_b1_5d(lp_logvar)
                    lp_logvar_c = clip_logvar(lp_logvar, lv_min, lv_max)
                    var_lp = logvar_to_var(lp_logvar_c, float(args.uq_var_eps))
                else:
                    var_lp = jnp.zeros_like(lp_j)

                if use_psf_uq_flag:
                    psf_logvar = _to_b1_5d(psf_logvar)
                    psf_logvar_c = clip_logvar(psf_logvar, lv_min, lv_max)
                    var_psf = logvar_to_var(psf_logvar_c, float(args.uq_var_eps))
                else:
                    var_psf = jnp.zeros_like(psf_j)

                mu_s = deconv_j if deconv_j is not None else (emitter_j * lp_j)
                var_s = product_variance(emitter_j, var_emitter, lp_j, var_lp)
                var_rec_hr = propagate_rec_variance(
                    mu_s,
                    var_s,
                    psf_j,
                    var_psf,
                    psf_var_weight=float(args.uq_psf_var_weight) if use_psf_uq_flag else 0.0,
                )
                var_rec = downsample_variance(var_rec_hr, args.rescale)
                var_rec = jnp.maximum(var_rec, float(args.uq_var_floor))
                rec_std = jnp.sqrt(jnp.maximum(var_rec, 0.0))
                rec_std_np = np.array(jax.device_get(rec_std)).astype(np.float32)
                if use_emitter_uq_flag:
                    emitter_std = jnp.sqrt(jnp.maximum(var_emitter, 0.0))
                    emitter_std_np = np.array(jax.device_get(emitter_std)).astype(np.float32)
            elif use_emitter_uq:
                emitter_logvar = _to_b1_5d(emitter_logvar)
                emitter_logvar_c = clip_logvar(emitter_logvar, lv_min, lv_max)
                var_emitter = logvar_to_var(emitter_logvar_c, float(args.uq_var_eps))
                emitter_std = jnp.sqrt(jnp.maximum(var_emitter, 0.0))
                emitter_std_np = np.array(jax.device_get(emitter_std)).astype(np.float32)

            gt_pack = None
            emitter_mse = None
            emitter_mae = None
            lp_mse = None
            lp_mae = None
            gt_em_t = None
            gt_lp_t = None
            if bool(args.use_gt_metrics):
                gt_em_path, gt_lp_path = derive_gt_paths(
                    [str(debug_path)],
                    args.gt_dir_token,
                    args.gt_dir_repl,
                    args.gt_emitter_suffix,
                    args.gt_lp_suffix,
                )[0]
                if os.path.exists(gt_em_path) and os.path.exists(gt_lp_path):
                    gt_em_np = imread(gt_em_path).astype(np.float32)
                    gt_lp_np = imread(gt_lp_path).astype(np.float32)
                    if gt_em_np.ndim == 2:
                        gt_em_np = gt_em_np[None, ...]
                    if gt_lp_np.ndim == 2:
                        gt_lp_np = gt_lp_np[None, ...]
                    gt_em_t = _to_b1_5d(jnp.asarray(gt_em_np))
                    gt_lp_t = _to_b1_5d(jnp.asarray(gt_lp_np))
                    gt_em_t = _downsample_hw_like(gt_em_t, emitter_j)
                    gt_em_t = _align_like(emitter_j, gt_em_t)
                    gt_lp_t = _downsample_hw_like(gt_lp_t, lp_j)
                    gt_lp_t = _align_like(lp_j, gt_lp_t)
                    gt_em_vis = np.array(jax.device_get(gt_em_t)).astype(np.float32)
                    gt_lp_vis = np.array(jax.device_get(gt_lp_t)).astype(np.float32)
                    gt_pack = {"emitter_gt": gt_em_vis, "lp_gt": gt_lp_vis}
                    emitter_mse = float(np.mean((emitter_np - gt_em_vis) ** 2))
                    emitter_mae = float(np.mean(np.abs(emitter_np - gt_em_vis)))
                    lp_mse = float(np.mean((lp_np - gt_lp_vis) ** 2))
                    lp_mae = float(np.mean(np.abs(lp_np - gt_lp_vis)))

            # visuals
            _save_debug_visuals(
                out_dir=out_dir,
                args=args,
                x_np=x_np,
                rec_np=rec_np,
                lp_np=lp_np,
                emitter_np=emitter_np,
                psf_np=psf_np,
                deconv_np=deconv_np,
                mask_np=mask_np,
                emitter_std_np=emitter_std_np,
                rec_std_np=rec_std_np,
                gt_pack=gt_pack,
            )

            # debug metrics
            dbg_metrics = {
                "rec_mse": float(jnp.mean((rec_j - x_j) ** 2)),
                "rec_mae": float(jnp.mean(jnp.abs(rec_j - x_j))),
                "psnr": float(psnr_from_mse(jnp.mean((rec_j - x_j) ** 2), max_val=1.0)),
                "psf_tv": float(TV_Loss(psf_j)),
                "lp_tv": float(TV_Loss(lp_j)),
                "psf_center": float(center_loss(psf_j)),
            }

            if use_plain:
                dbg_metrics["rec_nll"] = float(rec_loss(x_j, rec_j))
            elif use_selfsup_uq and (var_rec is not None):
                dbg_metrics["rec_nll"] = float(gaussian_nll_from_var(rec_j - x_j, var_rec, mask=None))
            else:
                dbg_metrics["rec_nll"] = float("nan")

            if use_selfsup_uq and float(args.uq_aux_rec_weight) > 0:
                dbg_metrics["rec_aux"] = float(rec_loss(x_j, rec_j))
            else:
                dbg_metrics["rec_aux"] = float("nan")

            if var_rec is not None:
                dbg_metrics["rec_std_mean"] = float(jnp.mean(jnp.sqrt(jnp.maximum(var_rec, 0.0))))
            if var_emitter is not None:
                dbg_metrics["emitter_std_mean"] = float(jnp.mean(jnp.sqrt(jnp.maximum(var_emitter, 0.0))))
            if var_lp is not None:
                dbg_metrics["lp_std_mean"] = float(jnp.mean(jnp.sqrt(jnp.maximum(var_lp, 0.0))))
            if var_psf is not None:
                dbg_metrics["psf_std_mean"] = float(jnp.mean(jnp.sqrt(jnp.maximum(var_psf, 0.0))))

            if gt_em_t is not None:
                dbg_metrics["emitter_mse"] = emitter_mse
                dbg_metrics["emitter_mae"] = emitter_mae
                if (var_emitter is not None) and use_emitter_uq_flag:
                    dbg_metrics["emitter_nll"] = float(
                        gaussian_nll_from_var(emitter_j - gt_em_t, var_emitter, mask=None)
                    )
            if gt_lp_t is not None:
                dbg_metrics["lp_mse"] = lp_mse
                dbg_metrics["lp_mae"] = lp_mae

            # ---- calibration curves ----
            try:
                if use_selfsup_uq and (var_rec is not None):
                    rec_var_np = np.array(jax.device_get(var_rec)).astype(np.float32)
                    rec_err2 = (rec_np - x_np) ** 2
                    x_cal, y_cal = _calibration_curve_np(rec_var_np, rec_err2, bins=10)
                    _plot_calibration_curve(
                        x_cal,
                        y_cal,
                        "Recon calibration",
                        os.path.join(out_dir, "calibration_recon.png"),
                    )
                if (gt_em_t is not None) and use_emitter_uq_flag and (var_emitter is not None):
                    em_var_np = np.array(jax.device_get(var_emitter)).astype(np.float32)
                    em_err2 = (emitter_np - np.array(jax.device_get(gt_em_t)).astype(np.float32)) ** 2
                    x_cal, y_cal = _calibration_curve_np(em_var_np, em_err2, bins=10)
                    _plot_calibration_curve(
                        x_cal,
                        y_cal,
                        "Emitter calibration",
                        os.path.join(out_dir, "calibration_emitter.png"),
                    )
            except Exception:
                pass

            note = f"debug_sample:{out_dir}"

            csv_append(
                metrics_csv,
                {
                    "time": time.time(),
                    "epoch": epoch,
                    "step": step,
                    "loss": "",
                    "rec_nll": dbg_metrics.get("rec_nll", ""),
                    "rec_mse": dbg_metrics["rec_mse"],
                    "rec_mae": dbg_metrics["rec_mae"],
                    "psnr": dbg_metrics["psnr"],
                    "emitter_nll": dbg_metrics.get("emitter_nll", ""),
                    "rec_aux": "",
                    "logvar_reg": "",
                    "psf_tv": dbg_metrics["psf_tv"],
                    "lp_tv": dbg_metrics["lp_tv"],
                    "psf_center": dbg_metrics["psf_center"],
                    "rec_std_mean": dbg_metrics.get("rec_std_mean", ""),
                    "emitter_std_mean": dbg_metrics.get("emitter_std_mean", ""),
                    "lp_std_mean": dbg_metrics.get("lp_std_mean", ""),
                    "psf_std_mean": dbg_metrics.get("psf_std_mean", ""),
                    "emitter_mse": dbg_metrics.get("emitter_mse", ""),
                    "emitter_mae": dbg_metrics.get("emitter_mae", ""),
                    "lp_mse": dbg_metrics.get("lp_mse", ""),
                    "lp_mae": dbg_metrics.get("lp_mae", ""),
                    "note": note,
                },
                header=csv_header,
            )
            print(f"[debug] saved to: {out_dir}")


if __name__ == "__main__":
    main()

# 例：
'''
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  uv run python uq_train_data_uq.py --train_glob "/data/repo/SIMFormer/data/SIM-simulation/*/*/standard/train/*.tif" \
  --batch_size 7 --epochs 500 --lr 1e-4 --ckpt_dir "./ckpts_uq3" --debug_every_epochs 5 \
  --uq_aux_rec_weight 0.1 --uq_psf_var_weight 1.0 \
  --uq_targets emitter
'''
