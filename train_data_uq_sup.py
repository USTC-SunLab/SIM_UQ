# train_data_uq_sup.py
# -*- coding: utf-8 -*-
"""
监督学习版的数据不确定性（heteroscedastic regression）训练脚本：
- 不再用 rec vs input 的自监督 NLL
- 直接用输出与标签的 NLL（emitter / light_pattern / psf 可选）
- 依然保留 PSF / LP 的正则项（TV / center 等）
"""

from __future__ import annotations

import pickle
import os
import csv
import glob
import time
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.io import imread

import jax
import jax.numpy as jnp
from jax import lax
import optax
from flax.training import train_state, checkpoints
from flax import jax_utils
from flax import linen as nn
from flax.core import freeze, unfreeze

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
from uq_vis_data_uq import save_debug_artifacts_data_uq_sup, save_calibration_curves_from_dir


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
    - 5D: 修正 (1,B,...) -> (B,1,...)；其余不动
    - 4D: (B,Z,Y,X) -> (B,1,Z,Y,X)
    - 3D: (Z,Y,X) -> (1,1,Z,Y,X)
    - 2D: (Y,X) -> (1,1,1,Y,X)
    """
    if t is None:
        return None
    if t.ndim == 5:
        return _norm_5d_leading_dims(t)
    if t.ndim == 4:
        return t[:, None, ...]
    if t.ndim == 3:
        return t[None, None, ...]
    if t.ndim == 2:
        return t[None, None, None, ...]
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


def _downsample_hw_like(gt: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    将 GT 在 H/W 上下采样到与 target 一致。
    期望输入为 5D (B,1,Z,H,W)，只处理 H/W 维度。
    优先用 avg_pool（整倍数），否则用 resize 兜底。
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

    # fallback: resize to match
    g = jnp.transpose(gt, (0, 2, 3, 4, 1))  # (B,Z,H,W,1)
    g = jax.image.resize(g, shape=(g.shape[0], g.shape[1], th, tw, g.shape[4]), method="linear")
    g = jnp.transpose(g, (0, 4, 1, 2, 3))
    return g


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

    # ---- supervised GT paths ----
    p.add_argument("--gt_emitter_glob", type=str, default="", help="glob for emitter GT (optional)")
    p.add_argument("--gt_lp_glob", type=str, default="", help="glob for LP GT (optional)")
    p.add_argument(
        "--gt_dir_token",
        type=str,
        default="/train/",
        help="token in train path to replace when deriving GT paths",
    )
    p.add_argument(
        "--gt_dir_repl",
        type=str,
        default="/train_gt/",
        help="replacement for token in train path",
    )
    p.add_argument("--gt_emitter_suffix", type=str, default="", help="suffix before ext for emitter GT")
    p.add_argument("--gt_lp_suffix", type=str, default="_lp", help="suffix before ext for LP GT")
    p.add_argument("--psf_gt_path", type=str, default="", help="path to PSF GT (required if supervising psf)")
    p.add_argument(
        "--gt_norm",
        type=str,
        default="minmax_full",
        choices=["none", "minmax", "minmax_full", "minmax_per_channel"],
        help="normalization for GT emitter/lp (recommended: minmax_full for sparse emitter)",
    )

    # ---- data uncertainty / heteroscedastic ----
    p.add_argument("--uq_logvar_min", type=float, default=-10.0, help="min clamp for log-variance")
    p.add_argument("--uq_logvar_max", type=float, default=3.0, help="max clamp for log-variance")
    p.add_argument("--uq_logvar_init", type=float, default=-4.0, help="init bias for logvar heads")
    p.add_argument("--uq_var_eps", type=float, default=1e-6, help="epsilon added to variance")
    p.add_argument("--uq_var_reg", type=float, default=0.0, help="L2 reg weight for logvar maps")
    p.add_argument(
        "--uq_targets",
        type=str,
        default="emitter,lp,psf",
        help="comma-separated subset of {emitter,lp,psf} to supervise + enable UQ (use 'all' or 'none')",
    )

    # ---- supervised loss weights ----
    p.add_argument("--sup_weight_emitter", type=float, default=1.0, help="weight for emitter NLL")
    p.add_argument("--sup_weight_lp", type=float, default=1.0, help="weight for LP NLL")
    p.add_argument("--sup_weight_psf", type=float, default=1.0, help="weight for PSF NLL")

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

    if not (args.uq_emitter or args.uq_lp or args.uq_psf):
        raise ValueError("--uq_targets resolved to empty; choose at least one of emitter/lp/psf")

    # abs paths + debug_dir default
    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not str(args.debug_dir).strip():
        args.debug_dir = os.path.join(args.ckpt_dir, "debug_samples")
    args.debug_dir = os.path.abspath(args.debug_dir)
    return args


# =============================================================================
# loss helpers
# =============================================================================

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


def csv_append(path: str, row: Dict[str, Any], header: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _corrcoef_flat(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    cov = jnp.mean(a * b)
    va = jnp.mean(a * a)
    vb = jnp.mean(b * b)
    return cov / (jnp.sqrt(va * vb) + float(eps))


def _build_gt_paths(train_paths: List[str], args) -> List[Tuple[str, str]]:
    if str(args.gt_emitter_glob).strip() and str(args.gt_lp_glob).strip():
        emitter_paths = sorted(glob.glob(args.gt_emitter_glob))
        lp_paths = sorted(glob.glob(args.gt_lp_glob))
        if len(emitter_paths) != len(train_paths) or len(lp_paths) != len(train_paths):
            raise ValueError(
                "GT globs length mismatch: "
                f"train={len(train_paths)} emitter={len(emitter_paths)} lp={len(lp_paths)}"
            )
        return list(zip(emitter_paths, lp_paths))

    token = str(args.gt_dir_token)
    repl = str(args.gt_dir_repl)
    gt_paths: List[Tuple[str, str]] = []
    for p in train_paths:
        base = p.replace(token, repl) if token else p
        stem, ext = os.path.splitext(base)
        emitter_path = f"{stem}{args.gt_emitter_suffix}{ext}"
        lp_path = f"{stem}{args.gt_lp_suffix}{ext}"
        gt_paths.append((emitter_path, lp_path))
    return gt_paths


def _validate_gt_paths(gt_paths: List[Tuple[str, str]]):
    missing = []
    for e, l in gt_paths:
        if not os.path.exists(e):
            missing.append(e)
        if not os.path.exists(l):
            missing.append(l)
    if missing:
        show = "\n".join(missing[:5])
        more = f"\n... (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(f"Missing GT files (showing up to 5):\n{show}{more}")


def _load_psf_gt(path: str) -> np.ndarray:
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npy", ".npz"):
        if ext == ".npz":
            with np.load(path) as data:
                if "psf" in data:
                    arr = data["psf"]
                else:
                    keys = list(data.keys())
                    if not keys:
                        raise ValueError(f"Empty npz: {path}")
                    arr = data[keys[0]]
        else:
            arr = np.load(path)
    else:
        arr = imread(path)
    return np.asarray(arr, dtype=np.float32)


def _guess_meta_path(train_path: str) -> str:
    base = os.path.basename(train_path)
    stem, _ = os.path.splitext(base)
    d = os.path.dirname(train_path)
    parent = os.path.basename(d)
    if parent == "train":
        meta_dir = os.path.join(os.path.dirname(d), "train_meta")
    else:
        meta_dir = os.path.join(d, "train_meta")
    return os.path.join(meta_dir, f"{stem}.npz")


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
        from network import PiMAE

        self.base = PiMAE(self.img_size, self.patch_size, self.psf_size, self.rank)
        self.emitter_logvar_head = LogVarHead(self.logvar_init, name="emitter_logvar")
        self.lp_logvar_head = LogVarHead(self.logvar_init, name="lp_logvar")
        self.psf_logvar_head = LogVarHead(self.logvar_init, name="psf_logvar")

    def __call__(self, x, args, training):
        rec, light_pattern, emitter, psf, mask, deconv = self.base(x, args, training)
        emitter_logvar = self.emitter_logvar_head(emitter)
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

def create_state(rng: jax.Array, model, args) -> TrainState:
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

    tx = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
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

    # supervised GT paths
    need_gt = bool(args.uq_emitter or args.uq_lp)
    gt_paths = None
    if need_gt:
        gt_paths = _build_gt_paths(train_paths, args)
        _validate_gt_paths(gt_paths)

    # PSF GT
    psf_gt = None
    if args.uq_psf:
        if not str(args.psf_gt_path).strip():
            raise ValueError("--psf_gt_path is required when supervising PSF")
        psf_gt = _load_psf_gt(args.psf_gt_path)
        psf_gt = jnp.asarray(psf_gt, dtype=jnp.float32)
        psf_gt = _to_b1_5d(psf_gt)

    # dataset/loader
    train_ds = dataset_2d_sim_supervised(
        train_paths,
        crop_size=(args.crop_h, args.crop_w),
        use_gt=need_gt,
        gt_paths=gt_paths,
        gt_norm=args.gt_norm,
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
            use_gt=need_gt,
            gt_paths=([gt_paths[args.debug_index]] if need_gt else None),
            gt_norm=args.gt_norm,
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
    state = create_state(rng, model, args)
    print("[init] success")

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

        # Reset optimizer state to match (possibly updated) params tree
        state = state.replace(opt_state=state.tx.init(state.params))
        print("[pretrain] optimizer state reset")

        print("\033[92mSuccessfully loaded pretrained model\033[0m")

    state_rep = jax_utils.replicate(state)

    # pmap train step
    use_emitter_uq = bool(getattr(args, "uq_emitter", True))
    use_lp_uq = bool(getattr(args, "uq_lp", True))
    use_psf_uq = bool(getattr(args, "uq_psf", True))

    sup_w_em = float(args.sup_weight_emitter)
    sup_w_lp = float(args.sup_weight_lp)
    sup_w_psf = float(args.sup_weight_psf)

    lv_min = float(args.uq_logvar_min)
    lv_max = float(args.uq_logvar_max)

    def train_step_pmap(state: TrainState, x: jnp.ndarray, emitter_gt: jnp.ndarray, lp_gt: jnp.ndarray):
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

            # outputs
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

            # ---- supervised NLL (emitter/lp/psf) ----
            loss = jnp.zeros((), dtype=rec.dtype)
            logvar_reg = jnp.zeros((), dtype=rec.dtype)

            emitter_nll = jnp.zeros((), dtype=rec.dtype)
            lp_nll = jnp.zeros((), dtype=rec.dtype)
            psf_nll = jnp.zeros((), dtype=rec.dtype)
            emitter_mse = jnp.zeros((), dtype=rec.dtype)
            lp_mse = jnp.zeros((), dtype=rec.dtype)
            psf_mse = jnp.zeros((), dtype=rec.dtype)
            emitter_mae = jnp.zeros((), dtype=rec.dtype)
            lp_mae = jnp.zeros((), dtype=rec.dtype)
            psf_mae = jnp.zeros((), dtype=rec.dtype)

            emitter_std_mean = jnp.zeros((), dtype=rec.dtype)
            lp_std_mean = jnp.zeros((), dtype=rec.dtype)
            psf_std_mean = jnp.zeros((), dtype=rec.dtype)
            emitter_var_mean = jnp.zeros((), dtype=rec.dtype)
            lp_var_mean = jnp.zeros((), dtype=rec.dtype)
            psf_var_mean = jnp.zeros((), dtype=rec.dtype)
            emitter_calib_ratio = jnp.zeros((), dtype=rec.dtype)
            lp_calib_ratio = jnp.zeros((), dtype=rec.dtype)
            emitter_corr = jnp.zeros((), dtype=rec.dtype)
            lp_corr = jnp.zeros((), dtype=rec.dtype)

            # emitter
            if use_emitter_uq:
                emitter_logvar = clip_logvar(emitter_logvar, lv_min, lv_max)
                var_emitter = logvar_to_var(emitter_logvar, float(args.uq_var_eps))

                gt_em = _to_b1_5d(emitter_gt)
                gt_em = _downsample_hw_like(gt_em, emitter)
                gt_em = _align_like(emitter, gt_em)

                diff = emitter - gt_em
                emitter_nll = gaussian_nll_from_var(diff, var_emitter)
                emitter_mse = jnp.mean(diff * diff)
                emitter_mae = jnp.mean(jnp.abs(diff))
                emitter_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_emitter, 0.0)))
                emitter_var_mean = jnp.mean(var_emitter)
                emitter_calib_ratio = emitter_mse / (emitter_var_mean + 1e-8)
                emitter_corr = _corrcoef_flat(var_emitter, diff * diff)
                loss = loss + sup_w_em * emitter_nll
                if float(args.uq_var_reg) > 0:
                    logvar_reg = logvar_reg + jnp.mean(emitter_logvar * emitter_logvar)

            # light pattern
            if use_lp_uq:
                lp_logvar = clip_logvar(lp_logvar, lv_min, lv_max)
                var_lp = logvar_to_var(lp_logvar, float(args.uq_var_eps))

                gt_lp = _to_b1_5d(lp_gt)
                gt_lp = _downsample_hw_like(gt_lp, light_pattern)
                gt_lp = _align_like(light_pattern, gt_lp)

                diff = light_pattern - gt_lp
                lp_nll = gaussian_nll_from_var(diff, var_lp)
                lp_mse = jnp.mean(diff * diff)
                lp_mae = jnp.mean(jnp.abs(diff))
                lp_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_lp, 0.0)))
                lp_var_mean = jnp.mean(var_lp)
                lp_calib_ratio = lp_mse / (lp_var_mean + 1e-8)
                lp_corr = _corrcoef_flat(var_lp, diff * diff)
                loss = loss + sup_w_lp * lp_nll
                if float(args.uq_var_reg) > 0:
                    logvar_reg = logvar_reg + jnp.mean(lp_logvar * lp_logvar)

            # psf
            if use_psf_uq:
                psf_logvar = clip_logvar(psf_logvar, lv_min, lv_max)
                var_psf = logvar_to_var(psf_logvar, float(args.uq_var_eps))

                if psf_gt is None:
                    raise ValueError("psf_gt missing while supervising psf")
                gt_psf = _to_b1_5d(psf_gt)
                gt_psf = _downsample_hw_like(gt_psf, psf)
                gt_psf = _align_like(psf, gt_psf)

                diff = psf - gt_psf
                psf_nll = gaussian_nll_from_var(diff, var_psf)
                psf_mse = jnp.mean(diff * diff)
                psf_mae = jnp.mean(jnp.abs(diff))
                psf_std_mean = jnp.mean(jnp.sqrt(jnp.maximum(var_psf, 0.0)))
                psf_var_mean = jnp.mean(var_psf)
                loss = loss + sup_w_psf * psf_nll
                if float(args.uq_var_reg) > 0:
                    logvar_reg = logvar_reg + jnp.mean(psf_logvar * psf_logvar)

            # ---- regularizers (only when the target is enabled) ----
            if use_psf_uq:
                psf_tv = TV_Loss(psf)
                psf_center = center_loss(psf)
            else:
                psf_tv = jnp.zeros((), dtype=rec.dtype)
                psf_center = jnp.zeros((), dtype=rec.dtype)

            if use_lp_uq:
                lp_tv = TV_Loss(light_pattern)
            else:
                lp_tv = jnp.zeros((), dtype=rec.dtype)

            loss = loss + args.tv_loss * psf_tv + args.psfc_loss * psf_center + args.lp_tv * lp_tv
            loss = loss + float(args.uq_var_reg) * logvar_reg

            # metrics
            metrics = {
                "loss": loss,
                "emitter_nll": emitter_nll,
                "lp_nll": lp_nll,
                "psf_nll": psf_nll,
                "emitter_mse": emitter_mse,
                "lp_mse": lp_mse,
                "psf_mse": psf_mse,
                "emitter_mae": emitter_mae,
                "lp_mae": lp_mae,
                "psf_mae": psf_mae,
                "psf_tv": psf_tv,
                "lp_tv": lp_tv,
                "psf_center": psf_center,
                "logvar_reg": logvar_reg,
                "emitter_std_mean": emitter_std_mean,
                "lp_std_mean": lp_std_mean,
                "psf_std_mean": psf_std_mean,
                "emitter_var_mean": emitter_var_mean,
                "lp_var_mean": lp_var_mean,
                "psf_var_mean": psf_var_mean,
                "emitter_calib_ratio": emitter_calib_ratio,
                "lp_calib_ratio": lp_calib_ratio,
                "emitter_corr": emitter_corr,
                "lp_corr": lp_corr,
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

    P_TRAIN_STEP = jax.pmap(train_step_pmap, axis_name="devices", donate_argnums=(0,))

    # logs
    metrics_csv = os.path.join(args.ckpt_dir, "metrics.csv")
    csv_header = [
        "time",
        "epoch",
        "step",
        "loss",
        "emitter_nll",
        "lp_nll",
        "psf_nll",
        "emitter_mse",
        "lp_mse",
        "psf_mse",
        "emitter_mae",
        "lp_mae",
        "psf_mae",
        "psf_tv",
        "lp_tv",
        "psf_center",
        "logvar_reg",
        "emitter_std_mean",
        "lp_std_mean",
        "psf_std_mean",
        "emitter_var_mean",
        "lp_var_mean",
        "psf_var_mean",
        "emitter_calib_ratio",
        "lp_calib_ratio",
        "emitter_corr",
        "lp_corr",
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
                "emitter_nll": "",
                "lp_nll": "",
                "psf_nll": "",
                "emitter_mse": "",
                "lp_mse": "",
                "psf_mse": "",
                "emitter_mae": "",
                "lp_mae": "",
                "psf_mae": "",
                "psf_tv": "",
                "lp_tv": "",
                "psf_center": "",
                "logvar_reg": "",
                "emitter_std_mean": "",
                "lp_std_mean": "",
                "psf_std_mean": "",
                "emitter_var_mean": "",
                "lp_var_mean": "",
                "psf_var_mean": "",
                "emitter_calib_ratio": "",
                "lp_calib_ratio": "",
                "emitter_corr": "",
                "lp_corr": "",
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
                torch_em = batch[1] if (need_gt and len(batch) > 1) else None
                torch_lp = batch[2] if (need_gt and len(batch) > 2) else None
            else:
                torch_x = batch
                torch_em = None
                torch_lp = None

            dl_np = _to_numpy(torch_x).astype(np.float32)
            np_bchw = ensure_bchw_9(dl_np).astype(np.float32)  # (B,9,H,W)

            x = jnp.asarray(np_bchw)[:, None, ...]  # (B,1,9,H,W)
            x = shard_batch(x, n_devices)

            if need_gt:
                em_np = _to_numpy(torch_em).astype(np.float32)
                lp_np = _to_numpy(torch_lp).astype(np.float32)
                em = jnp.asarray(em_np)
                lp = jnp.asarray(lp_np)
                em = shard_batch(em, n_devices)
                lp = shard_batch(lp, n_devices)
            else:
                # dummy placeholders (won't be used if emitter/lp disabled)
                em = x
                lp = x

            state_rep, metrics = P_TRAIN_STEP(state_rep, x, em, lp)

            m = {k: float(jax.device_get(v)[0]) for k, v in metrics.items()}
            interval_buf.append(m)
            epoch_buf.append(m)

            step = int(jax.device_get(state_rep.step)[0])

            if (it % args.log_every) == 0:
                avg = {k: float(np.mean([d[k] for d in interval_buf])) for k in interval_buf[0].keys()}
                print(
                    f"[epoch {epoch:03d}][iter {it:05d}] "
                    f"loss={avg['loss']:.6f} "
                    f"em_nll={avg['emitter_nll']:.6f} lp_nll={avg['lp_nll']:.6f} psf_nll={avg['psf_nll']:.6f} "
                    f"psf_tv={avg['psf_tv']:.6f} lp_tv={avg['lp_tv']:.6f} psf_center={avg['psf_center']:.6f} "
                    f"em_calib={avg['emitter_calib_ratio']:.3f} lp_calib={avg['lp_calib_ratio']:.3f} "
                    f"em_corr={avg['emitter_corr']:.3f} lp_corr={avg['lp_corr']:.3f} "
                    f"(step={step})"
                )
                csv_append(
                    metrics_csv,
                    {
                        "time": time.time(),
                        "epoch": epoch,
                        "step": step,
                        "loss": avg["loss"],
                        "emitter_nll": avg["emitter_nll"],
                        "lp_nll": avg["lp_nll"],
                        "psf_nll": avg["psf_nll"],
                        "emitter_mse": avg["emitter_mse"],
                        "lp_mse": avg["lp_mse"],
                        "psf_mse": avg["psf_mse"],
                        "emitter_mae": avg["emitter_mae"],
                        "lp_mae": avg["lp_mae"],
                        "psf_mae": avg["psf_mae"],
                        "psf_tv": avg["psf_tv"],
                        "lp_tv": avg["lp_tv"],
                        "psf_center": avg["psf_center"],
                        "logvar_reg": avg["logvar_reg"],
                        "emitter_std_mean": avg["emitter_std_mean"],
                        "lp_std_mean": avg["lp_std_mean"],
                        "psf_std_mean": avg["psf_std_mean"],
                        "emitter_var_mean": avg["emitter_var_mean"],
                        "lp_var_mean": avg["lp_var_mean"],
                        "psf_var_mean": avg["psf_var_mean"],
                        "emitter_calib_ratio": avg["emitter_calib_ratio"],
                        "lp_calib_ratio": avg["lp_calib_ratio"],
                        "emitter_corr": avg["emitter_corr"],
                        "lp_corr": avg["lp_corr"],
                        "note": "train_interval",
                    },
                    header=csv_header,
                )
                interval_buf = []

        epoch_avg = {k: float(np.mean([d[k] for d in epoch_buf])) for k in epoch_buf[0].keys()}

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
        print(
            f"[epoch {epoch:03d}] "
            f"loss={epoch_avg['loss']:.6f} "
            f"em_nll={epoch_avg['emitter_nll']:.6f} lp_nll={epoch_avg['lp_nll']:.6f} psf_nll={epoch_avg['psf_nll']:.6f} "
            f"psf_tv={epoch_avg['psf_tv']:.6f} lp_tv={epoch_avg['lp_tv']:.6f} psf_center={epoch_avg['psf_center']:.6f} "
            f"em_calib={epoch_avg['emitter_calib_ratio']:.3f} lp_calib={epoch_avg['lp_calib_ratio']:.3f} "
            f"em_corr={epoch_avg['emitter_corr']:.3f} lp_corr={epoch_avg['lp_corr']:.3f} "
            f"| time={dt:.1f}s step={step}"
        )
        csv_append(
            metrics_csv,
            {
                "time": time.time(),
                "epoch": epoch,
                "step": step,
                "loss": epoch_avg["loss"],
                "emitter_nll": epoch_avg["emitter_nll"],
                "lp_nll": epoch_avg["lp_nll"],
                "psf_nll": epoch_avg["psf_nll"],
                "emitter_mse": epoch_avg["emitter_mse"],
                "lp_mse": epoch_avg["lp_mse"],
                "psf_mse": epoch_avg["psf_mse"],
                "emitter_mae": epoch_avg["emitter_mae"],
                "lp_mae": epoch_avg["lp_mae"],
                "psf_mae": epoch_avg["psf_mae"],
                "psf_tv": epoch_avg["psf_tv"],
                "lp_tv": epoch_avg["lp_tv"],
                "psf_center": epoch_avg["psf_center"],
                "logvar_reg": epoch_avg["logvar_reg"],
                "emitter_std_mean": epoch_avg["emitter_std_mean"],
                "lp_std_mean": epoch_avg["lp_std_mean"],
                "psf_std_mean": epoch_avg["psf_std_mean"],
                "emitter_var_mean": epoch_avg["emitter_var_mean"],
                "lp_var_mean": epoch_avg["lp_var_mean"],
                "psf_var_mean": epoch_avg["psf_var_mean"],
                "emitter_calib_ratio": epoch_avg["emitter_calib_ratio"],
                "lp_calib_ratio": epoch_avg["lp_calib_ratio"],
                "emitter_corr": epoch_avg["emitter_corr"],
                "lp_corr": epoch_avg["lp_corr"],
                "note": "train_epoch",
            },
            header=csv_header,
        )

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
            gt_em_np = None
            gt_lp_np = None
            if isinstance(debug_batch, (list, tuple)):
                torch_dbg = debug_batch[0]
                if need_gt and len(debug_batch) > 2:
                    gt_em_np = _to_numpy(debug_batch[1]).astype(np.float32)
                    gt_lp_np = _to_numpy(debug_batch[2]).astype(np.float32)
            else:
                torch_dbg = debug_batch
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

            # -------- data uncertainty (heteroscedastic) --------
            emitter = _to_b1_5d(emitter)
            lp = _to_b1_5d(lp)
            psf = _to_b1_5d(psf)
            if deconv is not None:
                deconv = _to_b1_5d(deconv)
            if use_emitter_uq:
                emitter_logvar = _to_b1_5d(emitter_logvar)
                emitter_logvar_c = clip_logvar(emitter_logvar, lv_min, lv_max)
                var_emitter = logvar_to_var(emitter_logvar_c, float(args.uq_var_eps))
            else:
                emitter_logvar_c = None
                var_emitter = jnp.zeros_like(emitter)

            if use_lp_uq:
                lp_logvar = _to_b1_5d(lp_logvar)
                lp_logvar_c = clip_logvar(lp_logvar, lv_min, lv_max)
                var_lp = logvar_to_var(lp_logvar_c, float(args.uq_var_eps))
            else:
                lp_logvar_c = None
                var_lp = jnp.zeros_like(lp)

            if use_psf_uq:
                psf_logvar = _to_b1_5d(psf_logvar)
                psf_logvar_c = clip_logvar(psf_logvar, lv_min, lv_max)
                var_psf = logvar_to_var(psf_logvar_c, float(args.uq_var_eps))
            else:
                psf_logvar_c = None
                var_psf = jnp.zeros_like(psf)

            mu_s = deconv if deconv is not None else (emitter * lp)
            var_s = product_variance(emitter, var_emitter, lp, var_lp)

            var_rec_hr = propagate_rec_variance(
                mu_s,
                var_s,
                psf,
                var_psf,
                psf_var_weight=1.0 if use_psf_uq else 0.0,
            )
            var_rec = downsample_variance(var_rec_hr, args.rescale)

            rec_std = jnp.sqrt(jnp.maximum(var_rec, 0.0))
            emitter_std = jnp.sqrt(jnp.maximum(var_emitter, 0.0))
            lp_std = jnp.sqrt(jnp.maximum(var_lp, 0.0))
            psf_std = jnp.sqrt(jnp.maximum(var_psf, 0.0))
            deconv_std = jnp.sqrt(jnp.maximum(var_s, 0.0))

            uq_pack = {
                "rec_std": np.array(jax.device_get(rec_std)).astype(np.float32),
                "deconv_std": np.array(jax.device_get(deconv_std)).astype(np.float32),
                "rec_var": np.array(jax.device_get(var_rec)).astype(np.float32),
            }
            if use_emitter_uq:
                uq_pack["emitter_std"] = np.array(jax.device_get(emitter_std)).astype(np.float32)
                uq_pack["emitter_logvar"] = np.array(jax.device_get(emitter_logvar_c)).astype(np.float32)
            if use_lp_uq:
                uq_pack["lp_std"] = np.array(jax.device_get(lp_std)).astype(np.float32)
                uq_pack["lp_logvar"] = np.array(jax.device_get(lp_logvar_c)).astype(np.float32)
            if use_psf_uq:
                uq_pack["psf_std"] = np.array(jax.device_get(psf_std)).astype(np.float32)
                uq_pack["psf_logvar"] = np.array(jax.device_get(psf_logvar_c)).astype(np.float32)

            # optional: per-pixel NLL maps using GT (for visualization)
            if use_emitter_uq and gt_em_np is not None:
                gt_em = jnp.asarray(gt_em_np)
                gt_em = _to_b1_5d(gt_em)
                gt_em = _downsample_hw_like(gt_em, emitter)
                gt_em = _align_like(emitter, gt_em)
                diff = emitter - gt_em
                em_nll = 0.5 * (diff * diff / jnp.maximum(var_emitter, 1e-8) + jnp.log(jnp.maximum(var_emitter, 1e-8)))
                uq_pack["emitter_nll"] = np.array(jax.device_get(em_nll)).astype(np.float32)

            if use_lp_uq and gt_lp_np is not None:
                gt_lp = jnp.asarray(gt_lp_np)
                gt_lp = _to_b1_5d(gt_lp)
                gt_lp = _downsample_hw_like(gt_lp, lp)
                gt_lp = _align_like(lp, gt_lp)
                diff = lp - gt_lp
                lp_nll = 0.5 * (diff * diff / jnp.maximum(var_lp, 1e-8) + jnp.log(jnp.maximum(var_lp, 1e-8)))
                uq_pack["lp_nll"] = np.array(jax.device_get(lp_nll)).astype(np.float32)

            # save artifacts + UQ maps
            # optional: load synthetic noise sigma map for comparison
            noise_sigma_np = None
            if debug_path:
                meta_path = _guess_meta_path(str(debug_path))
                if os.path.exists(meta_path):
                    try:
                        with np.load(meta_path) as z:
                            if "noise_sigma" in z:
                                noise_sigma_np = np.asarray(z["noise_sigma"], dtype=np.float32)
                    except Exception:
                        noise_sigma_np = None

            # align GT to prediction resolution for visualization
            gt_em_vis = None
            gt_lp_vis = None
            gt_psf_vis = None
            if gt_em_np is not None:
                gt_em_t = jnp.asarray(gt_em_np)
                gt_em_t = _to_b1_5d(gt_em_t)
                gt_em_t = _downsample_hw_like(gt_em_t, emitter)
                gt_em_t = _align_like(emitter, gt_em_t)
                gt_em_vis = np.array(jax.device_get(gt_em_t)).astype(np.float32)
            if gt_lp_np is not None:
                gt_lp_t = jnp.asarray(gt_lp_np)
                gt_lp_t = _to_b1_5d(gt_lp_t)
                gt_lp_t = _downsample_hw_like(gt_lp_t, lp)
                gt_lp_t = _align_like(lp, gt_lp_t)
                gt_lp_vis = np.array(jax.device_get(gt_lp_t)).astype(np.float32)
            if psf_gt is not None:
                gt_psf_t = _to_b1_5d(psf_gt)
                gt_psf_t = _downsample_hw_like(gt_psf_t, psf)
                gt_psf_t = _align_like(psf, gt_psf_t)
                gt_psf_vis = np.array(jax.device_get(gt_psf_t)).astype(np.float32)

            gt_pack = {
                "emitter_gt": gt_em_vis,
                "lp_gt": gt_lp_vis,
                "psf_gt": gt_psf_vis,
            }
            noise_pack = {"noise_sigma": noise_sigma_np} if noise_sigma_np is not None else None

            dbg_metrics = save_debug_artifacts_data_uq_sup(
                out_dir=out_dir,
                args=args,
                debug_path=str(debug_path),
                crop_record=crop_record,
                x_np=x_np,
                rec_np=rec_np,
                lp_np=lp_np,
                emitter_np=emitter_np,
                psf_np=psf_np,
                mask_np=mask_np,
                deconv_np=deconv_np,
                uq_pack=uq_pack,
                gt_pack=gt_pack,
                noise_pack=noise_pack,
            )
            # update calibration curves from all supervised debug samples
            try:
                save_calibration_curves_from_dir(args.debug_dir, bins=10)
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
                    "emitter_nll": "",
                    "lp_nll": "",
                    "psf_nll": "",
                    "emitter_mse": "",
                    "lp_mse": "",
                    "psf_mse": "",
                    "emitter_mae": "",
                    "lp_mae": "",
                    "psf_mae": "",
                    "psf_tv": dbg_metrics["psf_tv"],
                    "lp_tv": dbg_metrics["lp_tv"],
                    "psf_center": dbg_metrics["psf_center"],
                    "logvar_reg": "",
                    "emitter_std_mean": dbg_metrics.get("emitter_std_mean", ""),
                    "lp_std_mean": dbg_metrics.get("lp_std_mean", ""),
                    "psf_std_mean": dbg_metrics.get("psf_std_mean", ""),
                    "emitter_var_mean": "",
                    "lp_var_mean": "",
                    "psf_var_mean": "",
                    "emitter_calib_ratio": "",
                    "lp_calib_ratio": "",
                    "emitter_corr": "",
                    "lp_corr": "",
                    "note": note,
                },
                header=csv_header,
            )
            print(f"[debug] saved to: {out_dir}")


if __name__ == "__main__":
    main()

# 例：
'''
CUDA_VISIBLE_DEVICES=1,2,3,4 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  uv run python train_data_uq_sup.py --train_glob "/data/repo/SIMFormer/data/SIM-simulation/*/*/standard/train/*.tif" \
  --batch_size 7 --epochs 500 --lr 1e-4 --ckpt_dir "./ckpts_uq_sup" \
  --uq_targets emitter,lp --sup_weight_emitter 1.0 --sup_weight_lp 1.0
'''
