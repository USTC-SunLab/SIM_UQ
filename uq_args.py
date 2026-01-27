# uq_args.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import argparse
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("UQ multi-device training")

    # ---------------- data ----------------
    p.add_argument("--train_glob", type=str, required=True)
    p.add_argument("--crop_h", type=int, default=80)
    p.add_argument("--crop_w", type=int, default=80)

    # 原脚本里 type=tuple 在命令行基本不可用，这里改成 nargs=2 更稳
    p.add_argument("--rescale", type=int, nargs=2, default=(3, 3))
    p.add_argument("--crop_size", type=int, nargs=2, default=(80, 80))

    # ---------------- model ----------------
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lrc9_rank", type=int, default=64)

    # ---------------- train ----------------
    p.add_argument("--batch_size", type=int, default=7)  # must be divisible by n_devices
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--add_noise", type=float, default=1.0)
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)

    # ---------------- loss weights ----------------
    p.add_argument("--tv_loss", type=float, default=1e-3)      # psf_tv weight
    p.add_argument("--lp_tv", type=float, default=1e-3)        # light_pattern tv weight
    p.add_argument("--psfc_loss", type=float, default=1e-1)    # psf center weight
    p.add_argument("--psfg_loss", type=float, default=1e-2)    # gaussian prior weight
    p.add_argument("--psf_sigma", type=float, default=3.0)     # gaussian sigma

    # ---------------- ckpt & logs ----------------
    p.add_argument("--ckpt_dir", type=str, default="./ckpts_uq")
    p.add_argument("--resume_s1_path", type=str, default=None)
    p.add_argument("--keep_ckpts", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_pickle", type=str, default="/data/repo/SIMFormer/ckpt/BioSR_pretrain/pretrain_params.pkl")

    # ---------------- debug saving (optional) ----------------
    p.add_argument("--debug_dir", type=str, default="", help="default: ckpt_dir/debug_samples")
    p.add_argument("--debug_every_epochs", type=int, default=1, help="0=disable")
    p.add_argument("--debug_seed", type=int, default=123, help="fixed debug seed (does not affect training)")
    p.add_argument("--debug_batch_size", type=int, default=1, help="debug forward batch size (single device)")
    p.add_argument("--debug_frames", type=int, default=3, help="how many frames (<=9) to show")
    p.add_argument("--debug_train", action="store_true", help="debug forward uses training=True")
    p.add_argument("--debug_index", type=int, default=0, help="use which file index in train_paths as debug input")
    p.add_argument("--vis_pmin", type=float, default=1.0, help="pct-style vmin percentile")
    p.add_argument("--vis_pmax", type=float, default=99.0, help="pct-style vmax percentile")

    # ---------------- MC Dropout uncertainty (debug-only) ----------------
    p.add_argument("--mc_dropout_train_samples", type=int, default=0,
                   help="MC Dropout samples for uncertainty during debug saving. >1 enables uncertainty.")
    p.add_argument("--mc_device", type=str, default="cpu", choices=["cpu", "gpu"],
                   help="Device to run MC sampling. 'cpu' avoids extra GPU fragmentation.")
    p.add_argument("--mc_device_id", type=int, default=None,
                   help="Preferred GPU device id for MC sampling (JAX device id). If None, use last GPU.")
    p.add_argument("--mc_mask_ratio", type=float, default=0.0,
                   help="Override mask_ratio during MC sampling (often set to 0 for full reconstruction).")
    p.add_argument("--mc_disable_noise", action="store_true",
                   help="If set, disable add_noise during MC sampling (if args has add_noise field).")
    p.add_argument("--mc_seed", type=int, default=0,
                   help="Base seed for MC sampling; 0 means reuse debug_seed.")
    p.add_argument("--mc_target", type=str, default="both", choices=["rec", "deconv", "both"],
                   help="Which output to estimate uncertainty for.")

    return p


def parse_args(argv: Optional[Sequence[str]] = None):
    p = build_parser()
    args = p.parse_args(argv)

    # normalize tuples
    args.rescale = tuple(args.rescale)
    args.crop_size = tuple(args.crop_size)

    # abs paths + debug_dir default
    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not str(args.debug_dir).strip():
        args.debug_dir = os.path.join(args.ckpt_dir, "debug_samples")
    args.debug_dir = os.path.abspath(args.debug_dir)

    return args
