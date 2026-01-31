# uq_selfsup

Self-supervised data UQ for SIM raw images (JAX/Flax), using the **original project model** (`PiMAE` in `network.py`) and the **original self-supervised UQ pipeline** described in `README_data_uq.md`.

This trains on **input-only** data by default: the model predicts emitter / light pattern / PSF, reconstructs SIM raw, and (optionally) uses heteroscedastic NLL between reconstruction and input. Uncertainty is propagated through the physical forward.

## Loss modes
Three modes are supported via `--loss_mode`:
- `selfsup_uq` (default): heteroscedastic NLL on recon vs input, with variance propagation.
- `emitter_uq`: supervised heteroscedastic NLL on **emitter GT** only (no recon loss).
- `plain`: no UQ, self-supervised recon loss (L1 + MS-SSIM), aligned with `model.py::compute_metrics`.

## Data
- Input only (no GT): `*.tif` with 9 frames.
- Only `--train_glob` is required.

## Run (HR=2x dataset)
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  uv run python uq_selfsup/train_selfsup_uq.py \
  --train_glob "/data/repo/SIM_UQ/simple_data_uq/sim_dataset_hr2_types/curve/train/*.tif" \
  --val_glob "/data/repo/SIM_UQ/simple_data_uq/sim_dataset_hr2_types/curve/val/*.tif" \
  --batch_size 12 --epochs 200 --lr 1e-4 \
  --debug_every_epochs 5 \
  --uq_aux_rec_weight 0.1 \
  --uq_psf_var_weight 1.0 \
  --uq_targets emitter,lp,psf \
  --loss_mode selfsup_uq \
  --rescale 2 2 \
  --use_gt_metrics
```

Notes:
- `ckpt_dir` is **auto timestamped** each run. Use `--ckpt_dir_fixed` to disable.
- With `CUDA_VISIBLE_DEVICES=2,3,4,5,6,7` (6 GPUs), `--batch_size 12` means 2 per device.
- For **plain (no UQ)** training, SIMFormer-style hyperparams that often help:
  - `--mask_ratio 0.25` (mask is supported; parser comes from `uq_args.py`)
  - `--weight_decay 0`
  - `--lrc9_rank 32`

### Example: emitter-only supervised UQ
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  uv run python uq_selfsup/train_selfsup_uq.py \
  --train_glob "/data/repo/SIM_UQ/simple_data_uq/sim_dataset_hr2_types/curve/train/*.tif" \
  --val_glob "/data/repo/SIM_UQ/simple_data_uq/sim_dataset_hr2_types/curve/val/*.tif" \
  --batch_size 12 --epochs 200 --lr 1e-4 \
  --loss_mode emitter_uq \
  --uq_targets emitter \
  --rescale 2 2 \
  --use_gt_metrics
```

### Example: plain self-supervised (no UQ)
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  uv run python uq_selfsup/train_selfsup_uq.py \
  --train_glob "/data/repo/SIM_UQ/simple_data_uq/sim_dataset_hr2_types/curve/train/*.tif" \
  --val_glob "/data/repo/SIM_UQ/simple_data_uq/sim_dataset_hr2_types/curve/val/*.tif" \
  --batch_size 12 --epochs 200 --lr 1e-4 \
  --loss_mode plain \
  --mask_ratio 0.25 \
  --rescale 3 3 \
  --lrc9_rank 32 \
  --weight_decay 0 \
  --use_gt_metrics
```

## Notes
- Uses `PiMAE` (original model) and the self-supervised UQ propagation from `README_data_uq.md`.
- `--uq_targets` selects which components output log-variance (ignored for `loss_mode=plain`).
- `--val_glob` is optional; when provided, val metrics are logged each epoch.
- If GT exists alongside the input (e.g. `train_gt/*.tif` + `*_lp.tif`), enable `--use_gt_metrics` to include GT-based metrics and visualizations in debug output.
- `loss_mode=emitter_uq` requires emitter GT and `uq_targets` includes `emitter`. It disables recon loss by design.
- Emitter GT refers to the **pre-light-pattern** emitter (the saved `train_gt/*.tif`). `emitter_real_gt` is `emitter_gt * lp_gt`.
- Optimizer aligns with SIMFormer: **grad clip (1.0)** + **linear warmup to lr** over `7 * steps_per_epoch`, then constant lr.

## Train UQ std on top of a plain model
Load a converged plain checkpoint, freeze the base, and train only logvar heads (e.g., emitter std):
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  uv run python uq_selfsup/train_selfsup_uq.py \
  --train_glob "/data/repo/SIM_UQ/simple_data_uq/sim_dataset_hr2_types/curve/train/*.tif" \
  --val_glob "/data/repo/SIM_UQ/simple_data_uq/sim_dataset_hr2_types/curve/val/*.tif" \
  --batch_size 12 --epochs 200 --lr 1e-4 \
  --loss_mode selfsup_uq \
  --uq_targets emitter \
  --freeze_base \
  --init_from_ckpt "/path/to/plain/ckpt_dir" \
  --mask_ratio 0.25 \
  --weight_decay 0 \
  --rescale 3 3
```

## Visualization (debug)
Debug outputs are saved under `debug_samples/.../` (single folder):
- Core: `input.png`, `input_mean.png`, `recon.png`, `recon_mean.png`, `recon_err.png`, `recon_err_mean.png`
- Predictions: `lp_pred.png`, `lp_mean.png`, `emitter_mu_pred.png`, `lightfield_pred.png`, `psf.png`, `mask.png`
- Labels (GT if available): `lp_label.png`, `emitter_real_label.png`, `lightfield_label.png`, plus diffs `lp_diff.png`, `emitter_diff.png`
- UQ (when enabled): `emitter_std.png`, and `recon_std.png` (only in `loss_mode=selfsup_uq`)

## Calibration
- `calibration_emitter.png`: emitter var vs (emitter_pred - emitter_gt)^2 (requires GT + emitter UQ).
- `calibration_recon.png`: recon var vs (rec - input)^2 (only in `loss_mode=selfsup_uq`).

## Curves
Training curves are saved under `ckpt_dir/curves/` each epoch:
- `loss.png`, `rec_nll.png`, `rec_mse.png`, `psnr.png`, `emitter_nll.png`
- `metrics_grid.png` (loss/rec_nll/psnr/emitter_nll)
