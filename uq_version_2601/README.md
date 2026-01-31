# uq_version_2601

Simple supervised data UQ for SIM raw images (JAX/Flax). This is a lightweight baseline that predicts the emitter **mean + log-variance** directly from the 9-frame SIM input, without the full physics pipeline.

## Data assumptions
- SIM input: `*.tif` with 9 frames (shape like 9xH xW or HxW x9).
- Emitter GT: `*.tif` in a parallel folder, same basename as input.
- LP GT can exist but is **not used** here.
- Optional noise meta: `train_meta/*.npz` / `val_meta/*.npz` with `noise_sigma` (from `simple_data_uq/generate_sim_dataset_with_noise.py`).
  This is **only used for evaluation/visualization**, not for training.

Default path rule:
- input: `/train/xxx.tif`
- emitter GT: `/train_gt/xxx.tif`
- lp GT: `/train_gt/xxx_lp.tif`
- noise meta: `/train_meta/xxx.npz` (fallback `/val_meta/xxx.npz`)

You can change this with `--gt_dir_token`, `--gt_dir_repl`, `--gt_emitter_suffix`, `--gt_lp_suffix`.
Meta location can be changed with `--meta_dir_token`, `--meta_dir_repl`.

## Run
Example on the SIM noisy dataset (multi‑GPU):
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  uv run python uq_version_2601/train_simple_uq_sim.py \
  --train_glob "./simple_data_uq/sim_dataset/train/*.tif" \
  --val_glob "./simple_data_uq/sim_dataset/val/*.tif" \
  --ckpt_dir "./uq_version_2601/ckpts" \
  --epochs 120 --batch_size 12 --lr 1e-3 \
  --debug_every 5 \
  --crop_size 80 80 \
  --sigma_norm input_scale --sigma_reduce mean
```

## Outputs
- `ckpts/metrics.csv`          : train/val metrics per epoch
- `ckpts/loss_curve.png`       : loss curve
- `ckpts/debug/epoch_xxx/`     :
  - `input_frames.png`         : 9-frame input grid
  - `input_pred_gt.png`        : input mean vs pred vs GT
  - `pred_gt.png`              : pred mean vs GT
  - `pred_gt_separate.png`     : pred vs GT with separate normalization
- `pred_std.png`             : predicted std map
- `std_gt.png`               : GT std map (from train_meta/noise_sigma when available)
- `std_pred_vs_gt.png`       : pred std vs GT std
- `error_vs_sigma.png`        : abs error vs GT sigma (if meta exists)
  - `calibration.png`          : calibration curve

## Notes
- This script uses JAX, so CUDA is available when JAX detects GPU devices.
- If the emitter GT is higher resolution than input, it is **downsampled** to match the input size for loss and calibration.
- `sigma_norm=input_scale` scales `noise_sigma` by the same factor used to normalize inputs.
- `--sigma_weight*` flags are deprecated and ignored (no sigma supervision).
- This is meant as a sanity-check baseline for data UQ.
