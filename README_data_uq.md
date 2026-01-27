# Data UQ (Heteroscedastic) Training

本仓库新增了**数据不确定度（aleatoric）**训练脚本，使用异方差回归并适配自监督场景。  
核心思想：网络不仅输出重建均值，还输出 emitter / light pattern / psf 的 **log-variance**（可选），再把这些不确定度沿物理前向过程传播到重建域，对 SIM 原图做 **NLL 损失**。

## 新增文件
- `uq_train_data_uq.py`  
  新训练脚本：多卡 pmap + 异方差回归 + 自监督 NLL。
- `uq_data_uq_utils.py`  
  不确定度相关工具：logvar clamp、variance 传播、NLL、downsample 等。
- `uq_vis_data_uq.py`  
  debug 可视化：保存 std map / hist / summary。

## 方法概述（自监督适配）
- 观测：SIM 原图 `x`  
- 输出均值：`emitter`, `light_pattern`, `psf`  
- 输出方差：`logvar_emitter`, `logvar_lp`, `logvar_psf`（可通过 `--uq_targets` 选择）  
- 传播：
  - `S = emitter * light_pattern`
  - `Var(S) ≈ μ_e^2 Var(l) + μ_l^2 Var(e) + Var(e)Var(l)`
  - `Var(rec) ≈ conv(Var(S), μ_psf^2) + conv(μ_S^2, Var(psf)) + conv(Var(S), Var(psf))`
  - 再按 `avg_pool` 规则下采样（方差除以窗口像素数）
- 损失：  
  `NLL = 0.5 * ( (x - rec)^2 / Var(rec) + log Var(rec) )`  
  可选加 `L1 / MS-SSIM` 辅助项（`--uq_aux_rec_weight`）。

## 运行示例
```bash
uv run python uq_train_data_uq.py \
  --train_glob "/data/repo/SIMFormer/data/SIM-simulation/*/*/standard/train/*.tif" \
  --batch_size 7 --epochs 500 --lr 1e-4 \
  --ckpt_dir "./ckpts_uq_data" \
  --debug_every_epochs 5 \
  --uq_aux_rec_weight 0.1 \
  --uq_psf_var_weight 1.0
```

## 关键参数
- `--uq_logvar_min / --uq_logvar_max`  
  log-variance 裁剪区间，避免数值爆炸。
- `--uq_logvar_init`  
  logvar head 的 bias 初始化（默认 -4.0）。
- `--uq_var_floor`  
  传播后的方差下限。
- `--uq_psf_var_weight`  
  PSF 不确定度对 `Var(rec)` 的贡献权重（设 0 关闭）。
- `--uq_targets`  
  选择哪些分量启用数据不确定度（logvar/std）。默认 `emitter,lp,psf`。  
  例如仅对 emitter/lp 做 UQ：`--uq_targets emitter,lp`（PSF 将不输出 logvar/std）。
- `--uq_aux_rec_weight` / `--uq_aux_rec_mode`  
  是否叠加 L1 / MS-SSIM 辅助重建损失。

## Debug 输出
`debug_samples/epoch_xxx_step_xxx/` 内新增：
- `uq_uncertainty.npz`：rec/deconv std + 各分量 logvar/std（仅包含 `--uq_targets` 启用的项）
- `rec_std.png`, `lp_std.png`, `emitter_std.png`, `psf_std.png`, `deconv_std.png`
- `*_hist.png` 直方图
- `uq_summary.json` 统计均值与 p95
- （监督版）`emitter_nll.png`, `lp_nll.png`：基于 GT 的逐像素 NLL 可视化（需要 `train_data_uq_sup.py` 且 GT 可用）
- 训练日志新增 UQ 校准指标：`emitter_calib_ratio / lp_calib_ratio`（MSE/Var）、`emitter_corr / lp_corr`（Var 与误差的相关系数）

## 监督版可视化（整合对比）
监督训练的 debug 输出会生成更“成对”的对比图（减少杂乱）：  
- `input.png`：输入 SIM 9 帧  
- `emitter_pred_gt.png`：emitter 预测 vs GT  
- `lp_pred_gt.png`：LP 预测 vs GT  
- `psf_pred_gt.png`：PSF 预测 vs GT  
- `uq_vs_noise.png`：预测不确定性（rec_std）vs 合成噪声 sigma（需要使用合成数据）  

## Loss / Metric 曲线
训练完成后可用脚本画曲线：
```bash
uv run python plot_metrics_curves.py --metrics_csv "./ckpts_uq_sup/metrics.csv"
```

## 校准曲线（Calibration Curves）
校准曲线用于评估**预测方差是否与真实误差匹配**。  
按预测方差分箱，比较每个 bin 中的**平均预测方差**与**真实 MSE**，理想曲线接近 y=x。

```bash
uv run python plot_calibration_curves.py --root "./ckpts_uq_sup/debug_samples"
```

监督训练时，校准曲线已**自动集成到 debug 可视化**中：  
`debug_samples/calibration/calib_emitter.png`、`debug_samples/calib_lp.png`

## 注意事项
- 该脚本**不修改原 `uq_train.py`**。
- 传播方差会额外进行 FFT 卷积，训练速度会略慢。
- 当前实现使用独立性近似（详见 `data_uq.md` 的方法说明）。

## 合成数据集（用于验证 emitter / LP 的数据不确定性）
提供脚本生成带**空间异方差噪声**的 SIM 合成数据，可用于验证 UQ（观察 emitter/lp 的 NLL 与 std 对齐情况）。

```bash
uv run python gen_uq_sup_dataset.py \
  --out_dir "./data_uq_sup_sim" \
  --n_samples 1000 \
  --lr_h 80 --lr_w 80 \
  --hr_factor 6
```

数据结构：
- `data_uq_sup_sim/train/*.tif`            9 帧 SIM 输入（LR）
- `data_uq_sup_sim/train_gt/*.tif`         emitter GT（HR）
- `data_uq_sup_sim/train_gt/*_lp.tif`      lp GT（HR）
- `data_uq_sup_sim/train_meta/*.npz`       噪声 σ map / mask / psf_sigma（用于验证）

使用监督训练：
```bash
uv run python train_data_uq_sup.py \
  --train_glob "./data_uq_sup_sim/train/*.tif" \
  --ckpt_dir "./ckpts_uq_sup" \
  --batch_size 7 --epochs 200 --lr 1e-4 \
  --uq_targets emitter,lp \
  --gt_norm minmax_full
```
