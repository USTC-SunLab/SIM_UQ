# simple_data_uq

一个最小可复现的 **heteroscedastic UQ** 验证：
- 数据：2D emitter + 空间异方差噪声
- 模型：小型 CNN 输出 mean + logvar
- 目标：验证预测不确定性与真实噪声强度对齐

## 1) 生成数据
```bash
uv run python simple_data_uq/generate_simple_dataset.py \
  --out_dir "./simple_data_uq/data" \
  --n_train 1000 --n_val 200 \
  --h 80 --w 80
```

## 2) 训练（JAX 版本，使用 GPU 1-7）
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
  uv run python simple_data_uq/train_simple_uq_jax.py \
  --data_dir "./simple_data_uq/data" \
  --ckpt_dir "./simple_data_uq/ckpts_jax" \
  --epochs 50 --batch_size 32 --lr 1e-3 \
  --debug_every 5 \
  --sigma_weight 0.2 --sigma_weight_end 0.6 --sigma_warmup_epochs 20 \
  --base 48
```

## 3) 输出
- `simple_data_uq/ckpts_jax/metrics.csv`：loss / mse / nll / calib_ratio / corr / sigma_mae / sigma_corr
- `simple_data_uq/ckpts/metrics_curves.png`：loss / metric 趋势图
- `simple_data_uq/ckpts/debug/epoch_xxx/`：
  - `pred_gt.png`：预测 mean vs GT
  - `uq_vs_sigma.png`：预测 std vs 真实 sigma
  - `input_pred_gt.png`
  - `error_vs_sigma.png`
  - `calibration.png`

这能用来确认 **“异方差 NLL + logvar”** 是否能正确量化噪声水平。
