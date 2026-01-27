# img_compare.py
import math
import numpy as np

def to_2d(x):
    """torch/numpy -> float32 2D. If >2D, reduce leading dims by mean."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim < 2:
        raise ValueError(f"expected >=2D, got shape={x.shape}")
    if x.ndim > 2:
        x = x.mean(axis=tuple(range(x.ndim - 2)))
    return x.astype(np.float32)

def _normalize_like_ref(ref, x, method="pminmax", p=(1, 99), eps=1e-8):
    """Normalize x using params computed from ref (same transform for all)."""
    method = (method or "none").lower()
    ref = ref.astype(np.float32)
    x = x.astype(np.float32)

    if method in ("none",):
        dr = float(ref.max() - ref.min())
        return ref, x, (dr if dr > eps else 1.0)

    if method in ("minmax", "pminmax"):
        if method == "minmax":
            lo, hi = float(ref.min()), float(ref.max())
        else:
            lo, hi = np.percentile(ref, p[0]), np.percentile(ref, p[1])

        if hi - lo < eps:
            z = np.zeros_like(ref)
            return z, np.zeros_like(x), 1.0

        ref = np.clip(ref, lo, hi)
        x = np.clip(x, lo, hi)
        ref = (ref - lo) / (hi - lo)
        x = (x - lo) / (hi - lo)
        return ref, x, 1.0  # now in [0,1]

    if method in ("zscore", "meanstd"):
        mu, std = float(ref.mean()), float(ref.std())
        std = std if std > eps else 1.0
        ref = (ref - mu) / std
        x = (x - mu) / std
        dr = float(ref.max() - ref.min())
        return ref, x, (dr if dr > eps else 1.0)

    raise ValueError(f"unknown norm method: {method}")

def compare_images(ref, x, metric="mse", norm="pminmax", **norm_kwargs):
    """
    Compare x to ref with a chosen metric after applying the SAME normalization
    (params computed from ref).

    metric: mse | psnr | ssim | ms-ssim
    norm: none | minmax | pminmax | zscore
    """
    ref = to_2d(ref)
    x = to_2d(x)
    if ref.shape != x.shape:
        raise ValueError(f"shape mismatch: ref {ref.shape} vs x {x.shape}")

    ref, x, data_range = _normalize_like_ref(ref, x, method=norm, **norm_kwargs)
    metric = metric.lower()

    if metric == "mse":
        return float(np.mean((ref - x) ** 2))

    if metric == "psnr":
        mse = float(np.mean((ref - x) ** 2))
        if mse == 0.0:
            return float("inf")
        return 20 * math.log10(data_range) - 10 * math.log10(mse)

    if metric == "ssim":
        from skimage.metrics import structural_similarity as ssim_fn
        return float(ssim_fn(ref, x, data_range=data_range))

    if metric in ("ms-ssim", "ms_ssim", "msssim"):
        import torch
        from pytorch_msssim import ms_ssim as ms_ssim_fn
        ref_t = torch.from_numpy(ref)[None, None]  # (1,1,H,W)
        x_t = torch.from_numpy(x)[None, None]
        return float(ms_ssim_fn(ref_t, x_t, data_range=data_range).item())

    raise ValueError(f"unknown metric: {metric}")
