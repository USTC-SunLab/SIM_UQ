# uq_data_uq_utils.py
# -*- coding: utf-8 -*-
"""
Utilities for data (aleatoric) uncertainty via heteroscedastic regression.
Designed for JAX/Flax training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


def clip_logvar(logvar: jnp.ndarray, min_logvar: float, max_logvar: float) -> jnp.ndarray:
    logvar = jnp.maximum(logvar, float(min_logvar))
    logvar = jnp.minimum(logvar, float(max_logvar))
    return logvar


def logvar_to_var(logvar: jnp.ndarray, eps: float) -> jnp.ndarray:
    return jnp.exp(logvar) + float(eps)


def gaussian_nll_from_var(diff: jnp.ndarray, var: jnp.ndarray, mask: Optional[jnp.ndarray] = None, eps: float = 1e-8) -> jnp.ndarray:
    var = jnp.maximum(var, float(eps))
    logvar = jnp.log(var)
    nll = 0.5 * (diff * diff / var + logvar)
    if mask is None:
        return jnp.mean(nll)
    m = mask.astype(nll.dtype)
    return jnp.sum(nll * m) / (jnp.sum(m) + float(eps))


def product_variance(mu_a: jnp.ndarray, var_a: jnp.ndarray, mu_b: jnp.ndarray, var_b: jnp.ndarray) -> jnp.ndarray:
    # Var(A*B) for independent A,B
    return (mu_a * mu_a) * var_b + (mu_b * mu_b) * var_a + var_a * var_b


def convolve_fft_bz(x: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """
    Convolve each (Z,H,W) slice with the same PSF kernel.
    x: (B,1,Z,H,W), k: (1,1,1,H,W) or broadcastable.
    """
    x = jnp.asarray(x)
    k = jnp.asarray(k)
    if x.ndim != 5:
        raise ValueError(f"convolve_fft_bz: x must be 5D, got {x.shape}")
    if k.ndim != 5:
        raise ValueError(f"convolve_fft_bz: k must be 5D, got {k.shape}")

    B, _, Z, H, W = x.shape
    x_f = x.reshape((-1, H, W))

    # Use the first kernel (PSF shared across batch)
    k0 = k[:1]
    k_h, k_w = int(k0.shape[-2]), int(k0.shape[-1])
    k_f = k0.reshape((1, k_h, k_w))

    y_f = jax.scipy.signal.fftconvolve(x_f, k_f, mode="same")
    return y_f.reshape((B, 1, Z, H, W))


def propagate_rec_variance(
    mu_s: jnp.ndarray,
    var_s: jnp.ndarray,
    mu_psf: jnp.ndarray,
    var_psf: jnp.ndarray,
    psf_var_weight: float = 1.0,
) -> jnp.ndarray:
    # base term: var from S with mean PSF
    term_s = convolve_fft_bz(var_s, mu_psf * mu_psf)
    if float(psf_var_weight) <= 0.0:
        return term_s
    term_p = convolve_fft_bz(mu_s * mu_s, var_psf)
    term_sp = convolve_fft_bz(var_s, var_psf)
    return term_s + float(psf_var_weight) * (term_p + term_sp)


def downsample_variance(var_hr: jnp.ndarray, rescale: Tuple[int, int]) -> jnp.ndarray:
    """
    Downsample variance to match avg_pool in PiMAE.
    var(mean) = avg(var) / N, N = r0*r1
    """
    r0, r1 = int(rescale[0]), int(rescale[1])
    if r0 <= 1 and r1 <= 1:
        return var_hr
    var_hw = var_hr.transpose((0, 2, 3, 4, 1))
    var_lr = nn.avg_pool(var_hw, window_shape=(1, r0, r1), strides=(1, r0, r1), padding="VALID")
    var_lr = var_lr.transpose((0, 4, 1, 2, 3))
    n = float(r0 * r1)
    return var_lr / n
