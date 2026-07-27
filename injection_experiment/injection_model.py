"""注入实验使用的 FRB 动态谱模型。

这里放纯数学函数，不读写 FITS，也不负责参数抽样。`generate_injections.py` 会调用这些
函数生成每个频率通道上的高斯/散射高斯时间 profile，再叠加到真实背景数据中。
"""

from __future__ import annotations

import numpy as np
from scipy import special


DM_DELAY_SECONDS = 4148808.0 / 1000.0
TAU_FREQ_INDEX = 4.0


def gaussian_profile(x, center, sigma):
    """普通高斯脉冲；给 sigma 加下限，避免极窄脉冲导致除零。"""
    sigma = np.maximum(sigma, 1e-6)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _scattered_gaussian_raw(relative_times, sigma, taus):
    """未归一化的散射高斯，即高斯脉冲卷积单边指数拖尾。"""
    tau = np.maximum(taus, 1e-3)
    y = relative_times / sigma
    z = (sigma / tau - y) / np.sqrt(2.0)
    log_pdf = -np.log(2.0 * tau) - 0.5 * y * y + np.log(special.erfcx(np.clip(z, -25.0, 25.0)))

    tail = z < -20.0
    if np.any(tail):
        log_tail = -np.log(tau) + 0.5 * (sigma / tau) ** 2 - relative_times / tau
        log_pdf = np.where(tail, log_tail, log_pdf)
    return np.exp(np.clip(log_pdf, -745.0, 80.0))


def scattered_gaussian_profile(times, centers, sigma, taus):
    """生成逐通道散射高斯 profile，并按理论峰值归一到 1。

    ``times`` and ``centers`` use the same unit, usually raw sample bins.  The
    peak normalization is window-independent so a truncated tail is not
    renormalized into a false bright stripe.
    """
    times_2d = np.asarray(times, dtype=np.float64)[None, :]
    centers = np.asarray(centers, dtype=np.float64)[:, None]
    taus = np.asarray(taus, dtype=np.float64)[:, None]
    sigma = max(float(sigma), 1e-3)

    out = np.empty((centers.shape[0], times_2d.shape[1]), dtype=np.float64)
    gaussian_mask = (taus[:, 0] <= 1e-3) | (sigma / np.maximum(taus[:, 0], 1e-9) > 8.0)
    if np.any(gaussian_mask):
        out[gaussian_mask] = gaussian_profile(times_2d, centers[gaussian_mask], sigma)

    scat_mask = ~gaussian_mask
    if np.any(scat_mask):
        tau = np.maximum(taus[scat_mask], 1e-3)
        relative_times = times_2d - centers[scat_mask]
        raw = _scattered_gaussian_raw(relative_times, sigma, tau)

        peak_grid = np.linspace(-8.0 * sigma, 12.0 * sigma, 513, dtype=np.float64)[None, :]
        raw_peak = np.nanmax(_scattered_gaussian_raw(peak_grid, sigma, tau), axis=1, keepdims=True)
        raw_peak = np.where(np.isfinite(raw_peak) & (raw_peak > 0), raw_peak, 1.0)
        out[scat_mask] = raw / raw_peak

    return out
