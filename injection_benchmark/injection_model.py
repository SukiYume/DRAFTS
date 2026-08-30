# fmt: off

"""注入基准使用的 FRB 动态谱模型。

这里放纯数学函数，不读写 FITS，也不负责参数抽样。`generate_injections.py` 会调用这些
函数生成每个频率通道上的高斯/散射高斯时间 profile，再叠加到真实背景数据中。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import special

DM_DELAY_SECONDS = 4148808.0 / 1000.0
TAU_FREQ_INDEX   = 4.0


def gaussian_profile(x, center, sigma):
    """普通高斯脉冲；给 sigma 加下限，避免极窄脉冲导致除零。"""
    sigma = np.maximum(sigma, 1e-6)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _scattered_gaussian_raw(relative_times, sigma, taus):
    """未归一化的散射高斯，即高斯脉冲卷积单边指数拖尾。"""
    tau = np.maximum(taus, 1e-3)
    y   = relative_times / sigma
    z   = (sigma / tau - y) / np.sqrt(2.0)
    log_pdf = (
        -np.log(2.0 * tau)
        - 0.5 * y * y
        + np.log(special.erfcx(np.clip(z, -25.0, 25.0)))
    )

    tail = z < -20.0
    if np.any(tail):
        log_tail = -np.log(tau) + 0.5 * (sigma / tau) ** 2 - relative_times / tau
        log_pdf  = np.where(tail, log_tail, log_pdf)
    return np.exp(np.clip(log_pdf, -745.0, 80.0))


def scattered_gaussian_profile(times, centers, sigma, taus):
    """生成逐通道散射高斯 profile，并按理论峰值归一到 1。

    ``times`` and ``centers`` use the same unit, usually raw sample bins.  The
    peak normalization is window-independent so a truncated tail is not
    renormalized into a false bright stripe.
    """
    times_2d = np.asarray(times, dtype=np.float64)[None, :]
    centers  = np.asarray(centers, dtype=np.float64)[:, None]
    taus     = np.asarray(taus, dtype=np.float64)[:, None]
    sigma    = max(float(sigma), 1e-3)

    out           = np.empty((centers.shape[0], times_2d.shape[1]), dtype=np.float64)
    gaussian_mask = (taus[:, 0] <= 1e-3) | (sigma / np.maximum(taus[:, 0], 1e-9) > 8.0)
    if np.any(gaussian_mask):
        out[gaussian_mask] = gaussian_profile(times_2d, centers[gaussian_mask], sigma)

    scat_mask = ~gaussian_mask
    if np.any(scat_mask):
        tau            = np.maximum(taus[scat_mask], 1e-3)
        relative_times = times_2d - centers[scat_mask]
        raw            = _scattered_gaussian_raw(relative_times, sigma, tau)

        peak_grid = np.linspace(-8.0 * sigma, 12.0 * sigma, 513, dtype=np.float64)[
            None, :
        ]
        raw_peak = np.nanmax(
            _scattered_gaussian_raw(peak_grid, sigma, tau), axis=1, keepdims=True
        )
        raw_peak       = np.where(np.isfinite(raw_peak) & (raw_peak > 0), raw_peak, 1.0)
        out[scat_mask] = raw / raw_peak

    return out


@lru_cache(maxsize=1)
def _scattered_l2_energy_table():
    """Tabulate the L2 energy of a peak-normalized scattered Gaussian.

    The profile shape depends only on ``tau / sigma``.  Tabulating that ratio
    once avoids constructing a full channel-by-time template for every
    injection while retaining the same peak normalization used above.
    """
    ratios         = np.geomspace(1.0 / 8.0, 1.0e3, 2048, dtype=np.float64)
    relative_times = np.linspace(-8.0, 12.0, 513, dtype=np.float64)[None, :]
    raw = _scattered_gaussian_raw(
        relative_times,
        1.0,
        ratios[:, None],
    )
    raw_peak               = np.nanmax(raw, axis=1)
    raw_energy             = special.erfcx(1.0 / ratios) / (2.0 * ratios)
    peak_normalized_energy = raw_energy / np.square(raw_peak)
    return np.log(ratios), peak_normalized_energy


def pulse_l2_energy_samples(sigma_samples, tau_samples):
    """Return ``sum(profile**2)`` for each peak-normalized time profile.

    The result uses the continuous-profile integral expressed in sample units.
    FAST search-mode pulses in this workflow span several raw samples even at
    the minimum width, so this is an accurate and much cheaper approximation
    to constructing every discrete profile solely for normalization.
    """
    sigma     = max(float(sigma_samples), 1e-3)
    taus      = np.asarray(tau_samples, dtype=np.float64)
    flat_taus = taus.reshape(-1)
    energy    = np.full(flat_taus.shape, np.sqrt(np.pi) * sigma, dtype=np.float64)
    scattered = (flat_taus > 1e-3) & (sigma / np.maximum(flat_taus, 1e-9) <= 8.0)
    if np.any(scattered):
        log_ratios, energy_ratios = _scattered_l2_energy_table()
        ratios                    = flat_taus[scattered] / sigma
        energy[scattered] = sigma * np.interp(
            np.log(ratios),
            log_ratios,
            energy_ratios,
        )
    return energy.reshape(taus.shape)


def template_l2_norm(freq_envelope, sigma_time_samples, tau_samples):
    """L2 norm of the whitened frequency-time injection template.

    If a template with this norm is multiplied by
    ``target_fluence_snr / norm``, its ideal matched-filter S/N is the target
    value under independent unit-variance noise.
    """
    envelope = np.asarray(freq_envelope, dtype=np.float64)
    taus     = np.asarray(tau_samples, dtype=np.float64)
    if envelope.ndim != 1 or taus.shape != envelope.shape:
        raise ValueError(
            "Frequency envelope and scattering array must be 1-D and aligned"
        )
    norm_squared = float(
        np.sum(np.square(envelope) * pulse_l2_energy_samples(sigma_time_samples, taus))
    )
    if not np.isfinite(norm_squared) or norm_squared <= 0:
        raise ValueError("Injection template has a non-positive L2 norm")
    return float(np.sqrt(norm_squared))

# fmt: on
