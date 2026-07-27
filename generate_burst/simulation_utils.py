"""time-DM 训练数据生成的模拟信号与 FAST 背景数据工具。

两块内容、一个模块：
  A. FRB 动态谱模型     —— 频率高斯包络、（散射）高斯时间 profile、色散延迟常数
  B. FAST 背景数据工具  —— 背景 FITS 发现、头信息读取/校验、每通道噪声 std 估计

generate_dataset.py 把本模块同时当作 ``model``（用 A）和 ``inj``（用 B）。
完整的“注入→写 raw8/2-bit FITS→搜索”流程不在本模块中；若需该流程，请使用
`injection_experiment/generate_injections.py` 和 `injection_experiment/run_campaign.py`。
"""

from __future__ import annotations

import concurrent.futures
import math
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import special


# ===========================================================================
# A. FRB 动态谱模型
# ===========================================================================

# 色散延迟常数 K_DM（单位：s·MHz^2·pc^-1·cm^3）：t = K_DM·DM·(f^-2 - f_ref^-2)
DM_DELAY_SECONDS = 4148808.0 / 1000.0


def gaussian_profile(x, center, sigma):
    """高斯 profile（sigma 给个很小的下界防除零）。频率包络和纯高斯时间脉冲都用它。"""
    sigma = np.maximum(sigma, 1e-6)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _scattered_gaussian_raw(relative_times, sigma, taus):
    """未归一化的「高斯 ⊛ 单边指数尾」（ex-Gaussian），用 erfcx 提升数值稳定性。"""
    tau = np.maximum(taus, 1e-3)
    y = relative_times / sigma
    z = (sigma / tau - y) / np.sqrt(2.0)
    log_pdf = -np.log(2.0 * tau) - 0.5 * y * y + np.log(special.erfcx(np.clip(z, -25.0, 25.0)))

    # z 很负时 erfcx 溢出，改用解析的指数尾渐近式
    tail = z < -20.0
    if np.any(tail):
        log_tail = -np.log(tau) + 0.5 * (sigma / tau) ** 2 - relative_times / tau
        log_pdf = np.where(tail, log_tail, log_pdf)
    return np.exp(np.clip(log_pdf, -745.0, 80.0))


def scattered_gaussian_profile(times, centers, sigma, taus):
    """每通道散射高斯时间 profile，峰值归一化到 1。

    ``times`` 与 ``centers`` 同单位（一般是 raw 采样 bin）。**频率依赖 τ(ν)∝ν^-4 由调用方
    先算好后通过 ``taus``（每通道一个）传入**，本函数不做频率标度。峰值归一与窗口无关，
    被截断的散射尾不会被错误地重新归一成假亮条纹。tau≈0 或 sigma/tau>8 时退化为纯高斯。
    返回形状 [n_channel, n_time]。
    """
    times_2d = np.asarray(times, dtype=np.float64)[None, :]
    centers = np.asarray(centers, dtype=np.float64)[:, None]
    taus = np.asarray(taus, dtype=np.float64)[:, None]
    sigma = max(float(sigma), 1e-3)

    out = np.empty((centers.shape[0], times_2d.shape[1]), dtype=np.float64)
    # 散射可忽略的通道走纯高斯（更快更稳）
    gaussian_mask = (taus[:, 0] <= 1e-3) | (sigma / np.maximum(taus[:, 0], 1e-9) > 8.0)
    if np.any(gaussian_mask):
        out[gaussian_mask] = gaussian_profile(times_2d, centers[gaussian_mask], sigma)

    # 其余通道算 ex-Gaussian，再用一段密集网格上的最大值做峰值归一
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


# ===========================================================================
# B. FAST 背景数据工具
# ===========================================================================

# --- B1. 背景文件发现 ------------------------------------------------------
def file_number(path_or_stem: str) -> int:
    """从形如 ``*-M01_0123`` 的文件名解析 4 位编号（0123 -> 123）。"""
    match = re.search(r"[-_]M\d+_(\d{4})$", Path(path_or_stem).stem)
    if not match:
        raise ValueError(f"Cannot parse M01 file number from {path_or_stem}")
    return int(match.group(1))


def list_background_fits(background_dir: Path, file_first: int, file_last: int) -> list[Path]:
    """列出 [file_first, file_last] 区间内的 M01 背景 FITS（按编号），数量不符直接报缺哪些。"""
    pattern = re.compile(r".*-M01_(\d{4})\.fits$")
    files = []
    for path in sorted(background_dir.glob("*.fits")):
        match = pattern.match(path.name)
        if not match:
            continue
        if "_N_" in path.name or "_W_" in path.name or "_F_" in path.name:
            continue
        number = int(match.group(1))
        if file_first <= number <= file_last:
            files.append(path)
    expected = file_last - file_first + 1
    if len(files) != expected:
        found = [file_number(path.stem) for path in files]
        missing = [num for num in range(file_first, file_last + 1) if num not in found]
        raise FileNotFoundError(
            f"Expected {expected} M01 FITS files in {background_dir}, found {len(files)}; missing {missing[:10]}"
        )
    return files


# --- B2. 头信息读取 / 一致性校验 / 频率轴 ----------------------------------
def read_metadata(path: Path) -> dict:
    """读单个 FITS 的关键头信息（采样数、通道数、时间分辨率、中心频率/带宽等）。"""
    with fits.open(path, memmap=True) as hdul:
        h0 = hdul[0].header
        h1 = hdul[1].header
        meta = {
            "naxis2": int(h1["NAXIS2"]),
            "nsblk": int(h1["NSBLK"]),
            "npol": int(h1["NPOL"]),
            "nchan": int(h1["NCHAN"]),
            "nbits": int(h1["NBITS"]),
            "time_reso_seconds": float(h1["TBIN"]),
            "obsfreq_mhz": float(h0["OBSFREQ"]),
            "obsbw_mhz": float(h0["OBSBW"]),
        }
    meta["samples_per_file"] = meta["naxis2"] * meta["nsblk"]
    meta["duration_per_file_s"] = meta["samples_per_file"] * meta["time_reso_seconds"]
    return meta


def validate_consistent_headers(files: list[Path], meta0: dict) -> None:
    """校验其余文件的关键头信息与首文件一致（防止混入不同观测设置的数据）。"""
    keys = ("naxis2", "nsblk", "npol", "nchan", "nbits", "time_reso_seconds", "obsfreq_mhz", "obsbw_mhz")
    for path in files[1:]:
        meta = read_metadata(path)
        mismatches = {key: (meta0[key], meta[key]) for key in keys if meta0[key] != meta[key]}
        if mismatches:
            raise ValueError(f"Header mismatch in {path}: {mismatches}")


def frequency_axis(meta: dict) -> np.ndarray:
    """由中心频率/带宽/通道数生成每通道频率（MHz），通道 0 为频带低端。"""
    return (
        meta["obsfreq_mhz"]
        - meta["obsbw_mhz"] / 2.0
        + np.arange(meta["nchan"], dtype=np.float64) * meta["obsbw_mhz"] / meta["nchan"]
    )


# --- B3. 每通道背景噪声 std 估计（注入幅度的标度）-------------------------
def data_5d_view(data_field: np.ndarray, meta: dict) -> np.ndarray:
    """把 FITS DATA 列还原成 [rows, nsblk, npol, nchan, 1] 视图（校验尾维为 1）。"""
    rows = meta["naxis2"]
    nsblk = meta["nsblk"]
    npol = meta["npol"]
    nchan = meta["nchan"]
    arr = np.asarray(data_field)
    view = arr.reshape(rows, nsblk, npol, nchan, -1)
    if view.shape[-1] != 1:
        raise ValueError(f"Unexpected DATA trailing dimension: {view.shape}")
    return view


def sampled_stokes_i_from_data(data_field: np.ndarray, meta: dict, max_samples: int) -> np.ndarray:
    """沿时间均匀抽 max_samples 个采样，前两路偏振取平均得到近似 Stokes-I [samples, nchan]。"""
    view = data_5d_view(data_field, meta).reshape(
        meta["samples_per_file"],
        meta["npol"],
        meta["nchan"],
    )
    sample_count = min(meta["samples_per_file"], max_samples)
    if sample_count <= 0:
        raise ValueError("max_samples must be positive")
    indices = np.linspace(0, meta["samples_per_file"] - 1, sample_count, dtype=np.int64)
    return 0.5 * (
        view[indices, 0, :].astype(np.float32)
        + view[indices, 1, :].astype(np.float32)
    )


def robust_channel_std(stokes_i: np.ndarray, max_samples: int) -> np.ndarray:
    """每通道稳健噪声 std：用 MAD×1.4826（异常时退化到普通 std，再用中位数兜底无效通道）。"""
    if stokes_i.shape[0] > max_samples:
        step = int(math.ceil(stokes_i.shape[0] / max_samples))
        sample = stokes_i[::step][:max_samples]
    else:
        sample = stokes_i
    med = np.nanmedian(sample, axis=0)
    mad = np.nanmedian(np.abs(sample - med[None, :]), axis=0)
    std = 1.4826 * mad
    fallback = np.nanstd(sample, axis=0)
    std = np.where(np.isfinite(std) & (std > 0.1), std, fallback)
    floor = float(np.nanmedian(std[np.isfinite(std) & (std > 0)]))
    return np.where(np.isfinite(std) & (std > 0), std, floor).astype(np.float32)


def estimate_std_job(job: dict) -> dict:
    """单文件 std 估计任务（可在子进程里跑）：读 DATA → 抽样 Stokes-I → 稳健 std。"""
    path = Path(job["path"])
    meta = job["meta"]
    max_samples_per_file = int(job["max_samples_per_file"])
    with fits.open(path, memmap=True) as source:
        stokes_i = sampled_stokes_i_from_data(source[1].data["DATA"], meta, max_samples_per_file)
        std = robust_channel_std(stokes_i, max_samples_per_file)
    return {"index": int(job["index"]), "name": path.name, "std": std}


def estimate_global_channel_std(files: list[Path], meta: dict, max_samples_per_file: int, workers: int) -> np.ndarray:
    """对若干背景文件分别估计每通道 std，再取跨文件中位数作为全局每通道噪声标度。

    workers<=1 时串行，否则用进程池并行；结果按文件顺序聚合后对通道取中位数。
    """
    std_rows = []
    jobs = [
        {"index": idx, "path": str(path), "meta": meta, "max_samples_per_file": max_samples_per_file}
        for idx, path in enumerate(files, start=1)
    ]
    if workers <= 1:
        for job in jobs:
            result = estimate_std_job(job)
            std_rows.append(result["std"])
            print(f"[std] {result['index']:03d}/{len(files):03d} {result['name']}", flush=True)
    else:
        ordered: list[np.ndarray | None] = [None] * len(jobs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_index = {executor.submit(estimate_std_job, job): job["index"] for job in jobs}
            for future in concurrent.futures.as_completed(future_to_index):
                result = future.result()
                ordered[result["index"] - 1] = result["std"]
                print(f"[std] {result['index']:03d}/{len(files):03d} {result['name']}", flush=True)
        std_rows = [row for row in ordered if row is not None]
    return np.nanmedian(np.vstack(std_rows), axis=0).astype(np.float32)
