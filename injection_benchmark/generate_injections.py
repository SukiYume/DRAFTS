"""Inject simulated FRBs into full-length FAST search-mode FITS files.

Default parameter ranges and distributions
------------------------------------------

Background:
    /path/to/generation_data/rawdata
Files:
    FRB20220912A_tracking-M01_0011.fits through M01_0279.fits.  M01_0280
    is a shorter tail file in the current rawdata copy and is intentionally
    excluded so global truth sample indices remain uniform.
Output:
    simdata/<run_label>_raw8
    simdata/<run_label>_packed2

Injected signal parameters:
    DM:
        Latin-hypercube stratified uniform in [100, 2000] pc cm^-3.
    Frequency-integrated peak S/N:
        Latin-hypercube log-uniform in [5, 100].
    Intrinsic FWHM:
        Latin-hypercube log-uniform in [1, 20] ms.
    Frequency FWHM:
        Latin-hypercube log-uniform in [50, 500] MHz. The component center is
        sampled across the observed band; out-of-band tails are truncated by
        the finite observing bandwidth.
    Scattering:
        15% exactly zero. Otherwise tau_1GHz is log-uniform in
        [0.03, 10] ms and scales as tau proportional to frequency^-4.
    Center frequency:
        Uniform over the observed band.

The script preserves the full source FITS length and table structure for the
raw8 output. It writes packed2 FITS files for DRAFTS-style 2-bit evaluation by
quantizing the injected Stokes-I stream per frequency channel.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits

from injection_model import (
    DM_DELAY_SECONDS,
    TAU_FREQ_INDEX,
    gaussian_profile,
    scattered_gaussian_profile,
)


DEFAULT_BACKGROUND_DIR = Path("/path/to/generation_data/rawdata")
DEFAULT_OUTPUT_ROOT = Path("simdata")
DEFAULT_RUN_LABEL = "eval001"
DEFAULT_FILE_FIRST = 11
DEFAULT_FILE_LAST = 279
DEFAULT_COUNT = 512
DEFAULT_SEED = 20260613

DM_RANGE_PC_CM3 = (100.0, 2000.0)
SNR_RANGE = (5.0, 100.0)
WIDTH_FWHM_MS_RANGE = (1.0, 20.0)
BANDWIDTH_FWHM_MHZ_RANGE = (50.0, 500.0)
SCATTERING_ZERO_FRACTION = 0.15
SCATTERING_TAU_1GHZ_MS_RANGE = (0.03, 10.0)
MIN_TOA_SEPARATION_S = 0.24
EDGE_GUARD_S = 0.18
MODEL_SUPPORT_SIGMA = 8.0
CHANNEL_CHUNK = 128
IO_SEMAPHORE = None


@dataclass(frozen=True)
class InjectionTruth:
    injection_id: str
    run_label: str
    source_file: str
    output_file_stem: str
    output_file_index: int
    output_file_number: int
    highest_freq_toa_global_raw_sample: int
    highest_freq_toa_file_raw_sample: int
    highest_freq_toa_seconds: float
    dm_pc_cm3: float
    snr: float
    width_ms_fwhm: float
    sigma_time_ms: float
    bandwidth_mhz_fwhm: float
    sigma_freq_mhz: float
    center_freq_mhz: float
    scattering_ms_at_1ghz: float
    scattering_ms_at_highest_freq: float
    freq_low_mhz: float
    freq_high_mhz: float
    effective_channels_fwhm: float
    per_channel_peak_snr: float
    nchan: int
    nsblk: int
    rows_per_file: int
    naxis2: int
    time_reso_seconds: float


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def init_process_worker(io_semaphore) -> None:
    global IO_SEMAPHORE
    IO_SEMAPHORE = io_semaphore


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def file_number(path_or_stem: str) -> int:
    match = re.search(r"[-_]M\d+_(\d{4})$", Path(path_or_stem).stem)
    if not match:
        raise ValueError(f"Cannot parse M01 file number from {path_or_stem}")
    return int(match.group(1))


def list_background_fits(background_dir: Path, file_first: int, file_last: int) -> list[Path]:
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


def read_metadata(path: Path) -> dict:
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
    keys = ("naxis2", "nsblk", "npol", "nchan", "nbits", "time_reso_seconds", "obsfreq_mhz", "obsbw_mhz")
    for path in files[1:]:
        meta = read_metadata(path)
        mismatches = {key: (meta0[key], meta[key]) for key in keys if meta0[key] != meta[key]}
        if mismatches:
            raise ValueError(f"Header mismatch in {path}: {mismatches}")


def frequency_axis(meta: dict) -> np.ndarray:
    return (
        meta["obsfreq_mhz"]
        - meta["obsbw_mhz"] / 2.0
        + np.arange(meta["nchan"], dtype=np.float64) * meta["obsbw_mhz"] / meta["nchan"]
    )


def lhs_unit(count: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty((count, dimensions), dtype=np.float64)
    for dim in range(dimensions):
        values[:, dim] = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        rng.shuffle(values[:, dim])
    return values


def log_uniform(lo: float, hi: float, unit_value: float) -> float:
    return float(np.exp(np.log(lo) + unit_value * (np.log(hi) - np.log(lo))))


def sample_parameter_rows(count: int, rng: np.random.Generator, scattering_ref_freq_mhz: float) -> list[dict]:
    unit = lhs_unit(count, 5, rng)
    rows = []
    for row in unit:
        width_ms = log_uniform(WIDTH_FWHM_MS_RANGE[0], WIDTH_FWHM_MS_RANGE[1], row[0])
        bandwidth_mhz = log_uniform(BANDWIDTH_FWHM_MHZ_RANGE[0], BANDWIDTH_FWHM_MHZ_RANGE[1], row[1])
        snr = log_uniform(SNR_RANGE[0], SNR_RANGE[1], row[2])
        dm = DM_RANGE_PC_CM3[0] + row[3] * (DM_RANGE_PC_CM3[1] - DM_RANGE_PC_CM3[0])
        if row[4] < SCATTERING_ZERO_FRACTION:
            scattering_1ghz_ms = 0.0
        else:
            scattering_1ghz_ms = log_uniform(
                SCATTERING_TAU_1GHZ_MS_RANGE[0],
                SCATTERING_TAU_1GHZ_MS_RANGE[1],
                (row[4] - SCATTERING_ZERO_FRACTION) / (1.0 - SCATTERING_ZERO_FRACTION),
            )
        scattering_high_ms = scattering_1ghz_ms * (scattering_ref_freq_mhz / 1000.0) ** (-TAU_FREQ_INDEX)
        rows.append({
            "width_ms_fwhm": width_ms,
            "bandwidth_mhz_fwhm": bandwidth_mhz,
            "snr": snr,
            "dm_pc_cm3": float(dm),
            "scattering_ms_at_1ghz": float(scattering_1ghz_ms),
            "scattering_ms_at_highest_freq": float(scattering_high_ms),
        })
    return rows


def max_delay_samples(dm_pc_cm3: float, freqs_mhz: np.ndarray, time_reso_seconds: float) -> int:
    ref = float(np.nanmax(freqs_mhz))
    delay_s = DM_DELAY_SECONDS * dm_pc_cm3 * (np.nanmin(freqs_mhz) ** -2 - ref ** -2)
    return int(math.ceil(delay_s / time_reso_seconds))


def support_samples(params: dict, freqs_mhz: np.ndarray, time_reso_seconds: float) -> int:
    ref = float(np.nanmax(freqs_mhz))
    min_freq = float(np.nanmin(freqs_mhz))
    sigma_time_ms = params["width_ms_fwhm"] / 2.355
    max_scat_ms = params["scattering_ms_at_highest_freq"] * (min_freq / ref) ** (-TAU_FREQ_INDEX)
    support_s = max(
        EDGE_GUARD_S,
        MODEL_SUPPORT_SIGMA * (sigma_time_ms + max_scat_ms) / 1000.0,
    )
    return int(math.ceil(support_s / time_reso_seconds))


def choose_toas(
    params: list[dict],
    total_samples: int,
    freqs_mhz: np.ndarray,
    time_reso_seconds: float,
    rng: np.random.Generator,
    allowed_start_sample: int = 0,
    allowed_stop_sample: int | None = None,
) -> list[int]:
    """为每个注入选一个高频到达采样，保证彼此最小间隔且色散尾不越界。"""
    out = []
    previous = []
    min_sep = int(round(MIN_TOA_SEPARATION_S / time_reso_seconds))
    base_guard = int(round(EDGE_GUARD_S / time_reso_seconds))
    if allowed_stop_sample is None:
        allowed_stop_sample = total_samples
    allowed_start_sample = int(allowed_start_sample)
    allowed_stop_sample = int(allowed_stop_sample)
    if allowed_start_sample < 0 or allowed_stop_sample > total_samples or allowed_stop_sample <= allowed_start_sample:
        raise ValueError(
            f"Invalid injection TOA window: start={allowed_start_sample} "
            f"stop={allowed_stop_sample} total={total_samples}"
        )
    for row in params:
        delay = max_delay_samples(row["dm_pc_cm3"], freqs_mhz, time_reso_seconds)
        guard = max(base_guard, support_samples(row, freqs_mhz, time_reso_seconds))
        # highest-frequency arrival stays inside the injection window; the
        # dispersed/scattered tail may extend into the surrounding search guard files.
        earliest = max(guard, allowed_start_sample + guard)
        latest = min(total_samples - delay - guard - 1, allowed_stop_sample - guard - 1)
        if latest <= earliest:
            raise ValueError(
                "Selected injection window is too short for the sampled high-DM signal: "
                f"earliest={earliest} latest={latest} delay={delay} guard={guard}"
            )
        for _ in range(1024):
            toa = int(rng.integers(earliest, latest))
            if all(abs(toa - old) >= min_sep for old in previous):
                previous.append(toa)
                out.append(toa)
                break
        else:
            raise ValueError(
                "Cannot place all injections while preserving the minimum TOA "
                f"separation: placed={len(previous)} window=[{earliest}, {latest}) "
                f"min_separation_samples={min_sep}. Reduce --count, widen the "
                "injection file window, or lower MIN_TOA_SEPARATION_S."
            )
    return out


def injection_sample_window(args: argparse.Namespace, files: list[Path], meta: dict) -> tuple[int, int]:
    samples_per_file = int(meta["samples_per_file"])
    index_by_number = {file_number(path.stem): index for index, path in enumerate(files)}
    missing = [
        number
        for number in range(args.inject_file_first, args.inject_file_last + 1)
        if number not in index_by_number
    ]
    if missing:
        raise ValueError(
            f"Injection file window {args.inject_file_first}-{args.inject_file_last} "
            f"is not contained in selected files; missing {missing[:10]}"
        )
    start_index = index_by_number[args.inject_file_first]
    stop_index = index_by_number[args.inject_file_last] + 1
    return start_index * samples_per_file, stop_index * samples_per_file


def build_truth_rows(args: argparse.Namespace, files: list[Path], meta: dict) -> list[InjectionTruth]:
    """采样所有注入信号的参数与到达时间，生成真值（truth）列表。

    ``highest_freq_toa_global_raw_sample`` 是最高频通道到达时刻的全局原始采样数
    （= output_file_index * samples_per_file + 文件内采样），评估端正是按这个口径
    把候选对齐到同一条时间轴。
    """
    rng = np.random.default_rng(args.seed)
    freqs = frequency_axis(meta)
    freq_min = float(np.nanmin(freqs))
    freq_max = float(np.nanmax(freqs))
    params = sample_parameter_rows(args.count, rng, freq_max)
    total_samples = len(files) * meta["samples_per_file"]
    inject_start, inject_stop = injection_sample_window(args, files, meta)
    toas = choose_toas(
        params,
        total_samples,
        freqs,
        meta["time_reso_seconds"],
        rng,
        allowed_start_sample=inject_start,
        allowed_stop_sample=inject_stop,
    )

    truths = []
    for index, (row, toa) in enumerate(zip(params, toas), start=1):
        file_index = int(toa // meta["samples_per_file"])
        file_toa = int(toa - file_index * meta["samples_per_file"])
        source_file = files[file_index]
        bandwidth = row["bandwidth_mhz_fwhm"]
        center_freq = float(rng.uniform(freq_min, freq_max))
        sigma_freq = bandwidth / 2.355
        sigma_time = row["width_ms_fwhm"] / 2.355
        envelope = gaussian_profile(freqs, center_freq, sigma_freq)
        effective_channels = max(1.0, float(np.sum(envelope > 0.5)))
        per_channel_snr = row["snr"] / math.sqrt(effective_channels)
        truths.append(InjectionTruth(
            injection_id=f"{args.run_label}_inj{index:05d}",
            run_label=args.run_label,
            source_file=str(source_file),
            output_file_stem=source_file.stem,
            output_file_index=file_index,
            output_file_number=file_number(source_file.stem),
            highest_freq_toa_global_raw_sample=int(toa),
            highest_freq_toa_file_raw_sample=file_toa,
            highest_freq_toa_seconds=float(toa * meta["time_reso_seconds"]),
            dm_pc_cm3=row["dm_pc_cm3"],
            snr=row["snr"],
            width_ms_fwhm=row["width_ms_fwhm"],
            sigma_time_ms=sigma_time,
            bandwidth_mhz_fwhm=bandwidth,
            sigma_freq_mhz=sigma_freq,
            center_freq_mhz=center_freq,
            scattering_ms_at_1ghz=row["scattering_ms_at_1ghz"],
            scattering_ms_at_highest_freq=row["scattering_ms_at_highest_freq"],
            freq_low_mhz=max(freq_min, center_freq - bandwidth / 2.0),
            freq_high_mhz=min(freq_max, center_freq + bandwidth / 2.0),
            effective_channels_fwhm=effective_channels,
            per_channel_peak_snr=per_channel_snr,
            nchan=meta["nchan"],
            nsblk=meta["nsblk"],
            rows_per_file=meta["naxis2"],
            naxis2=meta["naxis2"],
            time_reso_seconds=meta["time_reso_seconds"],
        ))
    return truths


def data_5d_view(data_field: np.ndarray, meta: dict) -> np.ndarray:
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
    path = Path(job["path"])
    meta = job["meta"]
    max_samples_per_file = int(job["max_samples_per_file"])
    with fits.open(path, memmap=True) as source:
        stokes_i = sampled_stokes_i_from_data(source[1].data["DATA"], meta, max_samples_per_file)
        std = robust_channel_std(stokes_i, max_samples_per_file)
    return {"index": int(job["index"]), "name": path.name, "std": std}


def estimate_global_channel_std(files: list[Path], meta: dict, max_samples_per_file: int, workers: int) -> np.ndarray:
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


def truth_overlaps_file(truth: InjectionTruth, file_start: int, file_stop: int, freqs_mhz: np.ndarray) -> bool:
    params = {
        "width_ms_fwhm": truth.width_ms_fwhm,
        "scattering_ms_at_highest_freq": truth.scattering_ms_at_highest_freq,
    }
    support = support_samples(params, freqs_mhz, truth.time_reso_seconds)
    delay = max_delay_samples(truth.dm_pc_cm3, freqs_mhz, truth.time_reso_seconds)
    return (
        truth.highest_freq_toa_global_raw_sample - support < file_stop
        and truth.highest_freq_toa_global_raw_sample + delay + support >= file_start
    )


def add_injections_to_data(
    data_field: np.ndarray,
    file_global_start: int,
    truths: list[InjectionTruth],
    freqs_mhz: np.ndarray,
    chan_std: np.ndarray,
    meta: dict,
) -> tuple[int, list[tuple[int, int]]]:
    """把落在本文件内的注入信号（色散+散射+频谱包络）叠加到 DATA 数组上。

    这是 packed2 路径用的版本：直接改 ``data_field`` 数组，返回叠加过信号的
    行窗口，供调用方回写。raw8-only 路径用 ``add_injections_to_table_rows``。
    """
    view = data_5d_view(data_field, meta)
    n_time = meta["samples_per_file"]
    nchan = meta["nchan"]
    file_global_stop = file_global_start + n_time
    overlapping = [
        truth for truth in truths
        if truth_overlaps_file(truth, file_global_start, file_global_stop, freqs_mhz)
    ]
    if not overlapping:
        return 0, []

    accumulated = np.zeros((n_time, nchan), dtype=np.float32)
    ref_freq = float(np.nanmax(freqs_mhz))
    touched_windows: list[tuple[int, int]] = []
    has_signal = False
    for truth in overlapping:
        sigma_bins = (truth.sigma_time_ms / 1000.0) / truth.time_reso_seconds
        tau_ref_bins = (truth.scattering_ms_at_highest_freq / 1000.0) / truth.time_reso_seconds
        support = support_samples(
            {
                "width_ms_fwhm": truth.width_ms_fwhm,
                "scattering_ms_at_highest_freq": truth.scattering_ms_at_highest_freq,
            },
            freqs_mhz,
            truth.time_reso_seconds,
        )
        all_delays = DM_DELAY_SECONDS * truth.dm_pc_cm3 * (freqs_mhz ** -2 - ref_freq ** -2)
        all_centers = truth.highest_freq_toa_global_raw_sample + all_delays / truth.time_reso_seconds
        window_start = int(math.floor(float(np.nanmin(all_centers)) - support))
        window_stop = int(math.ceil(float(np.nanmax(all_centers)) + support)) + 1
        local_start = max(0, window_start - file_global_start)
        local_stop = min(n_time, window_stop - file_global_start)
        if local_stop <= local_start:
            continue
        touched_windows.append((local_start, local_stop))
        for start in range(0, nchan, CHANNEL_CHUNK):
            end = min(start + CHANNEL_CHUNK, nchan)
            freqs = freqs_mhz[start:end]
            envelope = gaussian_profile(freqs, truth.center_freq_mhz, truth.sigma_freq_mhz).astype(np.float32)
            if float(np.nanmax(envelope)) <= 1e-5:
                continue
            delays = DM_DELAY_SECONDS * truth.dm_pc_cm3 * (freqs ** -2 - ref_freq ** -2)
            centers = truth.highest_freq_toa_global_raw_sample + delays / truth.time_reso_seconds
            chunk_start = max(
                local_start,
                int(math.floor(float(np.nanmin(centers)) - support)) - file_global_start,
            )
            chunk_stop = min(
                local_stop,
                int(math.ceil(float(np.nanmax(centers)) + support)) + 1 - file_global_start,
            )
            if chunk_stop <= chunk_start:
                continue
            global_times = file_global_start + np.arange(chunk_start, chunk_stop, dtype=np.float64)
            tau_by_freq = tau_ref_bins * (freqs / ref_freq) ** (-TAU_FREQ_INDEX)
            pulse = scattered_gaussian_profile(global_times, centers, sigma_bins, tau_by_freq).astype(np.float32)
            model = truth.per_channel_peak_snr * envelope[:, None] * pulse
            accumulated[chunk_start:chunk_stop, start:end] += (
                model.T * chan_std[start:end][None, :]
            ).astype(np.float32)
            has_signal = True

    merged_windows: list[tuple[int, int]] = []
    if has_signal:
        for start, stop in sorted(touched_windows):
            if not merged_windows or start > merged_windows[-1][1]:
                merged_windows.append((start, stop))
            else:
                prev_start, prev_stop = merged_windows[-1]
                merged_windows[-1] = (prev_start, max(prev_stop, stop))
        data_tpf = view.reshape(n_time, meta["npol"], nchan)
        for pol in (0, 1):
            for start, stop in merged_windows:
                base = data_tpf[start:stop, pol, :].astype(np.float32)
                updated = np.clip(np.rint(base + accumulated[start:stop]), 0, 255).astype(np.uint8)
                data_tpf[start:stop, pol, :] = updated
    return len(overlapping), merged_windows


def add_injections_to_table_rows(
    table_data,
    file_global_start: int,
    truths: list[InjectionTruth],
    freqs_mhz: np.ndarray,
    chan_std: np.ndarray,
    meta: dict,
) -> int:
    """raw8-only 路径：按 subint 行块就地把注入信号叠加到 table 行的 DATA 上。"""
    n_time = meta["samples_per_file"]
    nchan = meta["nchan"]
    nsblk = meta["nsblk"]
    npol = meta["npol"]
    file_global_stop = file_global_start + n_time
    ref_freq = float(np.nanmax(freqs_mhz))

    truth_windows = []
    row_ranges: list[tuple[int, int]] = []
    for truth in truths:
        if not truth_overlaps_file(truth, file_global_start, file_global_stop, freqs_mhz):
            continue
        sigma_bins = (truth.sigma_time_ms / 1000.0) / truth.time_reso_seconds
        tau_ref_bins = (truth.scattering_ms_at_highest_freq / 1000.0) / truth.time_reso_seconds
        support = support_samples(
            {
                "width_ms_fwhm": truth.width_ms_fwhm,
                "scattering_ms_at_highest_freq": truth.scattering_ms_at_highest_freq,
            },
            freqs_mhz,
            truth.time_reso_seconds,
        )
        all_delays = DM_DELAY_SECONDS * truth.dm_pc_cm3 * (freqs_mhz ** -2 - ref_freq ** -2)
        all_centers = truth.highest_freq_toa_global_raw_sample + all_delays / truth.time_reso_seconds
        window_start = int(math.floor(float(np.nanmin(all_centers)) - support))
        window_stop = int(math.ceil(float(np.nanmax(all_centers)) + support)) + 1
        local_start = max(0, window_start - file_global_start)
        local_stop = min(n_time, window_stop - file_global_start)
        if local_stop <= local_start:
            continue
        row_start = max(0, local_start // nsblk)
        row_stop = min(meta["naxis2"], (local_stop + nsblk - 1) // nsblk)
        truth_windows.append((truth, local_start, local_stop, sigma_bins, tau_ref_bins, support))
        row_ranges.append((row_start, row_stop))

    if not truth_windows:
        return 0

    merged_rows: list[tuple[int, int]] = []
    for start, stop in sorted(row_ranges):
        if not merged_rows or start > merged_rows[-1][1]:
            merged_rows.append((start, stop))
        else:
            prev_start, prev_stop = merged_rows[-1]
            merged_rows[-1] = (prev_start, max(prev_stop, stop))

    for row_start, row_stop in merged_rows:
        block = np.array([table_data[row_idx]["DATA"] for row_idx in range(row_start, row_stop)], copy=True)
        block_time = (row_stop - row_start) * nsblk
        block_file_start = row_start * nsblk
        block_file_stop = block_file_start + block_time
        accumulated = np.zeros((block_time, nchan), dtype=np.float32)
        has_signal = False

        for truth, local_start, local_stop, sigma_bins, tau_ref_bins, support in truth_windows:
            intersect_start = max(local_start, block_file_start)
            intersect_stop = min(local_stop, block_file_stop)
            if intersect_stop <= intersect_start:
                continue
            for start in range(0, nchan, CHANNEL_CHUNK):
                end = min(start + CHANNEL_CHUNK, nchan)
                freqs = freqs_mhz[start:end]
                envelope = gaussian_profile(freqs, truth.center_freq_mhz, truth.sigma_freq_mhz).astype(np.float32)
                if float(np.nanmax(envelope)) <= 1e-5:
                    continue
                delays = DM_DELAY_SECONDS * truth.dm_pc_cm3 * (freqs ** -2 - ref_freq ** -2)
                centers = truth.highest_freq_toa_global_raw_sample + delays / truth.time_reso_seconds
                chunk_start = max(
                    intersect_start,
                    int(math.floor(float(np.nanmin(centers)) - support)) - file_global_start,
                )
                chunk_stop = min(
                    intersect_stop,
                    int(math.ceil(float(np.nanmax(centers)) + support)) + 1 - file_global_start,
                )
                if chunk_stop <= chunk_start:
                    continue
                block_start = chunk_start - block_file_start
                block_stop = chunk_stop - block_file_start
                global_times = file_global_start + np.arange(chunk_start, chunk_stop, dtype=np.float64)
                tau_by_freq = tau_ref_bins * (freqs / ref_freq) ** (-TAU_FREQ_INDEX)
                pulse = scattered_gaussian_profile(global_times, centers, sigma_bins, tau_by_freq).astype(np.float32)
                model = truth.per_channel_peak_snr * envelope[:, None] * pulse
                accumulated[block_start:block_stop, start:end] += (
                    model.T * chan_std[start:end][None, :]
                ).astype(np.float32)
                has_signal = True

        if not has_signal:
            continue
        block_tpf = block.reshape(block_time, npol, nchan)
        for pol in (0, 1):
            base = block_tpf[:, pol, :].astype(np.float32)
            block_tpf[:, pol, :] = np.clip(np.rint(base + accumulated), 0, 255).astype(np.uint8)
        for rel_idx, row_idx in enumerate(range(row_start, row_stop)):
            table_data[row_idx]["DATA"] = block[rel_idx]

    return len(truth_windows)


def stokes_sum_from_data(data_field: np.ndarray, meta: dict) -> np.ndarray:
    view = data_5d_view(data_field, meta).reshape(
        meta["samples_per_file"],
        meta["npol"],
        meta["nchan"],
    )
    return view[:, 0, :].astype(np.uint16) + view[:, 1, :].astype(np.uint16)


def quantize_stokes_sum_to_2bit(stokes_sum: np.ndarray) -> np.ndarray:
    """Quantize Stokes-I sums to two bits using per-channel quartiles.

    ``stokes_sum`` is ``pol0 + pol1`` and therefore has integer range 0..510.
    Working in this integer domain avoids sorting a 2 GB float32 Stokes-I array
    for every FITS file.
    """
    n_time, nchan = stokes_sum.shape
    targets = np.array([n_time // 4, n_time // 2, (3 * n_time) // 4], dtype=np.int64)
    thresholds = np.empty((3, nchan), dtype=np.uint16)
    for chan in range(nchan):
        hist = np.bincount(stokes_sum[:, chan], minlength=511)
        cdf = np.cumsum(hist)
        thresholds[:, chan] = np.searchsorted(cdf, targets, side="right")
    levels = np.zeros(stokes_sum.shape, dtype=np.uint8)
    levels += stokes_sum > thresholds[0][None, :]
    levels += stokes_sum > thresholds[1][None, :]
    levels += stokes_sum > thresholds[2][None, :]
    return levels


def pack_2bit_rows(levels: np.ndarray, rows_per_file: int, nsblk: int, nchan: int) -> np.ndarray:
    flat = levels.reshape(rows_per_file, nsblk * nchan)
    bits = np.empty((rows_per_file, flat.shape[1], 2), dtype=np.uint8)
    bits[:, :, 0] = (flat >> 1) & 1
    bits[:, :, 1] = flat & 1
    return np.packbits(bits.reshape(rows_per_file, -1), axis=1)


def psrfits_2bit_data_view(packed_flat: np.ndarray, nsblk: int, nchan: int) -> np.ndarray:
    return packed_flat.reshape(packed_flat.shape[0], nsblk // 4, 1, nchan, 1)


def copy_table_header_cards(source_header: fits.Header, target_header: fits.Header) -> None:
    structural = {
        "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "PCOUNT", "GCOUNT",
        "TFIELDS", "CHECKSUM", "DATASUM",
    }
    column_prefixes = ("TTYPE", "TFORM", "TUNIT", "TDIM", "TNULL", "TSCAL", "TZERO", "TDISP", "TBCOL")
    for card in source_header.cards:
        key = card.keyword
        if key in ("", "COMMENT", "HISTORY"):
            continue
        if key in structural or any(key.startswith(prefix) for prefix in column_prefixes):
            continue
        if key not in target_header:
            target_header[key] = (card.value, card.comment)


def copied_column(col: fits.Column, data: np.ndarray) -> fits.Column:
    kwargs = {
        "name": col.name,
        "format": col.format,
        "array": data,
    }
    for attr in ("unit", "null", "bscale", "bzero", "disp", "start", "dim"):
        value = getattr(col, attr, None)
        if value is not None:
            kwargs[attr] = value
    return fits.Column(**kwargs)


def packed2_aux_column(col: fits.Column, data: np.ndarray, meta: dict) -> fits.Column:
    if col.name in ("DAT_OFFS", "DAT_SCL") and data.ndim == 2 and data.shape[1] == meta["npol"] * meta["nchan"]:
        data = data.reshape(meta["naxis2"], meta["npol"], meta["nchan"])[:, 0, :]
        kwargs = {
            "name": col.name,
            "format": f"{meta['nchan']}E",
            "array": data,
        }
        if col.unit is not None:
            kwargs["unit"] = col.unit
        return fits.Column(**kwargs)
    return copied_column(col, data)


def write_packed2_fits(template_path: Path, output_path: Path, data_field: np.ndarray, meta: dict) -> None:
    stokes_sum = stokes_sum_from_data(data_field, meta)
    levels = quantize_stokes_sum_to_2bit(stokes_sum)
    packed_flat = pack_2bit_rows(levels, meta["naxis2"], meta["nsblk"], meta["nchan"])
    packed = psrfits_2bit_data_view(packed_flat, meta["nsblk"], meta["nchan"])

    with fits.open(template_path, memmap=True) as source:
        primary = fits.PrimaryHDU(header=source[0].header.copy())
        columns = []
        for col in source[1].columns:
            if col.name == "DATA":
                columns.append(fits.Column(
                    name="DATA",
                    format=f"{packed_flat.shape[1]}B",
                    dim=f"(1,{meta['nchan']},1,{meta['nsblk'] // 4})",
                    array=packed,
                ))
            else:
                columns.append(packed2_aux_column(col, np.array(source[1].data[col.name], copy=True), meta))
        table = fits.BinTableHDU.from_columns(columns, name=source[1].name or "SUBINT")
        copy_table_header_cards(source[1].header, table.header)
        table.header["NPOL"] = 1
        table.header["NBITS"] = 2
        table.header["EXTNAME"] = "SUBINT"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([primary, table]).writeto(output_path, overwrite=True)


def convert_raw8_to_packed2_file(raw8_path: Path, packed2_path: Path) -> None:
    meta = read_metadata(raw8_path)
    with fits.open(raw8_path, memmap=True) as source:
        write_packed2_fits(raw8_path, packed2_path, source[1].data["DATA"], meta)


def write_manifests(output_dirs: list[Path], args: argparse.Namespace, files: list[Path], meta: dict,
                    truths: list[InjectionTruth]) -> None:
    payload = {
        "run_label": args.run_label,
        "background_dir": str(args.background_dir),
        "file_first": args.file_first,
        "file_last": args.file_last,
        "inject_file_first": args.inject_file_first,
        "inject_file_last": args.inject_file_last,
        "file_count": len(files),
        "count": args.count,
        "seed": args.seed,
        "std_samples_per_file": args.std_samples_per_file,
        "std_file_limit": args.std_file_limit,
        "workers": args.workers,
        "io_workers": args.io_workers,
        "metadata": meta,
        "parameter_ranges": {
            "dm_pc_cm3": DM_RANGE_PC_CM3,
            "snr": SNR_RANGE,
            "width_ms_fwhm": WIDTH_FWHM_MS_RANGE,
            "bandwidth_mhz_fwhm": BANDWIDTH_FWHM_MHZ_RANGE,
            "scattering_zero_fraction": SCATTERING_ZERO_FRACTION,
            "scattering_tau_1ghz_ms": SCATTERING_TAU_1GHZ_MS_RANGE,
        },
        "parameter_distributions": {
            "dm_pc_cm3": "Latin-hypercube stratified uniform",
            "snr": "Latin-hypercube log-uniform",
            "width_ms_fwhm": "Latin-hypercube log-uniform",
            "bandwidth_mhz_fwhm": "Latin-hypercube log-uniform",
            "scattering": "15% zero; otherwise tau_1GHz log-uniform, scaled as frequency^-4",
            "center_freq_mhz": "uniform over the observed band; finite bandwidth truncates out-of-band tails",
        },
    }
    for out_dir in output_dirs:
        quantization = "packed2" if out_dir.name.endswith("_packed2") else "raw8"
        quantized_payload = {**payload, "quantization": quantization}
        rows = [{**asdict(truth), "quantization": quantization} for truth in truths]
        write_json(out_dir / "run_config.json", quantized_payload)
        write_jsonl(out_dir / "truth_manifest.jsonl", rows)


def prepare_output_dirs(output_root: Path, run_label: str, overwrite: bool, make_packed2: bool) -> tuple[Path, Path | None]:
    raw_dir = output_root / f"{run_label}_raw8"
    packed_dir = output_root / f"{run_label}_packed2" if make_packed2 else None
    for out_dir in [raw_dir, packed_dir]:
        if out_dir is None:
            continue
        if out_dir.exists() and overwrite:
            shutil.rmtree(out_dir)
        if out_dir.exists():
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace it")
        out_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, packed_dir


def prepare_packed2_output_dir(output_root: Path, run_label: str, overwrite: bool) -> Path:
    packed_dir = output_root / f"{run_label}_packed2"
    if packed_dir.exists() and overwrite:
        shutil.rmtree(packed_dir)
    if packed_dir.exists():
        raise FileExistsError(f"{packed_dir} exists; pass --overwrite to replace it")
    packed_dir.mkdir(parents=True, exist_ok=True)
    return packed_dir


def copy_existing_manifests(raw8_dir: Path, packed2_dir: Path) -> None:
    for name in ("run_config.json", "truth_manifest.jsonl"):
        source = raw8_dir / name
        if source.exists():
            shutil.copyfile(source, packed2_dir / name)


def process_file_job_unlocked(job: dict) -> dict:
    """处理单个 FITS：复制为 raw8 并注入信号；若需要再写出对应的 packed2。"""
    idx = int(job["index"])
    total = int(job["total"])
    source_path = Path(job["source_path"])
    raw_dir = Path(job["raw_dir"])
    packed_dir = Path(job["packed_dir"]) if job["packed_dir"] else None
    meta = job["meta"]
    truths = job["truths"]
    chan_std = job["chan_std"]
    freqs = frequency_axis(meta)
    file_global_start = idx * meta["samples_per_file"]
    raw_path = raw_dir / source_path.name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        raw_path.unlink()
    shutil.copyfile(source_path, raw_path)
    with fits.open(raw_path, mode="update", memmap=True) as raw_hdul:
        if packed_dir is None:
            n_overlap = add_injections_to_table_rows(
                raw_hdul[1].data,
                file_global_start,
                truths,
                freqs,
                chan_std,
                meta,
            )
        else:
            data_field = np.array(raw_hdul[1].data["DATA"], copy=True)
            n_overlap, touched_windows = add_injections_to_data(
                data_field,
                file_global_start,
                truths,
                freqs,
                chan_std,
                meta,
            )
            if touched_windows:
                touched_rows = set()
                for start, stop in touched_windows:
                    row_start = max(0, start // meta["nsblk"])
                    row_stop = min(meta["naxis2"], (stop + meta["nsblk"] - 1) // meta["nsblk"])
                    touched_rows.update(range(row_start, row_stop))
                for row_idx in sorted(touched_rows):
                    raw_hdul[1].data[row_idx]["DATA"] = data_field[row_idx]
        raw_hdul[1].header["NBITS"] = 8
        raw_hdul.flush()
        if packed_dir is not None:
            packed_path = packed_dir / source_path.name
            write_packed2_fits(raw_path, packed_path, data_field, meta)
    return {
        "index": idx + 1,
        "total": total,
        "name": source_path.name,
        "overlapping_injections": n_overlap,
    }


def process_file_job(job: dict) -> dict:
    if IO_SEMAPHORE is None:
        return process_file_job_unlocked(job)
    IO_SEMAPHORE.acquire()
    try:
        return process_file_job_unlocked(job)
    finally:
        IO_SEMAPHORE.release()


def process_files(args: argparse.Namespace, files: list[Path], meta: dict, truths: list[InjectionTruth],
                  raw_dir: Path, packed_dir: Path | None, chan_std: np.ndarray) -> None:
    jobs = [
        {
            "index": idx,
            "total": len(files),
            "source_path": str(source_path),
            "raw_dir": str(raw_dir),
            "packed_dir": str(packed_dir) if packed_dir is not None else "",
            "meta": meta,
            "truths": truths,
            "chan_std": chan_std,
        }
        for idx, source_path in enumerate(files)
    ]
    if args.workers <= 1:
        for job in jobs:
            result = process_file_job(job)
            print(
                f"[write] {result['index']:03d}/{result['total']:03d} {result['name']} "
                f"overlapping_injections={result['overlapping_injections']}",
                flush=True,
            )
        return

    io_workers = min(args.io_workers, args.workers)
    if io_workers >= args.workers:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_file_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(
                    f"[write] {result['index']:03d}/{result['total']:03d} {result['name']} "
                    f"overlapping_injections={result['overlapping_injections']}",
                    flush=True,
                )
        return

    with multiprocessing.Manager() as manager:
        semaphore = manager.BoundedSemaphore(io_workers)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_process_worker,
            initargs=(semaphore,),
        ) as executor:
            futures = [executor.submit(process_file_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(
                    f"[write] {result['index']:03d}/{result['total']:03d} {result['name']} "
                    f"overlapping_injections={result['overlapping_injections']}",
                    flush=True,
                )


def process_packed2_conversion_job_unlocked(job: dict) -> dict:
    idx = int(job["index"])
    total = int(job["total"])
    raw8_path = Path(job["raw8_path"])
    packed_dir = Path(job["packed_dir"])
    packed_path = packed_dir / raw8_path.name
    convert_raw8_to_packed2_file(raw8_path, packed_path)
    return {"index": idx + 1, "total": total, "name": raw8_path.name}


def process_packed2_conversion_job(job: dict) -> dict:
    if IO_SEMAPHORE is None:
        return process_packed2_conversion_job_unlocked(job)
    IO_SEMAPHORE.acquire()
    try:
        return process_packed2_conversion_job_unlocked(job)
    finally:
        IO_SEMAPHORE.release()


def process_packed2_conversion_files(args: argparse.Namespace, raw8_files: list[Path], packed_dir: Path) -> None:
    jobs = [
        {
            "index": idx,
            "total": len(raw8_files),
            "raw8_path": str(raw8_path),
            "packed_dir": str(packed_dir),
        }
        for idx, raw8_path in enumerate(raw8_files)
    ]
    if args.workers <= 1:
        for job in jobs:
            result = process_packed2_conversion_job(job)
            print(f"[packed2] {result['index']:03d}/{result['total']:03d} {result['name']}", flush=True)
        return

    io_workers = min(args.io_workers, args.workers)
    if io_workers >= args.workers:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_packed2_conversion_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(f"[packed2] {result['index']:03d}/{result['total']:03d} {result['name']}", flush=True)
        return

    with multiprocessing.Manager() as manager:
        semaphore = manager.BoundedSemaphore(io_workers)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_process_worker,
            initargs=(semaphore,),
        ) as executor:
            futures = [executor.submit(process_packed2_conversion_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(f"[packed2] {result['index']:03d}/{result['total']:03d} {result['name']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject simulated FRBs into full-length FAST FITS files.")
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUND_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--file-first", type=int, default=DEFAULT_FILE_FIRST)
    parser.add_argument("--file-last", type=int, default=DEFAULT_FILE_LAST)
    parser.add_argument(
        "--inject-file-first",
        type=int,
        default=None,
        help="First FITS file number allowed for highest-frequency injection TOAs; defaults to --file-first.",
    )
    parser.add_argument(
        "--inject-file-last",
        type=int,
        default=None,
        help="Last FITS file number allowed for highest-frequency injection TOAs; defaults to --file-last.",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--std-samples-per-file", type=int, default=16384)
    parser.add_argument(
        "--std-file-limit",
        type=int,
        default=8,
        help="Number of selected FITS files used to estimate channel noise; 0 means all files.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel FITS files to process.")
    parser.add_argument(
        "--io-workers",
        type=int,
        default=8,
        help="Maximum workers allowed to copy/write FITS files at the same time.",
    )
    parser.add_argument("--no-packed2", action="store_true", help="Only write raw8 FITS files.")
    parser.add_argument("--packed2-only", action="store_true", help="Convert existing raw8 FITS files to packed2 without injecting.")
    parser.add_argument("--raw8-dir", type=Path, default=None, help="Input raw8 directory for --packed2-only.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and sample truth rows without writing FITS.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inject_file_first is None:
        args.inject_file_first = args.file_first
    if args.inject_file_last is None:
        args.inject_file_last = args.file_last
    if args.file_first > args.file_last:
        raise SystemExit("--file-first must be <= --file-last")
    if args.inject_file_first > args.inject_file_last:
        raise SystemExit("--inject-file-first must be <= --inject-file-last")
    if not (args.file_first <= args.inject_file_first <= args.inject_file_last <= args.file_last):
        raise SystemExit(
            "--inject-file-first/--inject-file-last must be inside --file-first/--file-last; "
            f"got search={args.file_first}-{args.file_last} "
            f"inject={args.inject_file_first}-{args.inject_file_last}"
        )
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.io_workers < 1:
        raise SystemExit("--io-workers must be >= 1")
    if args.packed2_only:
        if args.no_packed2:
            raise SystemExit("--packed2-only cannot be combined with --no-packed2")
        raw8_dir = args.raw8_dir or (args.output_root / f"{args.run_label}_raw8")
        raw8_files = list_background_fits(raw8_dir, args.file_first, args.file_last)
        meta = read_metadata(raw8_files[0])
        validate_consistent_headers(raw8_files, meta)
        if meta["nbits"] != 8 or meta["npol"] < 2:
            raise SystemExit(f"--packed2-only expects raw8 files with at least two polarizations, got {meta}")
        if args.dry_run:
            print(
                f"[dry-run] packed2-only files={len(raw8_files)} input={raw8_dir} "
                f"output={args.output_root / f'{args.run_label}_packed2'}",
                flush=True,
            )
            return
        packed_dir = prepare_packed2_output_dir(args.output_root, args.run_label, args.overwrite)
        copy_existing_manifests(raw8_dir, packed_dir)
        process_packed2_conversion_files(args, raw8_files, packed_dir)
        print(f"[done] packed2={packed_dir}", flush=True)
        return

    files = list_background_fits(args.background_dir, args.file_first, args.file_last)
    meta = read_metadata(files[0])
    validate_consistent_headers(files, meta)
    truths = build_truth_rows(args, files, meta)
    total_s = len(files) * meta["duration_per_file_s"]
    print(
        f"[data] files={len(files)} duration={total_s:.6f}s ({total_s / 60.0:.6f} min) "
        f"samples_per_file={meta['samples_per_file']} workers={args.workers} io_workers={args.io_workers} "
        f"inject_files={args.inject_file_first}-{args.inject_file_last}",
        flush=True,
    )
    print(f"[truth] sampled_injections={len(truths)} seed={args.seed}", flush=True)
    if args.dry_run:
        preview = [asdict(truth) for truth in truths[:5]]
        print(json.dumps({"preview": preview}, indent=2, sort_keys=True), flush=True)
        return

    raw_dir, packed_dir = prepare_output_dirs(
        args.output_root,
        args.run_label,
        overwrite=args.overwrite,
        make_packed2=not args.no_packed2,
    )
    output_dirs = [raw_dir] + ([packed_dir] if packed_dir is not None else [])
    write_manifests(output_dirs, args, files, meta, truths)
    std_files = files if args.std_file_limit <= 0 else files[:min(args.std_file_limit, len(files))]
    std_workers = min(args.workers, len(std_files))
    print(
        f"[std] using_files={len(std_files)}/{len(files)} std_workers={std_workers} "
        f"samples_per_file={args.std_samples_per_file}",
        flush=True,
    )
    chan_std = estimate_global_channel_std(std_files, meta, args.std_samples_per_file, std_workers)
    process_files(args, files, meta, truths, raw_dir, packed_dir, chan_std)
    print(f"[done] raw8={raw_dir}", flush=True)
    if packed_dir is not None:
        print(f"[done] packed2={packed_dir}", flush=True)


if __name__ == "__main__":
    main()
