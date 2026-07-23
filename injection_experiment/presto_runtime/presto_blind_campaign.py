#!/usr/bin/env python3
"""在注入数据上运行 PRESTO 盲搜 single-pulse 基线。

This script follows the conventional PRESTO single-burst command pattern:

1. ``rfifind <context files...> -o <prefix> -time 1``
2. ``prepsubband -nobary -numout ... -nsub ... -lodm ... -dmstep ...
   -numdms ... -downsamp ... -mask <prefix>_rfifind.mask -o <prefix> <context files...>``
3. ``ls *.dat | xargs -n ... -P ... python single_pulse_search.py -b -m ... -t ...``

The default mode is a blind search over ``--dm-min`` to ``--dm-max`` split into
``--dm-block-size`` PRESTO blocks, matching the documented ``prepsubband`` +
``single_pulse_search.py`` workflow.  Each searched FITS is treated as the
center of a short multi-file window so high-DM dispersion delay can draw on
neighboring segments; only candidates whose de-dispersed sample falls in the
center file are retained for scoring.

中文维护说明：默认模式必须是盲搜 DM 范围，用于最终 PRESTO recall/precision
对照；已知 DM 窗口模式只保留作快速调试，不能作为论文性能统计来源。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from presto_common import (
    QUANTIZATIONS,
    SNR_BINS,
    bin_index,
    candidate_key,
    event_ddm,
    event_dt_ms,
    event_matches_truth,
    nearest_truth_with_distance,
    param_specs_for_truth,
    plot_parameter_cells,
    plot_snr_rows,
    snr_rows_from_matches,
    truth_key,
    write_csv,
)


DEFAULT_RUN_LABEL = "v10_det03_injection_10000"
DEFAULT_BASE_ROOT = Path(os.environ.get("INJECTION_EXPERIMENT_ROOT", "/path/to/drafts_runs/injection_experiment"))
DEFAULT_SIM_ROOT = DEFAULT_BASE_ROOT / "simdata"
DEFAULT_TRUTH_ROOT = DEFAULT_BASE_ROOT / "truth_archive"
DEFAULT_OUTPUT_BASE = Path("/path/to/drafts_runs/injection_experiment/presto_runtime")
DEFAULT_SCRATCH_ROOT = Path(os.environ.get("PRESTO_SCRATCH_ROOT", str(DEFAULT_OUTPUT_BASE / "scratch")))


@dataclass(frozen=True)
class Job:
    quantization: str
    batch: int
    file_stem: str
    fits_path: str
    context_fits_paths: tuple[str, ...]
    center_context_index: int
    center_start_sample: int
    center_num_samples: int
    truth_rows: tuple[dict, ...]
    gpu_id: int


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_batch(text: str) -> int | None:
    match = re.search(r"_b(\d{2})", text)
    return int(match.group(1)) if match else None


def truth_paths(truth_root: Path, batches: list[int], quantizations: list[str]) -> list[Path]:
    paths: list[Path] = []
    for batch in batches:
        for quant in quantizations:
            pattern = f"*_b{batch:02d}_{quant}/truth_manifest.jsonl"
            paths.extend(sorted(truth_root.glob(pattern)))
    return paths


def load_truth_archive(truth_root: Path, batches: list[int], quantizations: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in truth_paths(truth_root, batches, quantizations):
        batch = parse_batch(str(path))
        quant = "packed2" if "packed2" in str(path) else "raw8"
        for row in read_jsonl(path):
            row = dict(row)
            row["batch"] = int(batch if batch is not None else row.get("batch", -1))
            row["quantization"] = quant
            row["truth_manifest"] = str(path)
            rows.append(row)
    return rows


def build_jobs(args: argparse.Namespace, truth: list[dict]) -> list[Job]:
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in truth:
        stem = str(row["output_file_stem"])
        key = (str(row["quantization"]), int(row["batch"]), stem)
        groups.setdefault(key, []).append(row)

    gpu_ids = args.gpu_ids
    jobs: list[Job] = []
    job_items: list[tuple[str, int, int, Path, list[dict], tuple[Path, ...], int]] = []
    for batch in args.batches:
        for quant in args.quantizations:
            sim_dir = args.sim_root / f"{args.run_label}_b{batch:02d}_{quant}"
            fits_files = sorted(sim_dir.glob("*.fits"))
            fits_by_stem = {path.stem: (index, path) for index, path in enumerate(fits_files)}
            if args.search_all_files:
                stems = [path.stem for path in fits_files]
            else:
                stems = sorted(stem for q, b, stem in groups if q == quant and b == batch)
            for stem in stems:
                item = fits_by_stem.get(stem)
                if item is None:
                    print(f"[WARN] missing FITS: {sim_dir / f'{stem}.fits'}", flush=True)
                    continue
                file_index, fits_path = item
                context_start = max(0, file_index - args.context_left_files)
                context_end = min(len(fits_files), file_index + args.context_right_files + 1)
                context_paths = tuple(fits_files[context_start:context_end])
                center_context_index = file_index - context_start
                key = (quant, batch, stem)
                job_items.append((quant, batch, file_index, fits_path, groups.get(key, []), context_paths, center_context_index))

    for index, (quant, batch, _file_index, fits_path, rows, context_paths, center_context_index) in enumerate(job_items):
        if not context_paths:
            continue
        jobs.append(
            Job(
                quantization=quant,
                batch=batch,
                file_stem=fits_path.stem,
                fits_path=str(fits_path),
                context_fits_paths=tuple(str(path) for path in context_paths),
                center_context_index=center_context_index,
                center_start_sample=center_context_index * args.numout,
                center_num_samples=args.numout,
                truth_rows=tuple(rows),
                gpu_id=gpu_ids[index % len(gpu_ids)],
            )
        )
    if args.limit_jobs is not None:
        jobs = jobs[: args.limit_jobs]
    return jobs


def dm_intervals(rows: Iterable[dict], args: argparse.Namespace) -> list[tuple[float, int]]:
    if args.mode == "blind":
        intervals: list[tuple[float, int]] = []
        lodm = args.dm_min
        epsilon = args.dm_step * 1e-6
        while lodm <= args.dm_max + epsilon:
            remaining = int(math.floor((args.dm_max - lodm) / args.dm_step + epsilon)) + 1
            numdms = max(1, min(args.dm_block_size, remaining))
            intervals.append((lodm, numdms))
            lodm += numdms * args.dm_step
        return intervals

    raw: list[tuple[float, float]] = []
    for row in rows:
        dm = float(row["dm_pc_cm3"])
        lo = max(args.dm_min, math.floor((dm - args.dm_window) / args.dm_step) * args.dm_step)
        hi = min(args.dm_max, math.ceil((dm + args.dm_window) / args.dm_step) * args.dm_step)
        raw.append((lo, hi))
    raw.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in raw:
        if not merged or lo > merged[-1][1] + args.dm_step:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    intervals: list[tuple[float, int]] = []
    for lo, hi in merged:
        numdms = int(round((hi - lo) / args.dm_step)) + 1
        intervals.append((lo, max(1, numdms)))
    return intervals


def run_command(
    cmd: list[str] | str,
    cwd: Path,
    log_path: Path,
    *,
    shell: bool = False,
    env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=shell,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        log.write(f"[exit={proc.returncode} elapsed={time.time() - started:.3f}s]\n")
    return int(proc.returncode)


def remove_prepsubband_outputs(work_dir: Path) -> None:
    for suffix in (".dat", ".inf", ".singlepulse", ".ps"):
        for path in work_dir.glob(f"*{suffix}"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def rewrite_cropped_inf(path: Path, sample_offset: int, num_samples: int, dt_seconds: float) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith(" Number of bins in the time series"):
            out.append(f" Number of bins in the time series      =  {num_samples:<12d}")
        elif line.startswith(" Epoch of observation (MJD)"):
            try:
                mjd = float(line.split("=", 1)[1].strip())
                mjd += sample_offset * dt_seconds / 86400.0
                out.append(f" Epoch of observation (MJD)             =  {mjd:.17f}")
            except (IndexError, ValueError):
                out.append(line)
        elif line.startswith(" Any breaks in the data?"):
            out.append(" Any breaks in the data? (1 yes, 0 no)  =  0")
        elif line.startswith(" On/Off bin pair #"):
            continue
        else:
            out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def crop_prepsubband_outputs(work_dir: Path, job: Job, dt_seconds: float, log_path: Path) -> None:
    sample_offset = int(job.center_start_sample)
    sample_end = sample_offset + int(job.center_num_samples)
    cropped = 0
    skipped = 0
    for dat_path in sorted(work_dir.glob("*.dat")):
        data = np.fromfile(dat_path, dtype=np.float32)
        if data.size < sample_end:
            skipped += 1
            continue
        data[sample_offset:sample_end].copy().tofile(dat_path)
        inf_path = dat_path.with_suffix(".inf")
        if inf_path.exists():
            rewrite_cropped_inf(inf_path, sample_offset, int(job.center_num_samples), dt_seconds)
        cropped += 1
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(
            f"[center-crop] cropped={cropped} skipped={skipped} "
            f"sample_offset={sample_offset} num_samples={job.center_num_samples}\n"
        )


def write_valid_dat_list(work_dir: Path, args: argparse.Namespace, log_path: Path) -> Path:
    valid_path = work_dir / "valid_dat_files.txt"
    invalid_path = work_dir / "invalid_dat_files.txt"
    valid_count = 0
    invalid_count = 0
    with valid_path.open("w", encoding="utf-8") as valid, invalid_path.open("w", encoding="utf-8") as invalid:
        for path in sorted(work_dir.glob("*.dat")):
            try:
                data = np.fromfile(path, dtype=np.float32)
                std = float(np.nanstd(data)) if data.size else 0.0
                finite = bool(np.isfinite(data).all()) if data.size else False
            except Exception as exc:  # noqa: BLE001 - diagnostics only.
                invalid.write(f"{path.name}\tread_error\t{exc}\n")
                invalid_count += 1
                continue
            if finite and std > args.min_dat_std:
                valid.write(f"{path.name}\n")
                valid_count += 1
            else:
                invalid.write(f"{path.name}\tstd={std:.8g}\tfinite={finite}\n")
                invalid_count += 1
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"[dat-filter] valid={valid_count} invalid={invalid_count} min_std={args.min_dat_std}\n")
    return valid_path


def parse_singlepulse_file(path: Path, job: Job, dt_seconds: float, sample_offset: int) -> list[dict]:
    rows: list[dict] = []
    center_lo = int(sample_offset)
    center_hi = center_lo + int(job.center_num_samples)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            try:
                dm = float(parts[0])
                sigma = float(parts[1])
                time_s = float(parts[2])
                sample = int(float(parts[3]))
                downfact = int(float(parts[4]))
            except ValueError:
                continue
            if sample < center_lo or sample >= center_hi:
                continue
            file_sample = sample - center_lo
            rows.append(
                {
                    "quantization": job.quantization,
                    "batch": job.batch,
                    "file_stem": job.file_stem,
                    "source_file": job.fits_path,
                    "dm_pc_cm3": dm,
                    "sigma": sigma,
                    "time_s": file_sample * dt_seconds,
                    "sample": file_sample,
                    "downfact": downfact,
                    "time_ms": file_sample * dt_seconds * 1000.0,
                    "context_sample": sample,
                    "context_time_s": time_s,
                    "center_start_sample": center_lo,
                    "context_fits_count": len(job.context_fits_paths),
                    "dt_seconds": dt_seconds,
                    "singlepulse_file": str(path),
                }
            )
    return rows


def run_job(job: Job, args_dict: dict) -> dict:
    args = argparse.Namespace(**args_dict)
    candidate_path = args.output_root / "candidates" / job.quantization / f"b{job.batch:02d}" / f"{job.file_stem}.jsonl"
    event_path = args.output_root / "events" / job.quantization / f"b{job.batch:02d}" / f"{job.file_stem}.jsonl"
    status_path = args.output_root / "status" / job.quantization / f"b{job.batch:02d}" / f"{job.file_stem}.json"
    if event_path.exists() and status_path.exists() and not args.overwrite:
        return {
            "status": "skipped",
            "quantization": job.quantization,
            "batch": job.batch,
            "file_stem": job.file_stem,
            "candidate_path": str(candidate_path),
            "event_path": str(event_path),
        }

    file_token = job.file_stem.rsplit("_", 1)[-1]
    quant_token = {"raw8": "r8", "packed2": "p2"}.get(job.quantization, job.quantization[:4])
    short_job_name = f"{quant_token}b{job.batch:02d}_{file_token}"
    work_dir = args.scratch_root / args.output_root.name / short_job_name
    log_path = args.output_root / "logs" / job.quantization / f"b{job.batch:02d}" / f"{job.file_stem}.log"
    if args.overwrite and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    dt_seconds = float(job.truth_rows[0].get("time_reso_seconds", 4.9152e-05)) if job.truth_rows else 4.9152e-05
    started = time.time()
    status = {
        "status": "running",
        "quantization": job.quantization,
        "batch": job.batch,
        "file_stem": job.file_stem,
        "fits_path": job.fits_path,
        "context_fits_paths": list(job.context_fits_paths),
        "context_fits_count": len(job.context_fits_paths),
        "center_context_index": job.center_context_index,
        "center_start_sample": job.center_start_sample,
        "center_num_samples": job.center_num_samples,
        "prepsubband_numout": job.center_num_samples * len(job.context_fits_paths),
        "crop_dat_to_center": bool(args.crop_dat_to_center),
        "gpu_id": job.gpu_id,
        "truth_count": len(job.truth_rows),
        "candidate_path": str(candidate_path),
        "event_path": str(event_path),
        "work_dir": str(work_dir),
        "started_unix": started,
    }

    env = os.environ.copy()
    # PRESTO_GPU 在多进程同时跑时，对物理 GPU id 的处理不够稳。
    # 每个 job 只暴露一张物理卡，再让 prepsubband 在这个局部环境里使用 cuda 0。
    cuda_visible_devices = str(job.gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("OMP_NUM_THREADS", "1")
    presto_cuda_id = "0"
    status["cuda_visible_devices"] = cuda_visible_devices
    status["presto_cuda_id"] = presto_cuda_id
    prefix = work_dir / job.file_stem
    mask_path = work_dir / f"{job.file_stem}_rfifind.mask"

    try:
        if not args.skip_rfifind or not mask_path.exists():
            rc = run_command(
                ["rfifind", *job.context_fits_paths, "-o", str(prefix), "-time", str(args.rfifind_time)],
                work_dir,
                log_path,
                env=env,
            )
            if rc != 0 or not mask_path.exists():
                raise RuntimeError(f"rfifind failed rc={rc}")

        intervals = dm_intervals(job.truth_rows, args)
        status["dm_intervals"] = [{"lodm": lo, "numdms": ndm} for lo, ndm in intervals]
        if not intervals:
            if args.keep_candidates:
                write_jsonl(candidate_path, [])
            write_jsonl(event_path, [])
            status.update(
                {
                    "status": "done",
                    "candidate_count": 0,
                    "event_count": 0,
                    "finished_unix": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "note": "No DM intervals for this job.",
                }
            )
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return status

        for interval_index, (lodm, numdms) in enumerate(intervals):
            block_prefix = work_dir / f"{job.file_stem}_dm{lodm:.2f}_n{numdms:04d}"
            cmd = [
                "prepsubband",
                "-cuda",
                presto_cuda_id,
            ]
            if args.noclip:
                cmd.append("-noclip")
            cmd.extend(
                [
                    "-nobary",
                    "-numout",
                    str(job.center_num_samples * len(job.context_fits_paths)),
                    "-nsub",
                    str(args.nsub),
                    "-lodm",
                    f"{lodm:.6g}",
                    "-dmstep",
                    f"{args.dm_step:.6g}",
                    "-numdms",
                    str(numdms),
                    "-downsamp",
                    str(args.downsamp),
                    "-mask",
                    str(mask_path),
                    "-o",
                    str(block_prefix),
                    *job.context_fits_paths,
                ]
            )
            rc = run_command(cmd, work_dir, log_path, env=env)
            if rc != 0 and args.retry_nsub and args.retry_nsub > args.nsub:
                retry_cmd = [item for item in cmd]
                retry_cmd[retry_cmd.index("-nsub") + 1] = str(args.retry_nsub)
                rc = run_command(retry_cmd, work_dir, log_path, env=env)
            if rc != 0:
                raise RuntimeError(f"prepsubband interval {interval_index} failed rc={rc}")

        sample_offset = 0
        if args.crop_dat_to_center:
            crop_prepsubband_outputs(work_dir, job, dt_seconds, log_path)
        else:
            sample_offset = int(job.center_start_sample)

        valid_dat_path = write_valid_dat_list(work_dir, args, log_path)
        single_pulse = shutil.which("single_pulse_search.py") or "single_pulse_search.py"
        xargs_cmd = (
            f"cat {valid_dat_path.name} | "
            f"xargs -r -n {args.xargs_chunk} -P {args.xargs_procs} "
            f"python {single_pulse} -b -m {args.maxwidth} -t {args.threshold} -p"
        )
        rc = run_command(xargs_cmd, work_dir, log_path, shell=True, env=env)
        if rc != 0:
            fallback_cmd = (
                f"cat {valid_dat_path.name} | "
                f"xargs -r -n {args.xargs_chunk} -P {args.xargs_procs} sh -c "
            f"'for dat in \"$@\"; do "
            f"python {single_pulse} -b -m {args.maxwidth} -t {args.threshold} -p \"$dat\" "
            f"|| echo \"[WARN] single_pulse_search failed for $dat\" >&2; "
            f"done' sh"
            )
            rc = run_command(fallback_cmd, work_dir, log_path, shell=True, env=env)
        if rc != 0:
            raise RuntimeError(f"single_pulse_search failed rc={rc}")

        candidates: list[dict] = []
        events: list[dict] = []
        event_buckets: dict[tuple[int, int], list[dict]] = {}
        next_event_id = 0
        candidate_count = 0
        key = (job.quantization, job.batch, job.file_stem)
        for path in sorted(work_dir.glob("*.singlepulse")):
            rows = parse_singlepulse_file(path, job, dt_seconds, sample_offset)
            candidate_count += len(rows)
            if args.keep_candidates:
                candidates.extend(rows)
            for row in rows:
                next_event_id = add_candidate_to_event_index(row, key, events, event_buckets, next_event_id, args)
        if args.keep_candidates:
            write_jsonl(candidate_path, candidates)
        write_jsonl(event_path, events)
        status.update(
            {
                "status": "done",
                "candidate_count": candidate_count,
                "event_count": len(events),
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            }
        )
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.keep_dat:
            remove_prepsubband_outputs(work_dir)
        return status
    except Exception as exc:  # noqa: BLE001 - worker must report failures.
        status.update({"status": "failed", "error": str(exc), "finished_unix": time.time(), "elapsed_seconds": time.time() - started})
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return status


def load_candidates(candidate_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(candidate_root.glob("*/*/*.jsonl")):
        for row in read_jsonl(path):
            rows.append(row)
    return rows


def load_events(event_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(event_root.glob("*/*/*.jsonl")):
        for row in read_jsonl(path):
            rows.append(row)
    return rows


def load_statuses(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((output_root / "status").glob("*/*/*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rows


def event_from_candidate(row: dict, event_id: str) -> dict:
    return {
        "event_id": event_id,
        "quantization": str(row["quantization"]),
        "batch": int(row["batch"]),
        "file_stem": str(row["file_stem"]),
        "dm_pc_cm3": float(row["dm_pc_cm3"]),
        "sigma": float(row["sigma"]),
        "time_s": float(row["time_s"]),
        "time_ms": float(row["time_ms"]),
        "sample": int(row["sample"]),
        "downfact": int(row["downfact"]),
        "dt_seconds": float(row.get("dt_seconds", 4.9152e-05)),
        "context_sample": int(row.get("context_sample", row["sample"])),
        "center_start_sample": int(row.get("center_start_sample", 0)),
        "context_fits_count": int(row.get("context_fits_count", 1)),
        "event_size": 1,
        "source_file": row.get("source_file", ""),
    }


def update_event_if_stronger(event: dict, row: dict) -> None:
    event["event_size"] += 1
    if float(row["sigma"]) <= float(event["sigma"]):
        return
    event.update(
        {
            "dm_pc_cm3": float(row["dm_pc_cm3"]),
            "sigma": float(row["sigma"]),
            "time_s": float(row["time_s"]),
            "time_ms": float(row["time_ms"]),
            "sample": int(row["sample"]),
            "downfact": int(row["downfact"]),
            "dt_seconds": float(row.get("dt_seconds", 4.9152e-05)),
            "context_sample": int(row.get("context_sample", row["sample"])),
            "center_start_sample": int(row.get("center_start_sample", 0)),
            "context_fits_count": int(row.get("context_fits_count", 1)),
            "source_file": row.get("source_file", ""),
        }
    )


def add_candidate_to_event_index(
    row: dict,
    key: tuple[str, int, str],
    events: list[dict],
    buckets: dict[tuple[int, int], list[dict]],
    next_event_id: int,
    args: argparse.Namespace,
) -> int:
    dm_tol = max(float(args.event_dedup_dm_tolerance), 1e-9)
    time_tol = max(float(args.event_dedup_time_tolerance_ms), 1e-9)
    dm = float(row["dm_pc_cm3"])
    time_ms = float(row["time_ms"])
    dm_bucket = math.floor(dm / dm_tol)
    time_bucket = math.floor(time_ms / time_tol)
    for ii in range(dm_bucket - 1, dm_bucket + 2):
        for jj in range(time_bucket - 1, time_bucket + 2):
            for event in buckets.get((ii, jj), []):
                if abs(dm - float(event["dm_pc_cm3"])) <= dm_tol and abs(time_ms - float(event["time_ms"])) <= time_tol:
                    update_event_if_stronger(event, row)
                    return next_event_id

    next_event_id += 1
    event_id = f"{key[0]}_b{key[1]:02d}_{key[2]}_evt{next_event_id:05d}"
    event = event_from_candidate(row, event_id)
    events.append(event)
    buckets.setdefault((dm_bucket, time_bucket), []).append(event)
    return next_event_id


def cluster_events(candidates: list[dict], args: argparse.Namespace) -> list[dict]:
    groups: dict[tuple[str, int, str], tuple[list[dict], dict[tuple[int, int], list[dict]], int]] = {}
    for row in candidates:
        key = candidate_key(row)
        events, buckets, next_event_id = groups.setdefault(key, ([], {}, 0))
        next_event_id = add_candidate_to_event_index(row, key, events, buckets, next_event_id, args)
        groups[key] = (events, buckets, next_event_id)

    events: list[dict] = []
    for key in sorted(groups):
        events.extend(groups[key][0])
    return events


def analyze(
    truth: list[dict],
    events: list[dict],
    args: argparse.Namespace,
    processed_keys: set[tuple[str, int, str]] | None = None,
) -> tuple[list[dict], list[dict]]:
    if processed_keys is None:
        processed_keys = {candidate_key(row) for row in events}
    truth = [row for row in truth if truth_key(row) in processed_keys]

    truths_by_key: dict[tuple[str, int, str], list[dict]] = {}
    for row in truth:
        truths_by_key.setdefault(truth_key(row), []).append(row)
    events_by_key: dict[tuple[str, int, str], list[dict]] = {}
    for event in events:
        events_by_key.setdefault(candidate_key(event), []).append(event)

    used_events: set[str] = set()
    matches: list[dict] = []
    for row in sorted(truth, key=lambda item: (str(item["quantization"]), int(item["batch"]), str(item["output_file_stem"]), str(item["injection_id"]))):
        key = truth_key(row)
        best_event = None
        best_score = float("inf")
        for event in events_by_key.get(key, []):
            if event["event_id"] in used_events:
                continue
            if not event_matches_truth(event, row, args.source_dm_tolerance, args.source_time_tolerance_ms):
                continue
            dt_ms = event_dt_ms(event, row)
            ddm = event_ddm(event, row)
            score = abs(dt_ms) / max(args.source_time_tolerance_ms, 1e-6) + abs(ddm) / max(args.source_dm_tolerance, 1e-6) - 0.001 * float(event["sigma"])
            if score < best_score:
                best_score = score
                best_event = event
        detected = best_event is not None
        if detected:
            used_events.add(best_event["event_id"])
            dm_error = float(best_event["dm_pc_cm3"]) - float(row["dm_pc_cm3"])
            toa_error_ms = (float(best_event["sample"]) - float(row["highest_freq_toa_file_raw_sample"])) * float(row.get("time_reso_seconds", best_event.get("dt_seconds", 4.9152e-05))) * 1000.0
            localized = abs(dm_error) <= args.localize_dm_tolerance and abs(toa_error_ms) <= args.localize_time_tolerance_ms
        else:
            dm_error = ""
            toa_error_ms = ""
            localized = False

        out = dict(row)
        out.update(
            {
                "detected": bool(detected),
                "localized_within_tolerance": bool(localized),
                "matched_event_id": best_event["event_id"] if best_event else "",
                "pred_dm_pc_cm3": best_event["dm_pc_cm3"] if best_event else "",
                "pred_toa_file_raw_sample": best_event["sample"] if best_event else "",
                "dm_error_pc_cm3": dm_error,
                "toa_error_ms": toa_error_ms,
                "classifier_score": best_event["sigma"] if best_event else "",
                "detector_score": "",
                "event_size": best_event["event_size"] if best_event else "",
                "presto_downfact": best_event["downfact"] if best_event else "",
            }
        )
        matches.append(out)

    false_positives: list[dict] = []
    for event in events:
        if event["event_id"] in used_events:
            continue
        local_truth = truths_by_key.get(candidate_key(event), [])
        if any(event_matches_truth(event, row, args.source_dm_tolerance, args.source_time_tolerance_ms) for row in local_truth):
            continue
        assigned, _ = nearest_truth_with_distance(event, local_truth) if local_truth else (None, float("inf"))
        fp = dict(event)
        fp.update(
            {
                "assigned_quantization": event["quantization"],
                "assigned_injection_id": assigned.get("injection_id", "") if assigned else "",
                "assigned_snr": assigned.get("snr", "") if assigned else "",
                "assigned_dm_pc_cm3": assigned.get("dm_pc_cm3", "") if assigned else "",
                "assigned_width_ms_fwhm": assigned.get("width_ms_fwhm", "") if assigned else "",
                "assigned_bandwidth_mhz_fwhm": assigned.get("bandwidth_mhz_fwhm", "") if assigned else "",
                "assigned_scattering_ms_at_1ghz": assigned.get("scattering_ms_at_1ghz", "") if assigned else "",
                "assigned_center_freq_mhz": assigned.get("center_freq_mhz", "") if assigned else "",
                "assigned_per_channel_peak_snr": assigned.get("per_channel_peak_snr", "") if assigned else "",
                "assigned_toa_delta_ms": (
                    (float(event["sample"]) - float(assigned["highest_freq_toa_file_raw_sample"])) * float(assigned.get("time_reso_seconds", event.get("dt_seconds", 4.9152e-05))) * 1000.0
                    if assigned
                    else ""
                ),
                "assigned_dm_delta_pc_cm3": (float(event["dm_pc_cm3"]) - float(assigned["dm_pc_cm3"]) if assigned else ""),
                "classifier_score": event["sigma"],
                "detector_score": "",
            }
        )
        false_positives.append(fp)

    return matches, false_positives


def cell_table(matches: list[dict], false_positives: list[dict], parameter: str, bins: list[float]) -> list[dict]:
    rows: list[dict] = []
    for quant in QUANTIZATIONS:
        q_matches = [row for row in matches if row.get("quantization") == quant]
        q_fp = [row for row in false_positives if row.get("assigned_quantization") == quant]
        for y, (param_lo, param_hi) in enumerate(zip(bins[:-1], bins[1:])):
            for x, (snr_lo, snr_hi) in enumerate(zip(SNR_BINS[:-1], SNR_BINS[1:])):
                subset = [
                    row
                    for row in q_matches
                    if bin_index(float(row["snr"]), SNR_BINS) == x
                    and bin_index(float(row[parameter]), bins) == y
                ]
                fp_subset = [
                    row
                    for row in q_fp
                    if row.get("assigned_snr") not in ("", None)
                    and row.get(f"assigned_{parameter}") not in ("", None)
                    and bin_index(float(row["assigned_snr"]), SNR_BINS) == x
                    and bin_index(float(row[f"assigned_{parameter}"]), bins) == y
                ]
                total = len(subset)
                detected = sum(1 for row in subset if bool(row.get("detected")))
                fp_count = len(fp_subset)
                rows.append(
                    {
                        "quantization": quant,
                        "parameter": parameter,
                        "param_bin": y,
                        "snr_bin": x,
                        "param_low": param_lo,
                        "param_high": param_hi,
                        "snr_low": snr_lo,
                        "snr_high": snr_hi,
                        "n_injected": total,
                        "n_detected": detected,
                        "n_false_positive_assigned": fp_count,
                        "recall": detected / total if total else "",
                        "precision_proxy": detected / (detected + fp_count) if (detected + fp_count) else "",
                    }
                )
    return rows


def summarize(matches: list[dict], false_positives: list[dict]) -> list[dict]:
    rows = []
    for quant in QUANTIZATIONS:
        subset = [row for row in matches if row.get("quantization") == quant]
        fp = [row for row in false_positives if row.get("assigned_quantization") == quant]
        total = len(subset)
        detected = sum(1 for row in subset if bool(row.get("detected")))
        localized = sum(1 for row in subset if bool(row.get("localized_within_tolerance")))
        rows.append(
            {
                "quantization": quant,
                "truth_count": total,
                "detected_count": detected,
                "localized_count": localized,
                "false_positive_count": len(fp),
                "recall": detected / total if total else "",
                "localized_fraction": localized / total if total else "",
                "precision_proxy": detected / (detected + len(fp)) if detected + len(fp) else "",
            }
        )
    return rows


def aggregate_and_plot(args: argparse.Namespace, truth: list[dict]) -> None:
    events = load_events(args.output_root / "events")
    if not events and (args.output_root / "candidates").exists():
        events = cluster_events(load_candidates(args.output_root / "candidates"), args)
    statuses = load_statuses(args.output_root)
    processed_keys = {
        (str(row["quantization"]), int(row["batch"]), str(row["file_stem"]))
        for row in statuses
        if row.get("status") in {"done", "skipped"}
    }
    if not processed_keys:
        processed_keys = {candidate_key(row) for row in events}
    matches, false_positives = analyze(truth, events, args, processed_keys)
    aggregate = args.output_root / "aggregate"
    analysis = args.output_root / "analysis"
    figures = args.output_root / "publication_figures"

    match_fields = sorted({key for row in matches for key in row.keys()})
    fp_fields = sorted({key for row in false_positives for key in row.keys()})
    event_fields = sorted({key for row in events for key in row.keys()})
    write_csv(aggregate / "all_matches.csv", matches, match_fields)
    write_csv(aggregate / "all_false_positives.csv", false_positives, fp_fields or ["event_id"])
    write_csv(aggregate / "all_events.csv", events, event_fields or ["event_id"])

    summary = summarize(matches, false_positives)
    write_csv(aggregate / "summary.csv", summary, ["quantization", "truth_count", "detected_count", "localized_count", "false_positive_count", "recall", "localized_fraction", "precision_proxy"])

    specs = param_specs_for_truth(matches)
    cell_fields = [
        "quantization",
        "parameter",
        "param_bin",
        "snr_bin",
        "param_low",
        "param_high",
        "snr_low",
        "snr_high",
        "n_injected",
        "n_detected",
        "n_false_positive_assigned",
        "recall",
        "precision_proxy",
    ]
    all_cells: list[dict] = []
    for parameter, spec in specs.items():
        cells = cell_table(matches, false_positives, parameter, spec["bins"])
        all_cells.extend(cells)
        write_csv(analysis / f"cells_{parameter}.csv", cells, cell_fields)
        plot_parameter_cells(parameter, spec, cells, figures / "parameter_maps")
    write_csv(analysis / "cells_all_parameters.csv", all_cells, cell_fields)
    plot_snr_rows(snr_rows_from_matches(matches, false_positives), figures / "summary")

    run_summary = {
        "output_root": str(args.output_root),
        "sim_root": str(args.sim_root),
        "truth_root": str(args.truth_root),
        "candidate_count": int(sum(int(row.get("candidate_count", 0) or 0) for row in statuses)),
        "event_count": len(events),
        "job_count": len(statuses),
        "failed_jobs": int(sum(1 for row in statuses if row.get("status") == "failed")),
        "matches": len(matches),
        "false_positives": len(false_positives),
        "summary": summary,
        "mode": args.mode,
        "dm_window": args.dm_window,
        "dm_step": args.dm_step,
        "threshold": args.threshold,
        "maxwidth": args.maxwidth,
        "context_left_files": args.context_left_files,
        "context_right_files": args.context_right_files,
        "crop_dat_to_center": args.crop_dat_to_center,
        "center_numout": args.numout,
        "source_dm_tolerance": args.source_dm_tolerance,
        "source_time_tolerance_ms": args.source_time_tolerance_ms,
    }
    (args.output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_batches(text: str) -> list[int]:
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-run-root",
        type=Path,
        default=None,
        help="Legacy layout root containing simdata/ and truth_archive/; ignored when --sim-root/--truth-root are set.",
    )
    parser.add_argument("--sim-root", type=Path, default=None, help="Shared injected FITS root, usually ../simdata.")
    parser.add_argument("--truth-root", type=Path, default=None, help="Shared truth archive root, usually ../truth_archive.")
    parser.add_argument("--run-label", type=str, default=DEFAULT_RUN_LABEL)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--batches", type=parse_batches, default=parse_batches("0-19"))
    parser.add_argument("--quantizations", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=list(QUANTIZATIONS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-ids", type=lambda s: [int(x) for x in s.split(",") if x.strip()], default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--limit-jobs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", choices=["blind", "known-window"], default="blind")
    parser.add_argument("--search-all-files", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--dm-window", type=float, default=60.0)
    parser.add_argument("--dm-step", type=float, default=1.0)
    parser.add_argument("--dm-block-size", type=int, default=300)
    parser.add_argument("--dm-min", type=float, default=100.0)
    parser.add_argument("--dm-max", type=float, default=2000.0)
    parser.add_argument("--source-dm-tolerance", type=float, default=60.0)
    parser.add_argument("--source-time-tolerance-ms", type=float, default=30.0)
    parser.add_argument("--event-dedup-dm-tolerance", type=float, default=60.0)
    parser.add_argument("--event-dedup-time-tolerance-ms", type=float, default=30.0)
    parser.add_argument("--localize-dm-tolerance", type=float, default=25.0)
    parser.add_argument("--localize-time-tolerance-ms", type=float, default=30.0)

    parser.add_argument("--rfifind-time", type=float, default=1.0)
    parser.add_argument("--skip-rfifind", action="store_true")
    parser.add_argument("--numout", type=int, default=131072, help="Samples per center FITS file. PRESTO receives this value multiplied by the number of context files.")
    parser.add_argument("--context-left-files", type=int, default=1)
    parser.add_argument("--context-right-files", type=int, default=1)
    parser.add_argument("--crop-dat-to-center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nsub", type=int, default=1024)
    parser.add_argument("--retry-nsub", type=int, default=4096)
    parser.add_argument("--downsamp", type=int, default=1)
    parser.add_argument("--noclip", action="store_true")
    parser.add_argument("--xargs-chunk", type=int, default=408)
    parser.add_argument("--xargs-procs", type=int, default=12)
    parser.add_argument("--maxwidth", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--min-dat-std", type=float, default=1e-6)
    parser.add_argument("--keep-dat", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if args.output_root is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = DEFAULT_OUTPUT_BASE / "results" / f"presto_blind_{stamp}"
    if args.context_left_files < 0 or args.context_right_files < 0:
        raise SystemExit("--context-left-files and --context-right-files must be non-negative")
    if args.sim_root is None:
        args.sim_root = args.input_run_root / "simdata" if args.input_run_root is not None else DEFAULT_SIM_ROOT
    if args.truth_root is None:
        args.truth_root = args.input_run_root / "truth_archive" if args.input_run_root is not None else DEFAULT_TRUTH_ROOT
    if args.input_run_root is not None:
        args.input_run_root = args.input_run_root.resolve()
    args.sim_root = args.sim_root.resolve()
    args.truth_root = args.truth_root.resolve()
    args.output_root = args.output_root.resolve()
    args.scratch_root = args.scratch_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    args = parse_args()
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (args.output_root / "campaign_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    truth = load_truth_archive(args.truth_root, args.batches, args.quantizations)
    if not truth:
        raise SystemExit(f"No truth rows found under {args.truth_root}")
    (args.output_root / "truth_manifest_used.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in truth),
        encoding="utf-8",
    )

    if not args.aggregate_only:
        jobs = build_jobs(args, truth)
        print(f"[campaign] output_root={args.output_root}", flush=True)
        print(f"[campaign] sim_root={args.sim_root} truth_root={args.truth_root}", flush=True)
        print(f"[campaign] truth_rows={len(truth)} jobs={len(jobs)} workers={args.workers} gpu_ids={args.gpu_ids}", flush=True)
        failures = 0
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, job, vars(args)) for job in jobs]
            for future in as_completed(futures):
                status = future.result()
                done += 1
                if status.get("status") == "failed":
                    failures += 1
                if done == 1 or done % 20 == 0 or status.get("status") == "failed":
                    print(
                        f"[campaign] done={done}/{len(jobs)} failures={failures} "
                        f"last={status.get('quantization')} b{status.get('batch'):02d} {status.get('file_stem')} "
                        f"status={status.get('status')} cand={status.get('candidate_count', '')}",
                        flush=True,
                    )
        if failures:
            print(f"[WARN] {failures} jobs failed; aggregating successful jobs only.", flush=True)

    aggregate_and_plot(args, truth)
    print(json.dumps(json.loads((args.output_root / "run_summary.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
