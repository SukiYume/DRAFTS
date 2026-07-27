"""分批运行 raw8/packed2 注入评估和 DRAFTS 搜索。

中文维护说明：本文件是注入实验的主控入口，负责按 batch 调用注入、DL 搜索、
单批匹配分析和跨批汇总；真正的 FITS 注入在 `generate_injections.py`，真正的 GPU 搜索在
`search_runtime/`。当前默认搜索配置固定为 v10 CenterNet、det_prob=0.30、
classifier 输入 1024 个采样点并额外做 2 倍时间降采样。

The campaign runner is intentionally a thin orchestration layer around the
existing injection, search, analysis, and aggregation scripts.  It now uses a
bounded pipeline:

* write at most a small number of injection batches ahead of search;
* for each ready batch, search raw8 and packed2 at the same time;
* give each quantization search the full requested GPU list, so every GPU can
  carry one raw8 worker and one packed2 worker concurrently;
* analyze and delete each injected batch immediately after both searches finish.

For model/threshold comparisons, use ``--generate-only`` once to keep all
injected FITS on disk, then rerun ``--search-only`` with different checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


HERE = Path(__file__).resolve().parent
DEFAULT_BACKGROUND_DIR = Path("/path/to/generation_data/rawdata")
DEFAULT_BASE_ROOT = Path(os.environ.get("INJECTION_EXPERIMENT_ROOT", str(HERE)))
DEFAULT_WORK_ROOT = Path(os.environ.get("INJECTION_WORK_ROOT", str(DEFAULT_BASE_ROOT / "runs")))
DEFAULT_SIM_ROOT = Path(os.environ.get("INJECTION_SIM_ROOT", str(DEFAULT_BASE_ROOT / "simdata")))
DEFAULT_TRUTH_ROOT = Path(os.environ.get("INJECTION_TRUTH_ROOT", str(DEFAULT_BASE_ROOT / "truth_archive")))
DEFAULT_RUN_LABEL = "v10_det03_injection_10000"
DEFAULT_GPU_IDS = "0,1,2,3,4,5,6,7"
DEFAULT_PYTHON = sys.executable


@dataclass(frozen=True)
class CampaignDirs:
    root: Path
    sim: Path
    search: Path
    analysis: Path
    truth: Path
    logs: Path
    aggregate: Path
    status: Path


@dataclass(frozen=True)
class BatchPaths:
    index: int
    label: str
    seed: int
    raw8_dir: Path
    packed2_dir: Path


@dataclass
class RunningCommand:
    name: str
    batch: int | None
    phase: str
    cmd: list[str]
    cwd: Path
    log_path: Path
    process: subprocess.Popen
    log_handle: TextIO
    started: float


@dataclass
class SearchState:
    batch: BatchPaths
    pending_quantizations: list[str]
    running: list[tuple[str, RunningCommand]]
    started: float


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def command_text(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def run_command(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool = False) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cmd] {command_text(cmd)}", flush=True)
    if dry_run:
        return 0.0
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(f"command failed exit={proc.returncode} log={log_path}")
    print(f"[done-cmd] seconds={elapsed:.1f} log={log_path}", flush=True)
    return elapsed


def start_command(name: str, batch: int | None, phase: str, cmd: list[str], cwd: Path, log_path: Path) -> RunningCommand:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[start] {name} {command_text(cmd)}", flush=True)
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    return RunningCommand(
        name=name,
        batch=batch,
        phase=phase,
        cmd=cmd,
        cwd=cwd,
        log_path=log_path,
        process=proc,
        log_handle=handle,
        started=time.time(),
    )


def finish_command(running: RunningCommand) -> float:
    code = running.process.wait()
    running.log_handle.close()
    elapsed = time.time() - running.started
    if code != 0:
        raise RuntimeError(
            f"{running.name} failed exit={code} phase={running.phase} "
            f"batch={running.batch} log={running.log_path}"
        )
    print(f"[done] {running.name} seconds={elapsed:.1f} log={running.log_path}", flush=True)
    return elapsed


def terminate_running(commands: list[RunningCommand]) -> None:
    """异常退出时优雅终止所有仍在运行的子进程（先 terminate 后 kill）。"""
    for running in commands:
        if running.process.poll() is None:
            print(f"[terminate] {running.name} pid={running.process.pid}", flush=True)
            running.process.terminate()
    deadline = time.time() + 30.0
    for running in commands:
        if running.process.poll() is None:
            timeout = max(0.0, deadline - time.time())
            try:
                running.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"[kill] {running.name} pid={running.process.pid}", flush=True)
                running.process.kill()
                running.process.wait()
        running.log_handle.close()


def archive_truth(data_dir: Path, archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in ("truth_manifest.jsonl", "run_config.json"):
        source = data_dir / name
        if source.exists():
            shutil.copy2(source, archive_dir / name)


def remove_injected_fits_dir(data_dir: Path) -> None:
    if not data_dir.exists():
        return
    # 只删除当前 campaign 自己创建的 raw8/packed2 目录，避免误删背景数据。
    if not (data_dir.name.endswith("_raw8") or data_dir.name.endswith("_packed2")):
        raise RuntimeError(f"refuse to delete unexpected data dir: {data_dir}")
    shutil.rmtree(data_dir)


def build_common_search_args(args: argparse.Namespace) -> list[str]:
    return [
        "--runtime-dir", str(args.runtime_dir),
        "--gpu-num", str(args.gpu_num),
        "--gpu-ids", args.gpu_ids,
        "--beam", args.beam,
        "--dm-range", str(args.dm_range),
        "--dm-scale", str(args.dm_scale),
        "--dm-offset", str(args.dm_offset),
        "--dm-threshold", str(args.dm_threshold),
        "--block-size", str(args.block_size),
        "--dm-span", str(args.dm_span),
        "--dm-stride", str(args.dm_stride),
        "--det-prob", str(args.det_prob),
        "--class-threshold", str(args.class_threshold),
        "--class-block-size", str(args.class_block_size),
        "--class-time-downsample", str(args.class_time_downsample),
        "--time-factor", str(args.time_factor),
        "--classifier-batch-size", str(args.classifier_batch_size),
        "--dedup-dm-tolerance", str(args.dedup_dm_tolerance),
        "--dedup-time-tolerance-ms", str(args.dedup_time_tolerance_ms),
        "--detector-type", args.detector_type,
        "--detector-ckpt", args.detector_ckpt,
        "--classifier-ckpt", args.classifier_ckpt,
        "--classifier-model-name", args.classifier_model_name,
        "--cpu-threads-per-process", str(args.search_cpu_threads),
        "--no-save-npy",
        "--no-save-plot",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-batch DRAFTS injection campaign.")
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUND_DIR)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT, help="Root directory for shared injected raw8/packed2 FITS batches.")
    parser.add_argument("--truth-root", type=Path, default=DEFAULT_TRUTH_ROOT, help="Root directory for archived truth manifests shared by DL and PRESTO.")
    parser.add_argument("--run-label", type=str, default=DEFAULT_RUN_LABEL)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--count-per-batch", type=int, default=500)
    parser.add_argument("--search-file-first", type=int, default=11)
    parser.add_argument("--search-file-last", type=int, default=279)
    parser.add_argument("--inject-file-first", type=int, default=12)
    parser.add_argument("--inject-file-last", type=int, default=275)
    # 兼容旧脚本参数；如果提供，会覆盖 search-file-*。
    parser.add_argument("--file-first", type=int, default=None)
    parser.add_argument("--file-last", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--inject-workers", type=int, default=8)
    parser.add_argument("--inject-io-workers", type=int, default=8)
    parser.add_argument("--parallel-injection-batches", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--std-file-limit", type=int, default=8)
    parser.add_argument("--std-samples-per-file", type=int, default=16384)
    parser.add_argument("--runtime-dir", type=Path, default=HERE / "search_runtime")
    parser.add_argument("--gpu-num", type=int, default=8)
    parser.add_argument("--gpu-ids", type=str, default=DEFAULT_GPU_IDS)
    parser.add_argument("--beam", type=str, default="M01")
    parser.add_argument("--dm-range", type=int, default=4096)
    parser.add_argument("--dm-scale", type=float, default=0.5)
    parser.add_argument("--dm-offset", type=float, default=0.0)
    parser.add_argument("--dm-threshold", type=float, default=90.0)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--dm-span", type=int, default=1024)
    parser.add_argument("--dm-stride", type=int, default=0)
    parser.add_argument("--det-prob", type=float, default=0.30)
    parser.add_argument("--class-threshold", type=float, default=0.5)
    parser.add_argument("--class-block-size", type=int, default=1024)
    parser.add_argument("--class-time-downsample", type=int, default=2)
    parser.add_argument("--time-factor", type=float, default=8.0)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--dedup-dm-tolerance", type=float, default=0.0)
    parser.add_argument("--dedup-time-tolerance-ms", type=float, default=0.0)
    parser.add_argument("--detector-type", type=str, default="centernet_conv_tiny")
    parser.add_argument("--detector-ckpt", type=str, default="models/object_best_model_centernet_conv_tiny_ema_v10.pth")
    parser.add_argument("--classifier-ckpt", type=str, default="models/binary_best_model_conv_small_ema.pth")
    parser.add_argument("--classifier-model-name", type=str, default="convnext_small")
    parser.add_argument("--search-cpu-threads", type=int, default=1)
    parser.add_argument("--dm-tolerance", type=float, default=25.0)
    parser.add_argument("--time-tolerance-ms", type=float, default=30.0)
    parser.add_argument(
        "--source-dm-tolerance",
        type=float,
        default=60.0,
        help="Source-level DM association tolerance for recall/FP suppression.",
    )
    parser.add_argument(
        "--source-time-tolerance-ms",
        type=float,
        default=None,
        help="Source-level time association tolerance; defaults to --time-tolerance-ms.",
    )
    parser.add_argument("--event-dedup-dm-tolerance", type=float, default=None)
    parser.add_argument("--event-dedup-time-tolerance-ms", type=float, default=None)
    parser.add_argument("--keep-injected-fits", action="store_true")
    parser.add_argument("--generate-only", action="store_true", help="Only write and archive injected raw8/packed2 data.")
    parser.add_argument("--search-only", action="store_true", help="Search/analyze an existing generated campaign.")
    parser.add_argument("--overwrite-search", action="store_true", help="For --search-only, replace search/analysis/aggregate outputs only.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(parallel_quantization_searches=True)
    parser.add_argument("--parallel-quantization-searches", dest="parallel_quantization_searches", action="store_true")
    parser.add_argument("--no-parallel-quantization-searches", dest="parallel_quantization_searches", action="store_false")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> None:
    if args.file_first is not None:
        args.search_file_first = args.file_first
    if args.file_last is not None:
        args.search_file_last = args.file_last
    if args.batches < 1:
        raise SystemExit("--batches must be >= 1")
    if args.generate_only and args.search_only:
        raise SystemExit("--generate-only and --search-only are mutually exclusive")
    if args.search_only and args.overwrite:
        raise SystemExit("--search-only must not use --overwrite; use --overwrite-search to preserve generated FITS")
    if args.count_per_batch < 1:
        raise SystemExit("--count-per-batch must be >= 1")
    if args.search_file_first > args.search_file_last:
        raise SystemExit("--search-file-first must be <= --search-file-last")
    if args.inject_file_first > args.inject_file_last:
        raise SystemExit("--inject-file-first must be <= --inject-file-last")
    if not (args.search_file_first <= args.inject_file_first <= args.inject_file_last <= args.search_file_last):
        raise SystemExit(
            "Injection window must sit inside search window; "
            f"search={args.search_file_first}-{args.search_file_last} "
            f"inject={args.inject_file_first}-{args.inject_file_last}"
        )
    if args.inject_workers < 1:
        raise SystemExit("--inject-workers must be >= 1")
    if args.inject_io_workers < 1:
        raise SystemExit("--inject-io-workers must be >= 1")
    if args.parallel_injection_batches < 1:
        raise SystemExit("--parallel-injection-batches must be >= 1")
    if args.gpu_num < 1:
        raise SystemExit("--gpu-num must be >= 1")
    if args.search_cpu_threads < 1:
        raise SystemExit("--search-cpu-threads must be >= 1")
    if args.poll_seconds < 1.0:
        raise SystemExit("--poll-seconds must be >= 1")


def build_dirs(args: argparse.Namespace) -> CampaignDirs:
    root = args.work_root / args.run_label
    return CampaignDirs(
        root=root,
        sim=args.sim_root,
        search=root / "search",
        analysis=root / "analysis",
        truth=args.truth_root,
        logs=root / "logs",
        aggregate=root / "aggregate",
        status=root / "campaign_status.jsonl",
    )


def build_batches(args: argparse.Namespace, dirs: CampaignDirs) -> list[BatchPaths]:
    batches = []
    for index in range(args.batches):
        label = f"{args.run_label}_b{index:02d}"
        batches.append(BatchPaths(
            index=index,
            label=label,
            seed=args.seed + index,
            raw8_dir=dirs.sim / f"{label}_raw8",
            packed2_dir=dirs.sim / f"{label}_packed2",
        ))
    return batches


def build_inject_cmd(args: argparse.Namespace, dirs: CampaignDirs, batch: BatchPaths) -> list[str]:
    return [
        args.python, str(HERE / "generate_injections.py"),
        "--background-dir", str(args.background_dir),
        "--output-root", str(dirs.sim),
        "--run-label", batch.label,
        "--file-first", str(args.search_file_first),
        "--file-last", str(args.search_file_last),
        "--inject-file-first", str(args.inject_file_first),
        "--inject-file-last", str(args.inject_file_last),
        "--count", str(args.count_per_batch),
        "--seed", str(batch.seed),
        "--workers", str(args.inject_workers),
        "--io-workers", str(args.inject_io_workers),
        "--std-file-limit", str(args.std_file_limit),
        "--std-samples-per-file", str(args.std_samples_per_file),
        "--overwrite",
    ]


def quantization_dir(batch: BatchPaths, quantization: str) -> Path:
    if quantization == "raw8":
        return batch.raw8_dir
    if quantization == "packed2":
        return batch.packed2_dir
    raise ValueError(f"unknown quantization: {quantization}")


def build_search_cmd(
    args: argparse.Namespace,
    dirs: CampaignDirs,
    batch: BatchPaths,
    quantization: str,
    common_search_args: list[str],
) -> list[str]:
    run_label = f"{batch.label}_{quantization}"
    return [
        args.python, str(HERE / "launch_search.py"),
        "--data-path", str(quantization_dir(batch, quantization)),
        "--output-root", str(dirs.search / run_label),
        "--run-label", run_label,
        *common_search_args,
    ]


def build_analyze_cmd(args: argparse.Namespace, dirs: CampaignDirs, batch: BatchPaths, quantization: str) -> list[str]:
    run_label = f"{batch.label}_{quantization}"
    cmd = [
        args.python, str(HERE / "evaluate_results.py"),
        "--truth", str(dirs.truth / run_label / "truth_manifest.jsonl"),
        "--candidates", str(dirs.search / run_label / "candidate_manifests"),
        "--output-dir", str(dirs.analysis / run_label),
        "--dm-tolerance", str(args.dm_tolerance),
        "--time-tolerance-ms", str(args.time_tolerance_ms),
        "--source-dm-tolerance", str(args.source_dm_tolerance),
    ]
    if args.source_time_tolerance_ms is not None:
        cmd.extend(["--source-time-tolerance-ms", str(args.source_time_tolerance_ms)])
    if args.event_dedup_dm_tolerance is not None:
        cmd.extend(["--event-dedup-dm-tolerance", str(args.event_dedup_dm_tolerance)])
    if args.event_dedup_time_tolerance_ms is not None:
        cmd.extend(["--event-dedup-time-tolerance-ms", str(args.event_dedup_time_tolerance_ms)])
    return cmd


def build_aggregate_cmd(args: argparse.Namespace, dirs: CampaignDirs) -> list[str]:
    return [
        args.python, str(HERE / "aggregate_results.py"),
        "--analysis-root", str(dirs.analysis),
        "--output-dir", str(dirs.aggregate),
    ]


def archive_batch_truth(batch: BatchPaths, dirs: CampaignDirs) -> None:
    for quantization in ("raw8", "packed2"):
        run_label = f"{batch.label}_{quantization}"
        archive_truth(quantization_dir(batch, quantization), dirs.truth / run_label)


def maybe_start_more_searches(
    args: argparse.Namespace,
    dirs: CampaignDirs,
    state: SearchState,
    common_search_args: list[str],
) -> None:
    limit = 2 if args.parallel_quantization_searches else 1
    while state.pending_quantizations and len(state.running) < limit:
        quantization = state.pending_quantizations.pop(0)
        run_label = f"{state.batch.label}_{quantization}"
        cmd = build_search_cmd(args, dirs, state.batch, quantization, common_search_args)
        running = start_command(
            name=f"{run_label}_search",
            batch=state.batch.index,
            phase="search",
            cmd=cmd,
            cwd=HERE,
            log_path=dirs.logs / f"{run_label}_search.log",
        )
        append_jsonl(dirs.status, {
            "event": "start_search",
            "batch": state.batch.index,
            "label": state.batch.label,
            "quantization": quantization,
            "pid": running.process.pid,
            "time": time.time(),
        })
        state.running.append((quantization, running))


def start_search_state(
    args: argparse.Namespace,
    dirs: CampaignDirs,
    batch: BatchPaths,
    common_search_args: list[str],
) -> SearchState:
    archive_batch_truth(batch, dirs)
    state = SearchState(
        batch=batch,
        pending_quantizations=["raw8", "packed2"],
        running=[],
        started=time.time(),
    )
    maybe_start_more_searches(args, dirs, state, common_search_args)
    return state


def poll_search_state(
    args: argparse.Namespace,
    dirs: CampaignDirs,
    state: SearchState,
    common_search_args: list[str],
) -> bool:
    progressed = False
    for quantization, running in list(state.running):
        if running.process.poll() is None:
            continue
        elapsed = finish_command(running)
        append_jsonl(dirs.status, {
            "event": "done_search",
            "batch": state.batch.index,
            "label": state.batch.label,
            "quantization": quantization,
            "seconds": elapsed,
            "time": time.time(),
        })
        state.running.remove((quantization, running))
        progressed = True
    maybe_start_more_searches(args, dirs, state, common_search_args)
    return progressed


def analyze_and_cleanup_batch(args: argparse.Namespace, dirs: CampaignDirs, batch: BatchPaths) -> None:
    for quantization in ("raw8", "packed2"):
        run_label = f"{batch.label}_{quantization}"
        append_jsonl(dirs.status, {
            "event": "start_analyze",
            "batch": batch.index,
            "label": batch.label,
            "quantization": quantization,
            "time": time.time(),
        })
        elapsed = run_command(
            build_analyze_cmd(args, dirs, batch, quantization),
            HERE,
            dirs.logs / f"{run_label}_analyze.log",
            dry_run=False,
        )
        append_jsonl(dirs.status, {
            "event": "done_analyze",
            "batch": batch.index,
            "label": batch.label,
            "quantization": quantization,
            "seconds": elapsed,
            "time": time.time(),
        })

    if not args.keep_injected_fits and not args.search_only:
        remove_injected_fits_dir(batch.raw8_dir)
        remove_injected_fits_dir(batch.packed2_dir)
        append_jsonl(dirs.status, {"event": "deleted_injected_fits", "batch": batch.index, "time": time.time()})

    append_jsonl(dirs.status, {"event": "done_batch", "batch": batch.index, "label": batch.label, "time": time.time()})


def write_campaign_config(args: argparse.Namespace, dirs: CampaignDirs, filename: str = "campaign_config.json") -> None:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["campaign_root"] = str(dirs.root)
    config["total_injections"] = args.batches * args.count_per_batch
    config["mode"] = "generate-only" if args.generate_only else "search-only" if args.search_only else "pipeline"
    config["pipeline"] = {
        "parallel_injection_batches": args.parallel_injection_batches,
        "parallel_quantization_searches": args.parallel_quantization_searches,
        "search_uses_full_gpu_list_per_quantization": True,
    }
    write_json(dirs.root / filename, config)


def dry_run_campaign(args: argparse.Namespace, dirs: CampaignDirs, batches: list[BatchPaths], common_search_args: list[str]) -> None:
    write_campaign_config(args, dirs)
    for batch in batches:
        if not args.search_only:
            run_command(build_inject_cmd(args, dirs, batch), HERE, dirs.logs / f"{batch.label}_inject.log", dry_run=True)
        if args.generate_only:
            continue
        for quantization in ("raw8", "packed2"):
            run_label = f"{batch.label}_{quantization}"
            run_command(
                build_search_cmd(args, dirs, batch, quantization, common_search_args),
                HERE,
                dirs.logs / f"{run_label}_search.log",
                dry_run=True,
            )
            run_command(
                build_analyze_cmd(args, dirs, batch, quantization),
                HERE,
                dirs.logs / f"{run_label}_analyze.log",
                dry_run=True,
            )
    if not args.generate_only:
        run_command(build_aggregate_cmd(args, dirs), HERE, dirs.logs / "aggregate.log", dry_run=True)
    print(f"[dry-run-done] campaign_root={dirs.root}", flush=True)


def start_injection_batch(args: argparse.Namespace, dirs: CampaignDirs, batch: BatchPaths) -> RunningCommand:
    """启动单个 batch 的注入子进程，并写入 start_batch / start_inject 状态。

    被 generate-only 和 pipeline 两条路径共用，避免重复同一段启动逻辑。
    """
    append_jsonl(dirs.status, {
        "event": "start_batch",
        "batch": batch.index,
        "label": batch.label,
        "seed": batch.seed,
        "time": time.time(),
    })
    running = start_command(
        name=f"{batch.label}_inject",
        batch=batch.index,
        phase="inject",
        cmd=build_inject_cmd(args, dirs, batch),
        cwd=HERE,
        log_path=dirs.logs / f"{batch.label}_inject.log",
    )
    append_jsonl(dirs.status, {
        "event": "start_inject",
        "batch": batch.index,
        "label": batch.label,
        "pid": running.process.pid,
        "time": time.time(),
    })
    return running


def run_generate_only(args: argparse.Namespace, dirs: CampaignDirs, batches: list[BatchPaths]) -> None:
    """只生成并归档 truth，保留 raw8/packed2 数据供后续多次搜索。"""
    pending_batches = list(batches)
    running_injections: list[tuple[BatchPaths, RunningCommand]] = []

    try:
        while pending_batches or running_injections:
            progressed = False

            while pending_batches and len(running_injections) < args.parallel_injection_batches:
                batch = pending_batches.pop(0)
                running = start_injection_batch(args, dirs, batch)
                running_injections.append((batch, running))
                progressed = True

            for batch, running in list(running_injections):
                if running.process.poll() is None:
                    continue
                elapsed = finish_command(running)
                running_injections.remove((batch, running))
                archive_batch_truth(batch, dirs)
                append_jsonl(dirs.status, {
                    "event": "done_inject",
                    "batch": batch.index,
                    "label": batch.label,
                    "seconds": elapsed,
                    "time": time.time(),
                })
                append_jsonl(dirs.status, {
                    "event": "done_generate_batch",
                    "batch": batch.index,
                    "label": batch.label,
                    "time": time.time(),
                })
                progressed = True

            if not progressed:
                time.sleep(args.poll_seconds)
    except Exception:
        terminate_running([running for _, running in running_injections])
        raise


def run_search_existing(
    args: argparse.Namespace,
    dirs: CampaignDirs,
    batches: list[BatchPaths],
    common_search_args: list[str],
) -> None:
    """搜索已经生成好的数据；不会删除 simdata。"""
    for batch in batches:
        for data_dir in (batch.raw8_dir, batch.packed2_dir):
            if not data_dir.exists():
                raise RuntimeError(f"missing generated data dir for --search-only: {data_dir}")

    ready_batches = list(batches)
    active_search: SearchState | None = None

    try:
        while ready_batches or active_search is not None:
            progressed = False

            if active_search is None and ready_batches:
                batch = ready_batches.pop(0)
                append_jsonl(dirs.status, {
                    "event": "start_search_batch",
                    "batch": batch.index,
                    "label": batch.label,
                    "time": time.time(),
                })
                active_search = start_search_state(args, dirs, batch, common_search_args)
                progressed = True

            if active_search is not None:
                progressed = poll_search_state(args, dirs, active_search, common_search_args) or progressed
                if not active_search.pending_quantizations and not active_search.running:
                    analyze_and_cleanup_batch(args, dirs, active_search.batch)
                    active_search = None
                    progressed = True

            if not progressed:
                time.sleep(args.poll_seconds)
    except Exception:
        if active_search is not None:
            terminate_running([running for _, running in active_search.running])
        raise


def run_pipeline(args: argparse.Namespace, dirs: CampaignDirs, batches: list[BatchPaths], common_search_args: list[str]) -> None:
    """完整流水线：边注入、边搜索、边分析并删除已搜完的注入数据。

    用 pending/running/ready/active 四个状态把注入提前量限制在
    parallel_injection_batches 之内，避免一次性占满磁盘。
    """
    pending_batches = list(batches)
    running_injections: list[tuple[BatchPaths, RunningCommand]] = []
    ready_batches: list[BatchPaths] = []
    active_search: SearchState | None = None

    def live_commands() -> list[RunningCommand]:
        commands = [running for _, running in running_injections]
        if active_search is not None:
            commands.extend(running for _, running in active_search.running)
        return commands

    try:
        while pending_batches or running_injections or ready_batches or active_search is not None:
            progressed = False

            # 控制注入提前量：running + ready 不超过 parallel_injection_batches。
            while pending_batches and (len(running_injections) + len(ready_batches)) < args.parallel_injection_batches:
                batch = pending_batches.pop(0)
                running = start_injection_batch(args, dirs, batch)
                running_injections.append((batch, running))
                progressed = True

            for batch, running in list(running_injections):
                if running.process.poll() is None:
                    continue
                elapsed = finish_command(running)
                running_injections.remove((batch, running))
                ready_batches.append(batch)
                append_jsonl(dirs.status, {
                    "event": "done_inject",
                    "batch": batch.index,
                    "label": batch.label,
                    "seconds": elapsed,
                    "time": time.time(),
                })
                progressed = True

            if active_search is None and ready_batches:
                batch = ready_batches.pop(0)
                active_search = start_search_state(args, dirs, batch, common_search_args)
                progressed = True

            if active_search is not None:
                progressed = poll_search_state(args, dirs, active_search, common_search_args) or progressed
                if not active_search.pending_quantizations and not active_search.running:
                    analyze_and_cleanup_batch(args, dirs, active_search.batch)
                    active_search = None
                    progressed = True

            if not progressed:
                time.sleep(args.poll_seconds)
    except Exception:
        terminate_running(live_commands())
        raise


def main() -> None:
    args = parse_args()
    normalize_args(args)
    dirs = build_dirs(args)
    if args.search_only:
        if not dirs.root.exists() and not args.dry_run:
            raise SystemExit(f"{dirs.root} does not exist; run --generate-only first")
        if args.overwrite_search and not args.dry_run:
            for path in (dirs.search, dirs.analysis, dirs.aggregate):
                if path.exists():
                    shutil.rmtree(path)
    else:
        if dirs.root.exists() and args.overwrite:
            shutil.rmtree(dirs.root)
        if dirs.root.exists() and not args.dry_run:
            raise SystemExit(f"{dirs.root} exists; pass --overwrite or choose a new --run-label")

    write_campaign_config(args, dirs, filename="search_config.json" if args.search_only else "campaign_config.json")
    batches = build_batches(args, dirs)
    common_search_args = build_common_search_args(args)
    mode = "generate-only" if args.generate_only else "search-only" if args.search_only else "pipeline"
    print(
        f"[config] mode={mode} batches={args.batches} count_per_batch={args.count_per_batch} "
        f"search_files={args.search_file_first}-{args.search_file_last} "
        f"inject_files={args.inject_file_first}-{args.inject_file_last} "
        f"sim_root={dirs.sim} truth_root={dirs.truth} "
        f"parallel_injections={args.parallel_injection_batches} "
        f"parallel_quant_search={args.parallel_quantization_searches} "
        f"gpu_num={args.gpu_num} gpu_ids={args.gpu_ids}",
        flush=True,
    )

    if args.dry_run:
        dry_run_campaign(args, dirs, batches, common_search_args)
        return

    if args.generate_only:
        run_generate_only(args, dirs, batches)
        append_jsonl(dirs.status, {
            "event": "all_generated",
            "time": time.time(),
            "campaign_root": str(dirs.root),
        })
        print(f"[all-generated] campaign_root={dirs.root}", flush=True)
        return

    if args.search_only:
        run_search_existing(args, dirs, batches, common_search_args)
    else:
        run_pipeline(args, dirs, batches, common_search_args)

    elapsed = run_command(build_aggregate_cmd(args, dirs), HERE, dirs.logs / "aggregate.log", dry_run=False)
    append_jsonl(dirs.status, {
        "event": "all_done",
        "time": time.time(),
        "aggregate_root": str(dirs.aggregate),
        "aggregate_seconds": elapsed,
    })
    print(f"[all-done] campaign_root={dirs.root}", flush=True)


if __name__ == "__main__":
    main()
