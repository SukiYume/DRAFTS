"""启动单个注入数据目录的 DRAFTS 分段搜索。

本脚本只负责把一个 raw8 或 packed2 FITS 目录切成多个 section 并行搜索。
真正的目标检测、分类和候选落盘逻辑在 `search_runtime/t-blind-section.py`
及其调用的 core/gate 文件中。每个 section 单独写 log 和候选 JSONL，避免
并发进程抢写同一个文件。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch DRAFTS sections for an injection run.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("search_runtime"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, required=True)
    parser.add_argument("--gpu-num", type=int, default=1)
    parser.add_argument("--gpu-ids", type=str, default="")
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
    parser.add_argument(
        "--cpu-threads-per-process",
        type=int,
        default=1,
        help="OpenMP/BLAS/Torch CPU threads allowed for each section process.",
    )
    parser.add_argument("--no-save-npy", action="store_true")
    parser.add_argument("--no-save-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_thread_limits(env: dict[str, str], threads: int) -> None:
    """限制每个 section 进程的 OpenMP/BLAS/Torch 线程数，防止线程爆炸。"""
    if threads < 1:
        raise SystemExit("--cpu-threads-per-process must be >= 1")
    value = str(threads)
    # 同时启动 raw8/packed2 时会有 16 个 section 进程；不限制这些
    # OpenMP/BLAS 线程会触发 libgomp thread creation failed。
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env[key] = value
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def main() -> None:
    # 把一份注入数据切成 gpu_num 个 section，每个 section 独占一张 GPU 并行搜索；
    # 每个 section 写各自的候选 manifest，避免并发写同一文件。
    args = parse_args()
    runtime_dir = args.runtime_dir.resolve()
    section_script = runtime_dir / "t-blind-section.py"
    if not section_script.exists():
        raise SystemExit(f"missing section script: {section_script}")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        gpu_ids = [str(i) for i in range(args.gpu_num)]
    if len(gpu_ids) < args.gpu_num:
        raise SystemExit("--gpu-ids must provide at least --gpu-num ids")

    data_path = args.data_path.resolve()
    output_root = args.output_root.resolve()
    logs_dir = output_root / "logs"
    candidate_dir = output_root / "candidate_manifests"
    proposal_dir = output_root / "proposal_manifests"
    logs_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    proposal_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    processes = []
    commands = []
    for section in range(args.gpu_num):
        manifest = candidate_dir / f"{args.run_label}_section{section:02d}_candidates.jsonl"
        proposal_manifest = proposal_dir / f"{args.run_label}_section{section:02d}_proposals.jsonl"
        cmd = [
            sys.executable, str(section_script),
            "--section", str(section),
            "--gpu-num", str(args.gpu_num),
            "--data-path", str(data_path),
            "--output-root", str(output_root / "search_outputs"),
            "--run-label", args.run_label,
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
            "--candidate-manifest", str(manifest),
            "--proposal-manifest", str(proposal_manifest),
        ]
        if args.no_save_npy:
            cmd.append("--no-save-npy")
        if args.no_save_plot:
            cmd.append("--no-save-plot")
        if args.dry_run:
            cmd.append("--dry-run")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[section]
        apply_thread_limits(env, args.cpu_threads_per_process)
        log_path = logs_dir / f"{args.run_label}_section{section:02d}.log"
        commands.append({
            "section": section,
            "gpu": gpu_ids[section],
            "log": str(log_path),
            "manifest": str(manifest),
            "proposal_manifest": str(proposal_manifest),
            "cmd": cmd,
        })
        log_handle = log_path.open("w", encoding="utf-8")
        print(f"[launch] section={section} gpu={gpu_ids[section]} log={log_path}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(runtime_dir), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((section, proc, log_handle))

    failures = []
    for section, proc, log_handle in processes:
        code = proc.wait()
        log_handle.close()
        print(f"[done] section={section} exit={code}", flush=True)
        if code != 0:
            failures.append({"section": section, "exit_code": code})
    ended = time.time()
    write_json(output_root / "run_metadata.json", {
        "run_label": args.run_label,
        "data_path": str(data_path),
        "output_root": str(output_root),
        "gpu_num": args.gpu_num,
        "gpu_ids": gpu_ids[:args.gpu_num],
        "det_prob": args.det_prob,
        "class_threshold": args.class_threshold,
        "duration_seconds": ended - started,
        "commands": commands,
        "failures": failures,
        "dry_run": args.dry_run,
    })
    if failures:
        raise SystemExit(f"{len(failures)} section(s) failed")


if __name__ == "__main__":
    main()
