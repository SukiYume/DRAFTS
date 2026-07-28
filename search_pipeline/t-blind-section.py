"""Run one d-center-binary-gate section for a data path.

This wrapper keeps d-center-binary-gate.py as the source of truth for
organization, GPU-section slicing, model loading, and processing.

Example, production-like CRAFTS parameters on one local section:
    CUDA_VISIBLE_DEVICES=0 python t-blind-section.py \
        --section 0 --gpu-num 1 \
        --data-path /path/to/observations/source/date \
        --output-root /path/to/drafts_runs/blind \
        --beam M01 \
        --dm-range 4096 --dm-scale 1 --dm-offset 0 \
        --dm-threshold 10 --block-size 4096 --dm-span 1024 --det-prob 0.45
"""

import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
GATE_PATH = BASE_DIR / "d-center-binary-gate.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("d_center_binary_gate", GATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load gate module from {GATE_PATH}")
    gate = importlib.util.module_from_spec(spec)
    sys.modules["d_center_binary_gate"] = gate
    spec.loader.exec_module(gate)
    return gate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--detector-type", type=str, default="centernet_conv_tiny")
    parser.add_argument("--detector-ckpt", type=str, default="models/object_best_model_centernet_conv_tiny_ema_v10.pth")
    parser.add_argument("--classifier-ckpt", type=str, default="models/binary_best_model_conv_small_ema.pth")
    parser.add_argument("--beam", type=str, default="M01")
    parser.add_argument("--gpu-num", type=int, default=8)
    parser.add_argument("--dm-range", type=int, default=4096)
    parser.add_argument("--dm-scale", type=float, default=1.0)
    parser.add_argument("--dm-offset", type=float, default=0.0)
    parser.add_argument("--dm-threshold", type=float, default=50.0)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--dm-span", type=int, default=1024)
    parser.add_argument("--det-prob", type=float, default=0.45)
    parser.add_argument("--time-factor", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.section < 0 or args.section >= args.gpu_num:
        raise ValueError(f"--section must be in [0, gpu_num), got {args.section}/{args.gpu_num}")

    os.chdir(BASE_DIR)
    gate = load_gate()

    detector_ckpt = str((BASE_DIR / args.detector_ckpt).resolve())
    classifier_ckpt = str((BASE_DIR / args.classifier_ckpt).resolve())
    output_root = Path(args.output_root).resolve()

    process_config = gate.ProcessConfig(
        dm_range=args.dm_range,
        dm_scale=args.dm_scale,
        dm_offset=args.dm_offset,
        dm_threshold=args.dm_threshold,
        block_size=args.block_size,
        dm_span=args.dm_span,
        det_prob=args.det_prob,
        section_num=args.gpu_num,
        time_factor=args.time_factor,
    )

    print(f"[Run] section={args.section} data_path={args.data_path}", flush=True)
    print(f"[Run] output_root={output_root}", flush=True)
    print(
        f"[Run] dm_range={args.dm_range} dm_scale={args.dm_scale} dm_offset={args.dm_offset} "
        f"dm_threshold={args.dm_threshold} block_size={args.block_size} "
        f"dm_span={args.dm_span} time_factor={args.time_factor}",
        flush=True,
    )
    print(f"[Run] detector={args.detector_type} ckpt={detector_ckpt}", flush=True)
    print(f"[Run] classifier=convnext_small ckpt={classifier_ckpt}", flush=True)

    model, decode_fn = gate.load_detector(args.detector_type, detector_ckpt)
    class_model = gate.load_binary_classifier(
        classifier_ckpt,
        model_type="ConvNeXtNet",
        model_name="convnext_small",
    )

    all_file_lists = gate.organize_file_lists(args.data_path, args.beam)
    distribute_args = [all_file_lists, process_config.section_num, args.section]
    if len(inspect.signature(gate.distribute_file_lists).parameters) >= 4:
        distribute_args.append(process_config)
    section_file_lists = gate.distribute_file_lists(*distribute_args)

    print(f"[Run] total_file_lists={len(all_file_lists)} section_file_lists={len(section_file_lists)}", flush=True)
    if args.dry_run:
        for identifier, fits_list, info in section_file_lists:
            save_path = str(output_root / info["source"] / "CentData" / info["date"] / info["beam"]) + os.sep
            n_search = info.get("n_search_files", len(fits_list))
            print(
                f"[DryRun] {identifier}: files={len(fits_list)} n_search_files={n_search} "
                f"save_path={save_path}",
                flush=True,
            )
        return

    for identifier, fits_list, info in section_file_lists:
        save_path = str(output_root / info["source"] / "CentData" / info["date"] / info["beam"]) + os.sep
        print(f"[Run] processing {identifier}: files={len(fits_list)} save_path={save_path}", flush=True)
        process_kwargs = {"decode_fn": decode_fn}
        if "n_search_files" in inspect.signature(gate.process_fits_list).parameters:
            process_kwargs["n_search_files"] = info.get("n_search_files")
        gate.process_fits_list(fits_list, save_path, model, class_model, process_config, **process_kwargs)
        print(f"[Run] done {identifier}", flush=True)


if __name__ == "__main__":
    main()
