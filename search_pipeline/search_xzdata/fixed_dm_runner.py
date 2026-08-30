# fmt: off
#!/usr/bin/env python3
"""Run one section of a fixed-DM search over one observation and beam."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", type=int, required=True)
    parser.add_argument("--section-num", type=int, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--beam", required=True)
    parser.add_argument("--dm", type=float, required=True)
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--downsample-rates", required=True)
    parser.add_argument("--search-script", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def load_search_module(search_script: Path):
    search_script = search_script.resolve()
    if not search_script.is_file():
        raise FileNotFoundError(search_script)

    sys.path.insert(0, str(search_script.parent))
    spec = importlib.util.spec_from_file_location("fixed_dm_search", search_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load search module from {search_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.section_num < 1:
        raise ValueError("--section-num must be positive")
    if not 0 <= args.section < args.section_num:
        raise ValueError("--section must be in [0, section_num)")
    if not args.data_path.is_dir():
        raise NotADirectoryError(args.data_path)
    if re.fullmatch(r"M\d{2}", args.beam) is None:
        raise ValueError(
            f"Invalid beam: {args.beam}; expected M followed by two digits"
        )
    if not np.isfinite(args.dm) or args.dm < 0:
        raise ValueError("--dm must be a finite, non-negative value")
    if not np.isfinite(args.probability) or not 0 <= args.probability <= 1:
        raise ValueError("--probability must be in [0, 1]")

    rate_parts = [part.strip() for part in args.downsample_rates.split(",")]
    if not rate_parts or any(not part.isdecimal() for part in rate_parts):
        raise ValueError("--downsample-rates must be comma-separated positive integers")
    downsample_rates = np.array(
        sorted({int(part) for part in rate_parts}), dtype=np.int64
    )
    if downsample_rates[0] < 1:
        raise ValueError("--downsample-rates must contain only positive integers")
    pre_down_rate = min(int(downsample_rates[0]), 16)
    if np.any(downsample_rates % pre_down_rate):
        raise ValueError(
            "Every downsample rate must be divisible by the effective pre-downsample "
            f"rate ({pre_down_rate})"
        )

    classifier = args.classifier.resolve()
    if not classifier.is_file():
        raise FileNotFoundError(classifier)

    search       = load_search_module(args.search_script)
    required_api = (
        "ProcessConfig",
        "load_binary_classifier",
        "organize_file_lists_from_roots",
        "distribute_file_lists",
        "process_fits_list",
    )
    missing_api = [name for name in required_api if not hasattr(search, name)]
    if missing_api:
        raise AttributeError(
            f"{args.search_script} does not provide: {', '.join(missing_api)}"
        )

    config = search.ProcessConfig(
        DM                      = args.dm,
        prob                    = args.probability,
        block_size              = 512,
        section_num             = args.section_num,
        down_sampling_rate_list = downsample_rates,
    )
    model = search.load_binary_classifier(
        str(classifier),
        model_type = args.model_type,
        model_name = args.model_name,
    )
    print(f"[Model] loaded classifier from {classifier}", flush=True)

    data_path  = args.data_path.resolve()
    file_lists = search.organize_file_lists_from_roots([str(data_path)], args.beam)
    if len(file_lists) != 1:
        identifiers = [item[0] for item in file_lists]
        raise RuntimeError(
            f"Expected exactly one observation/beam list, got {len(file_lists)}: "
            f"{identifiers}"
        )

    tasks = search.distribute_file_lists(
        file_lists,
        args.section_num,
        args.section,
        config,
    )
    if len(tasks) != 1:
        raise RuntimeError(
            f"Section {args.section} expected one non-empty task, got {len(tasks)}"
        )

    identifier, fits_list, info = tasks[0]
    output = args.output_root.resolve() / "CalData" / info["date"] / info["beam"]
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"[Run] identifier={identifier} section={args.section}/{args.section_num} "
        f"files={len(fits_list)} dm={args.dm:g} rates={downsample_rates.tolist()} "
        f"output={output}",
        flush=True,
    )
    search.process_fits_list(
        fits_list,
        model,
        config,
        str(output) + os.sep,
        n_search_files=info.get("n_search_files"),
    )
    print(f"Successfully processed {identifier}", flush=True)


if __name__ == "__main__":
    main()
# fmt: on
