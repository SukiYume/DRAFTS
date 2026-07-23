"""从完整 PRESTO event 表导出可本地重算阈值的紧凑结果包。

完整 PRESTO 搜索会产生很大的 candidate/event 表。这个脚本保留阈值扫描必需的
truth、按 sigma 聚合的事件直方图和误报直方图，便于在本地反复画不同 sigma
阈值的 recall/precision-proxy 图。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from presto_common import (
    SNR_BINS,
    bin_index,
    candidate_key,
    event_matches_truth,
    nearest_truth,
    param_specs_for_truth,
    truth_key,
)
TRUTH_FIELDS = [
    "injection_id",
    "quantization",
    "batch",
    "output_file_stem",
    "highest_freq_toa_file_raw_sample",
    "time_reso_seconds",
    "snr",
    "dm_pc_cm3",
    "width_ms_fwhm",
    "bandwidth_mhz_fwhm",
    "scattering_ms_at_1ghz",
    "center_freq_mhz",
]
EVENT_FIELDS = [
    "event_id",
    "quantization",
    "batch",
    "file_stem",
    "sigma",
    "sigma_centi",
    "dm_pc_cm3",
    "sample",
    "event_size",
    "downfact",
    "dt_seconds",
]


def load_truth_manifest(path: Path) -> tuple[list[dict], dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    specs = param_specs_for_truth(rows)
    for row in rows:
        row["_snr_bin"] = bin_index(float(row["snr"]), SNR_BINS)
        for parameter, spec in specs.items():
            row[f"_bin_{parameter}"] = bin_index(float(row[parameter]), spec["bins"])
    return rows, specs


def sigma_centi(value: str | float) -> int:
    return int(round(float(value) * 100.0))


def write_truth(path: Path, truth: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRUTH_FIELDS)
        writer.writeheader()
        for row in truth:
            writer.writerow({field: row.get(field, "") for field in TRUTH_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dm-tolerance", type=float, default=60.0)
    parser.add_argument("--source-time-tolerance-ms", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=2_000_000)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    truth, specs = load_truth_manifest(result_root / "truth_manifest_used.jsonl")
    truth_by_key: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in truth:
        truth_by_key[truth_key(row)].append(row)
    write_truth(output_dir / "truth_slim.csv.gz", truth)

    fp_total_hist: Counter[tuple[str, int]] = Counter()
    fp_cell_hist: Counter[tuple[str, str, int, int, int]] = Counter()
    events_total = 0
    near_total = 0
    fp_total = 0
    fp_without_truth = 0
    started = time.time()

    with (result_root / "aggregate" / "all_events.csv").open("r", encoding="utf-8", newline="") as src, gzip.open(output_dir / "near_truth_events.csv.gz", "wt", encoding="utf-8", newline="") as near_handle:
        reader = csv.DictReader(src)
        near_writer = csv.DictWriter(near_handle, fieldnames=EVENT_FIELDS)
        near_writer.writeheader()
        for event in reader:
            events_total += 1
            key = candidate_key(event)
            local_truth = truth_by_key.get(key, [])
            sig = sigma_centi(event["sigma"])
            near = bool(local_truth) and any(event_matches_truth(event, row, args.source_dm_tolerance, args.source_time_tolerance_ms) for row in local_truth)
            if near:
                near_total += 1
                out = {field: event.get(field, "") for field in EVENT_FIELDS}
                out["sigma_centi"] = sig
                near_writer.writerow(out)
            else:
                fp_total += 1
                quant = str(event["quantization"])
                fp_total_hist[(quant, sig)] += 1
                if local_truth:
                    assigned = nearest_truth(event, local_truth)
                    if assigned is not None and assigned.get("_snr_bin") is not None:
                        for parameter, spec in specs.items():
                            param_bin = assigned.get(f"_bin_{parameter}")
                            if param_bin is not None:
                                fp_cell_hist[(parameter, quant, sig, int(assigned["_snr_bin"]), int(param_bin))] += 1
                else:
                    fp_without_truth += 1
            if events_total % args.progress_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(json.dumps({"events": events_total, "near": near_total, "fp": fp_total, "events_per_second": events_total / elapsed}, sort_keys=True), flush=True)

    with (output_dir / "fp_total_hist.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["quantization", "sigma_centi", "count"])
        writer.writeheader()
        for (quant, sig), count in sorted(fp_total_hist.items()):
            writer.writerow({"quantization": quant, "sigma_centi": sig, "count": count})

    with (output_dir / "fp_cell_hist.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter", "quantization", "sigma_centi", "snr_bin", "param_bin", "count"])
        writer.writeheader()
        for (parameter, quant, sig, snr_bin, param_bin), count in sorted(fp_cell_hist.items()):
            writer.writerow({"parameter": parameter, "quantization": quant, "sigma_centi": sig, "snr_bin": snr_bin, "param_bin": param_bin, "count": count})

    metadata = {
        "result_root": str(result_root),
        "truth_count": len(truth),
        "events_total": events_total,
        "near_truth_events": near_total,
        "false_positive_events": fp_total,
        "false_positive_events_without_local_truth": fp_without_truth,
        "source_dm_tolerance": args.source_dm_tolerance,
        "source_time_tolerance_ms": args.source_time_tolerance_ms,
        "snr_bins": SNR_BINS,
        "param_specs": specs,
        "created_files": [
            "truth_slim.csv.gz",
            "near_truth_events.csv.gz",
            "fp_total_hist.csv",
            "fp_cell_hist.csv",
            "metadata.json",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
