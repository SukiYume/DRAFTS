"""从压缩后的 PRESTO 结果包重算不同 sigma 阈值下的指标和图。

该脚本不重新跑 PRESTO，只读取 `export_threshold_data.py` 生成的
truth/event/histogram 文件，在本地快速筛选 sigma=3/5/7 等阈值并重画
recall、precision-proxy 热图。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from search_utils import (
    QUANTIZATIONS,
    SNR_BINS,
    bin_index,
    event_ddm,
    event_dt_ms,
    event_key,
    event_matches_truth,
    plot_parameter_cells,
    plot_snr_rows,
    read_csv,
    truth_key,
    write_csv,
)


def threshold_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def sigma_threshold_centi(value: float) -> int:
    return int(math.ceil(value * 100.0 - 1e-9))


def load_package(package_dir: Path) -> tuple[list[dict], dict, dict, Counter, Counter]:
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    truth = read_csv(package_dir / "truth_slim.csv.gz")
    specs = metadata["param_specs"]
    for row in truth:
        row["_snr_bin"] = bin_index(float(row["snr"]), SNR_BINS)
        for parameter, spec in specs.items():
            row[f"_bin_{parameter}"] = bin_index(float(row[parameter]), spec["bins"])

    events_by_key: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in read_csv(package_dir / "near_truth_events.csv.gz"):
        row["sigma_centi"] = int(row["sigma_centi"])
        events_by_key[event_key(row)].append(row)

    fp_total_hist: Counter[tuple[str, int]] = Counter()
    for row in read_csv(package_dir / "fp_total_hist.csv"):
        fp_total_hist[(row["quantization"], int(row["sigma_centi"]))] += int(row["count"])

    fp_cell_hist: Counter[tuple[str, str, int, int, int]] = Counter()
    for row in read_csv(package_dir / "fp_cell_hist.csv"):
        fp_cell_hist[(row["parameter"], row["quantization"], int(row["sigma_centi"]), int(row["snr_bin"]), int(row["param_bin"]))] += int(row["count"])
    return truth, specs, events_by_key, fp_total_hist, fp_cell_hist


def fp_total_for_threshold(hist: Counter, threshold_centi: int) -> dict[str, int]:
    out = {quant: 0 for quant in QUANTIZATIONS}
    for (quant, sig), count in hist.items():
        if sig >= threshold_centi:
            out[quant] += count
    return out


def fp_cells_for_threshold(hist: Counter, specs: dict, threshold_centi: int) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for parameter, spec in specs.items():
        for quant in QUANTIZATIONS:
            out[(parameter, quant)] = np.zeros((len(spec["bins"]) - 1, len(SNR_BINS) - 1), dtype=np.int64)
    for (parameter, quant, sig, snr_bin, param_bin), count in hist.items():
        if sig >= threshold_centi:
            out[(parameter, quant)][param_bin, snr_bin] += count
    return out


def compute_threshold(threshold: float, truth: list[dict], specs: dict, events_by_key: dict, fp_total_hist: Counter, fp_cell_hist: Counter, source_dm_tol: float, source_time_tol_ms: float, localize_dm_tol: float, localize_time_tol_ms: float) -> dict:
    thr_centi = sigma_threshold_centi(threshold)
    used_events: set[str] = set()
    sorted_truth = sorted(truth, key=lambda row: (str(row["quantization"]), int(row["batch"]), str(row["output_file_stem"]), str(row["injection_id"])))

    summary = {quant: {"truth_count": 0, "detected_count": 0, "localized_count": 0} for quant in QUANTIZATIONS}
    match_counts = {}
    detected_counts = {}
    snr_counts = {quant: np.zeros(len(SNR_BINS) - 1, dtype=np.int64) for quant in QUANTIZATIONS}
    snr_detected = {quant: np.zeros(len(SNR_BINS) - 1, dtype=np.int64) for quant in QUANTIZATIONS}
    for parameter, spec in specs.items():
        for quant in QUANTIZATIONS:
            match_counts[(parameter, quant)] = np.zeros((len(spec["bins"]) - 1, len(SNR_BINS) - 1), dtype=np.int64)
            detected_counts[(parameter, quant)] = np.zeros((len(spec["bins"]) - 1, len(SNR_BINS) - 1), dtype=np.int64)

    for row in sorted_truth:
        quant = row["quantization"]
        summary[quant]["truth_count"] += 1
        snr_bin = row["_snr_bin"]
        if snr_bin is not None:
            snr_counts[quant][snr_bin] += 1
        for parameter, spec in specs.items():
            pbin = row[f"_bin_{parameter}"]
            if snr_bin is not None and pbin is not None:
                match_counts[(parameter, quant)][pbin, snr_bin] += 1

        best_event = None
        best_score = float("inf")
        for event in events_by_key.get(truth_key(row), []):
            if int(event["sigma_centi"]) < thr_centi or event["event_id"] in used_events:
                continue
            if not event_matches_truth(event, row, source_dm_tol, source_time_tol_ms):
                continue
            dt_ms = event_dt_ms(event, row)
            ddm = event_ddm(event, row)
            score = abs(dt_ms) / max(source_time_tol_ms, 1e-6) + abs(ddm) / max(source_dm_tol, 1e-6) - 0.001 * float(event["sigma"])
            if score < best_score:
                best_score = score
                best_event = event

        if best_event is not None:
            used_events.add(best_event["event_id"])
            summary[quant]["detected_count"] += 1
            if snr_bin is not None:
                snr_detected[quant][snr_bin] += 1
            for parameter, spec in specs.items():
                pbin = row[f"_bin_{parameter}"]
                if snr_bin is not None and pbin is not None:
                    detected_counts[(parameter, quant)][pbin, snr_bin] += 1
            dm_error = event_ddm(best_event, row)
            toa_error_ms = event_dt_ms(best_event, row)
            if abs(dm_error) <= localize_dm_tol and abs(toa_error_ms) <= localize_time_tol_ms:
                summary[quant]["localized_count"] += 1

    fp_total = fp_total_for_threshold(fp_total_hist, thr_centi)
    fp_cells = fp_cells_for_threshold(fp_cell_hist, specs, thr_centi)

    summary_rows = []
    for quant in QUANTIZATIONS:
        total = summary[quant]["truth_count"]
        detected = summary[quant]["detected_count"]
        localized = summary[quant]["localized_count"]
        fp = fp_total[quant]
        summary_rows.append(
            {
                "quantization": quant,
                "truth_count": total,
                "detected_count": detected,
                "localized_count": localized,
                "false_positive_count": fp,
                "recall": detected / total if total else "",
                "localized_fraction": localized / total if total else "",
                "precision_proxy": detected / (detected + fp) if detected + fp else "",
            }
        )

    cell_rows = []
    for parameter, spec in specs.items():
        for quant in QUANTIZATIONS:
            for y, (param_lo, param_hi) in enumerate(zip(spec["bins"][:-1], spec["bins"][1:])):
                for x, (snr_lo, snr_hi) in enumerate(zip(SNR_BINS[:-1], SNR_BINS[1:])):
                    total = int(match_counts[(parameter, quant)][y, x])
                    detected = int(detected_counts[(parameter, quant)][y, x])
                    fp = int(fp_cells[(parameter, quant)][y, x])
                    cell_rows.append(
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
                            "n_false_positive_assigned": fp,
                            "recall": detected / total if total else "",
                            "precision_proxy": detected / (detected + fp) if detected + fp else "",
                        }
                    )

    snr_rows = []
    for quant in QUANTIZATIONS:
        for x, (lo, hi) in enumerate(zip(SNR_BINS[:-1], SNR_BINS[1:])):
            total = int(snr_counts[quant][x])
            detected = int(snr_detected[quant][x])
            fp = sum(int(fp_cells[("dm_pc_cm3", quant)][y, x]) for y in range(fp_cells[("dm_pc_cm3", quant)].shape[0]))
            snr_rows.append(
                {
                    "quantization": quant,
                    "snr_low": lo,
                    "snr_high": hi,
                    "snr_center": math.sqrt(lo * hi),
                    "n_injected": total,
                    "n_detected": detected,
                    "n_false_positive_assigned": fp,
                    "recall": detected / total if total else "",
                    "precision_proxy": detected / (detected + fp) if detected + fp else "",
                }
            )

    return {"summary": summary_rows, "cells": cell_rows, "snr": snr_rows}


def write_threshold_outputs(result: dict, specs: dict, out_dir: Path, threshold: float) -> None:
    aggregate = out_dir / "aggregate"
    analysis = out_dir / "analysis"
    figures = out_dir / "figures"
    summary_fields = ["quantization", "truth_count", "detected_count", "localized_count", "false_positive_count", "recall", "localized_fraction", "precision_proxy"]
    cell_fields = ["quantization", "parameter", "param_bin", "snr_bin", "param_low", "param_high", "snr_low", "snr_high", "n_injected", "n_detected", "n_false_positive_assigned", "recall", "precision_proxy"]
    snr_fields = ["quantization", "snr_low", "snr_high", "snr_center", "n_injected", "n_detected", "n_false_positive_assigned", "recall", "precision_proxy"]
    write_csv(aggregate / "summary.csv", result["summary"], summary_fields)
    write_csv(analysis / "cells_all_parameters.csv", result["cells"], cell_fields)
    write_csv(analysis / "snr_metrics.csv", result["snr"], snr_fields)
    for parameter, spec in specs.items():
        rows = [row for row in result["cells"] if row["parameter"] == parameter]
        write_csv(analysis / f"cells_{parameter}.csv", rows, cell_fields)
        plot_parameter_cells(parameter, spec, rows, figures / "parameter_maps", threshold=threshold)
    plot_snr_rows(result["snr"], figures / "summary", threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=str, default="3,5,7")
    parser.add_argument("--source-dm-tolerance", type=float, default=60.0)
    parser.add_argument("--source-time-tolerance-ms", type=float, default=30.0)
    parser.add_argument("--localize-dm-tolerance", type=float, default=25.0)
    parser.add_argument("--localize-time-tolerance-ms", type=float, default=30.0)
    args = parser.parse_args()

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    truth, specs, events_by_key, fp_total_hist, fp_cell_hist = load_package(args.package_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    for threshold in thresholds:
        result = compute_threshold(
            threshold,
            truth,
            specs,
            events_by_key,
            fp_total_hist,
            fp_cell_hist,
            args.source_dm_tolerance,
            args.source_time_tolerance_ms,
            args.localize_dm_tolerance,
            args.localize_time_tolerance_ms,
        )
        threshold_dir = args.output_dir / f"sigma_ge_{threshold_slug(threshold)}"
        write_threshold_outputs(result, specs, threshold_dir, threshold)
        for row in result["summary"]:
            out = dict(row)
            out["threshold"] = threshold
            combined.append(out)

    write_csv(args.output_dir / "summary_by_threshold.csv", combined, ["threshold", "quantization", "truth_count", "detected_count", "localized_count", "false_positive_count", "recall", "localized_fraction", "precision_proxy"])
    (args.output_dir / "run_summary.json").write_text(json.dumps({"package_dir": str(args.package_dir.resolve()), "thresholds": thresholds, "summary": combined}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "thresholds": thresholds, "summary": combined}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
