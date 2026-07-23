"""跨批次汇总注入评估结果，并绘制 S/N–参数二维热图。

每次搜索会产出一个 ``analysis/<batch>_<quantization>/`` 目录。本脚本把这些
单批 CSV 合并，再画 recall 和"precision 代理"的 S/N–参数热图：

* recall = cell 内被匹配的注入源 / cell 内注入源总数；
* precision_proxy = 被匹配的注入源 / (被匹配的注入源 + 归到该 cell 的误报事件)。

之所以叫"代理"：误报本身没有注入 S/N、宽度等真值参数，这里把每个误报硬归到
最近注入源的参数 cell，才能让它出现在 S/N–参数平面上。误报单位是去重后的
检测事件，不是来自不同 frequency split / 邻近 DM chunk 的单条候选。

重要口径说明：背景是真实望远镜数据，含真实暴发和 RFI，所以 precision_proxy 把
"算法真误报"和"背景真实信号"混在一起，不是干净的算法精确率，解读时需谨慎
（详见 analyze_search_results.py 顶部说明）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


QUANTIZATIONS = ("raw8", "packed2")
SNR_BINS = [5, 7, 10, 15, 22, 33, 50, 75, 100]
PARAM_SPECS = {
    "dm_pc_cm3": {
        "label": "DM (pc cm$^{-3}$)",
        "bins": [100, 300, 500, 800, 1100, 1400, 1700, 2000],
    },
    "width_ms_fwhm": {
        "label": "Temporal FWHM (ms)",
        "bins": [1, 1.5, 2.5, 4, 6.5, 10, 15, 20],
    },
    "bandwidth_mhz_fwhm": {
        "label": "Spectral FWHM (MHz)",
        "bins": [50, 75, 110, 160, 230, 340, 500],
    },
    "scattering_ms_at_1ghz": {
        "label": "Scattering tau at 1 GHz (ms)",
        "bins": [0, 1e-9, 0.03, 0.1, 0.3, 1, 3, 10],
        "labels": ["0", "0.00-0.03", "0.03-0.10", "0.10-0.30", "0.30-1", "1-3", "3-10"],
    },
    "center_freq_mhz": {
        "label": "Center frequency (MHz)",
        "bins": None,
    },
    "per_channel_peak_snr": {
        "label": "Peak S/N per FWHM-band channel",
        "bins": [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.5, 8],
    },
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def finite_float(row: dict, key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def detect_quantization(path: Path, row: dict, prefix: str = "") -> str:
    """判断该行属于 raw8 还是 packed2：优先看字段值，否则从路径名兜底。"""
    key = f"{prefix}quantization" if prefix else "quantization"
    value = str(row.get(key, "")).strip()
    if value in QUANTIZATIONS:
        return value
    text = str(path)
    if "packed2" in text:
        return "packed2"
    return "raw8"


def load_campaign_tables(analysis_root: Path) -> tuple[list[dict], list[dict]]:
    """递归收集所有单批的 matches.csv 与 false_positives.csv，并标注量化类型。"""
    matches: list[dict] = []
    false_positives: list[dict] = []
    for path in sorted(analysis_root.glob("**/matches.csv")):
        for row in read_csv(path):
            row["source_analysis_dir"] = str(path.parent)
            row["quantization"] = detect_quantization(path, row)
            matches.append(row)
    for path in sorted(analysis_root.glob("**/false_positives.csv")):
        for row in read_csv(path):
            row["source_analysis_dir"] = str(path.parent)
            row["assigned_quantization"] = detect_quantization(path, row, prefix="assigned_")
            false_positives.append(row)
    return matches, false_positives


def infer_center_freq_bins(rows: list[dict], bins: int = 7) -> list[float]:
    values = [finite_float(row, "center_freq_mhz") for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
    lo = math.floor(min(values) / 10.0) * 10.0
    hi = math.ceil(max(values) / 10.0) * 10.0
    if hi <= lo:
        hi = lo + 10.0
    return np.linspace(lo, hi, bins + 1).round(3).tolist()


def bin_index(value: float | None, bins: list[float]) -> int | None:
    if value is None:
        return None
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if (lo <= value < hi) or (idx == len(bins) - 2 and lo <= value <= hi):
            return idx
    return None


def labels_for_bins(bins: list[float], custom: list[str] | None = None) -> list[str]:
    if custom is not None:
        return custom
    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        labels.append(f"{lo:g}-{hi:g}")
    return labels


def cell_tables(
    matches: list[dict],
    false_positives: list[dict],
    parameter: str,
    parameter_bins: list[float],
    quantization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在 (参数 × S/N) 网格里统计每格的 recall 与 precision_proxy。

    返回 (recall, precision, total, fp) 四个二维数组：total 为注入数、fp 为
    归到该格的误报数；recall=detected/total，precision=detected/(detected+fp)。
    """
    shape = (len(parameter_bins) - 1, len(SNR_BINS) - 1)
    total = np.zeros(shape, dtype=int)
    detected = np.zeros(shape, dtype=int)
    fp = np.zeros(shape, dtype=int)

    for row in matches:
        if row.get("quantization") != quantization:
            continue
        x = bin_index(finite_float(row, "snr"), SNR_BINS)
        y = bin_index(finite_float(row, parameter), parameter_bins)
        if x is None or y is None:
            continue
        total[y, x] += 1
        if parse_bool(row.get("detected")):
            detected[y, x] += 1

    assigned_key = f"assigned_{parameter}"
    for row in false_positives:
        if row.get("assigned_quantization") != quantization:
            continue
        x = bin_index(finite_float(row, "assigned_snr"), SNR_BINS)
        y = bin_index(finite_float(row, assigned_key), parameter_bins)
        if x is None or y is None:
            continue
        fp[y, x] += 1

    recall = np.divide(detected, total, out=np.full(shape, np.nan, dtype=float), where=total > 0)
    precision = np.divide(
        detected,
        detected + fp,
        out=np.full(shape, np.nan, dtype=float),
        where=(detected + fp) > 0,
    )
    return recall, precision, total, fp


def draw_contours(ax, values: np.ndarray, levels: Iterable[float]) -> None:
    valid = np.isfinite(values)
    if valid.sum() < 4:
        return
    filled = values.copy()
    finite_values = filled[valid]
    filled[~valid] = float(np.nanmedian(finite_values))
    x = np.arange(values.shape[1])
    y = np.arange(values.shape[0])
    try:
        contours = ax.contour(x, y, filled, levels=list(levels), colors="white", linewidths=0.8, alpha=0.85)
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    except ValueError:
        return


def annotate_counts(ax, values: np.ndarray, counts: np.ndarray) -> None:
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            if counts[y, x] <= 0:
                continue
            value = values[y, x]
            label = f"{value:.2f}\nn={counts[y, x]}" if np.isfinite(value) else f"n={counts[y, x]}"
            color = "white" if np.nan_to_num(value, nan=0.0) < 0.55 else "black"
            ax.text(x, y, label, ha="center", va="center", fontsize=6.8, color=color)


def plot_metric_pair(
    output: Path,
    title: str,
    parameter_label: str,
    parameter_bins: list[float],
    parameter_labels: list[str] | None,
    recall: np.ndarray,
    precision: np.ndarray,
    counts: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), constrained_layout=True)
    for ax, data, name in zip(axes, [recall, precision], ["Recall", "Precision proxy"]):
        image = ax.imshow(data, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
        draw_contours(ax, data, [0.25, 0.5, 0.75, 0.9])
        annotate_counts(ax, data, counts)
        ax.set_title(name)
        ax.set_xticks(np.arange(len(SNR_BINS) - 1), labels_for_bins(SNR_BINS), rotation=35, ha="right")
        ax.set_yticks(np.arange(len(parameter_bins) - 1), labels_for_bins(parameter_bins, parameter_labels))
        ax.set_xlabel("Injected fluence S/N")
        ax.set_ylabel(parameter_label)
        ax.grid(color="white", alpha=0.18, linewidth=0.5)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=name.lower())
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_loss_pair(
    output: Path,
    title: str,
    parameter_label: str,
    parameter_bins: list[float],
    parameter_labels: list[str] | None,
    recall_loss: np.ndarray,
    precision_loss: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), constrained_layout=True)
    vmax = max(0.05, float(np.nanmax(np.abs([recall_loss, precision_loss]))) if np.isfinite([recall_loss, precision_loss]).any() else 0.05)
    for ax, data, name in zip(axes, [recall_loss, precision_loss], ["Recall loss raw8-packed2", "Precision proxy loss raw8-packed2"]):
        image = ax.imshow(data, origin="lower", vmin=-vmax, vmax=vmax, cmap="coolwarm", aspect="auto")
        draw_contours(ax, data, [-0.2, -0.1, 0, 0.1, 0.2])
        ax.set_title(name)
        ax.set_xticks(np.arange(len(SNR_BINS) - 1), labels_for_bins(SNR_BINS), rotation=35, ha="right")
        ax.set_yticks(np.arange(len(parameter_bins) - 1), labels_for_bins(parameter_bins, parameter_labels))
        ax.set_xlabel("Injected fluence S/N")
        ax.set_ylabel(parameter_label)
        ax.grid(color="black", alpha=0.12, linewidth=0.5)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="raw8 - packed2")
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def summarize(matches: list[dict], false_positives: list[dict]) -> dict:
    """按量化类型汇总 recall / precision_proxy，并给出 packed2 相对 raw8 的差。"""
    out: dict[str, dict] = {}
    for quantization in QUANTIZATIONS:
        rows = [row for row in matches if row.get("quantization") == quantization]
        fps = [row for row in false_positives if row.get("assigned_quantization") == quantization]
        detected = sum(1 for row in rows if parse_bool(row.get("detected")))
        total = len(rows)
        fp_count = len(fps)
        out[quantization] = {
            "truth_count": total,
            "detected_count": detected,
            "false_positive_count": fp_count,
            "false_positive_unit": "deduplicated_detection_event",
            "recall": detected / total if total else 0.0,
            "precision_proxy": detected / (detected + fp_count) if (detected + fp_count) else 0.0,
        }
    if out["raw8"]["truth_count"] and out["packed2"]["truth_count"]:
        out["packed2_minus_raw8"] = {
            "recall_delta": out["packed2"]["recall"] - out["raw8"]["recall"],
            "precision_proxy_delta": out["packed2"]["precision_proxy"] - out["raw8"]["precision_proxy"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate injection campaign analysis outputs.")
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    matches, false_positives = load_campaign_tables(args.analysis_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if matches:
        write_csv(args.output_dir / "all_matches.csv", matches, sorted({key for row in matches for key in row}))
    if false_positives:
        write_csv(
            args.output_dir / "all_false_positives.csv",
            false_positives,
            sorted({key for row in false_positives for key in row}),
        )

    summary = summarize(matches, false_positives)
    write_json(args.output_dir / "campaign_summary.json", summary)

    specs = dict(PARAM_SPECS)
    specs["center_freq_mhz"] = {
        **specs["center_freq_mhz"],
        "bins": infer_center_freq_bins(matches),
    }

    cell_rows = []
    for parameter, spec in specs.items():
        bins = spec["bins"]
        labels = spec.get("labels")
        per_quant = {}
        for quantization in QUANTIZATIONS:
            recall, precision, counts, fp = cell_tables(matches, false_positives, parameter, bins, quantization)
            per_quant[quantization] = (recall, precision, counts, fp)
            plot_metric_pair(
                args.output_dir / f"snr_vs_{parameter}_{quantization}.png",
                f"{quantization}: S/N vs {spec['label']}",
                spec["label"],
                bins,
                labels,
                recall,
                precision,
                counts,
            )
            for y, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
                for x, (snr_lo, snr_hi) in enumerate(zip(SNR_BINS[:-1], SNR_BINS[1:])):
                    cell_rows.append({
                        "quantization": quantization,
                        "parameter": parameter,
                        "param_low": lo,
                        "param_high": hi,
                        "snr_low": snr_lo,
                        "snr_high": snr_hi,
                        "n_injected": int(counts[y, x]),
                        "n_detected": int(round(np.nan_to_num(recall[y, x]) * counts[y, x])),
                        "n_false_positive_assigned": int(fp[y, x]),
                        "recall": "" if not np.isfinite(recall[y, x]) else float(recall[y, x]),
                        "precision_proxy": "" if not np.isfinite(precision[y, x]) else float(precision[y, x]),
                    })

        raw_recall, raw_precision, _, _ = per_quant["raw8"]
        packed_recall, packed_precision, _, _ = per_quant["packed2"]
        plot_loss_pair(
            args.output_dir / f"snr_vs_{parameter}_raw8_minus_packed2.png",
            f"2-bit quantization loss: S/N vs {spec['label']}",
            spec["label"],
            bins,
            labels,
            raw_recall - packed_recall,
            raw_precision - packed_precision,
        )

    write_csv(
        args.output_dir / "cell_metrics.csv",
        cell_rows,
        [
            "quantization",
            "parameter",
            "param_low",
            "param_high",
            "snr_low",
            "snr_high",
            "n_injected",
            "n_detected",
            "n_false_positive_assigned",
            "recall",
            "precision_proxy",
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
