"""PRESTO 注入实验的公共统计和画图工具。

本模块只放跨脚本共享的纯函数：参数分箱、truth/event 关联、CSV 读写以及最终
recall/precision-proxy 图。PRESTO 命令执行、结果包导出和阈值扫描入口分别留在
各自脚本中，避免把运行逻辑和展示逻辑混在一起。
"""

from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


QUANTIZATIONS = ("raw8", "packed2")
SNR_BINS = [5, 7, 10, 15, 22, 33, 50, 75, 100]
PARAM_SPECS = {
    "dm_pc_cm3": {
        "label": "DM (pc cm$^{-3}$)",
        "short": "DM",
        "bins": [100, 300, 500, 750, 1000, 1250, 1500, 1750, 2000],
    },
    "width_ms_fwhm": {
        "label": "Width FWHM (ms)",
        "short": "Width",
        "bins": [1, 1.4, 2, 3, 4.5, 7, 10, 14, 20],
    },
    "bandwidth_mhz_fwhm": {
        "label": "Bandwidth FWHM (MHz)",
        "short": "Bandwidth",
        "bins": [50, 70, 95, 130, 180, 240, 320, 410, 500],
    },
    "scattering_ms_at_1ghz": {
        "label": "Scattering tau at 1 GHz (ms)",
        "short": "Scattering",
        "bins": [0, 1e-9, 0.07, 0.16, 0.36, 0.83, 1.9, 4.4, 10],
        "labels": [
            "0",
            "0.03-0.07",
            "0.07-0.16",
            "0.16-0.36",
            "0.36-0.83",
            "0.83-1.9",
            "1.9-4.4",
            "4.4-10",
        ],
    },
    "center_freq_mhz": {
        "label": "Center frequency (MHz)",
        "short": "Center freq.",
        "bins": None,
    },
}
MAKO_LIKE = LinearSegmentedColormap.from_list(
    "mako_like",
    ["#0B0405", "#221150", "#352A87", "#276B8E", "#1F9E89", "#79D2A6", "#DEF5E5"],
)


# ---------------------------------------------------------------------------
# 文件读写
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    """读取普通 CSV 或 gzip 压缩 CSV。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """按指定字段写 CSV；多余字段忽略，缺失字段写空字符串。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 分箱和 truth/event 关联
# ---------------------------------------------------------------------------

def bin_index(value: float, bins: list[float]) -> int | None:
    """返回 value 所在的半开区间索引；最后一个 bin 右端点闭合。"""
    for index, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if (value >= lo and value < hi) or (index == len(bins) - 2 and value == hi):
            return index
    return None


def labels_for_bins(bins: list[float], labels: list[str] | None = None) -> list[str]:
    """生成图上使用的 bin 标签；散射等不规则 bin 可传入人工标签。"""
    if labels:
        return labels
    return [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])]


def infer_center_freq_bins(truth: list[dict], bins: int = 8) -> list[float]:
    """根据 truth 中实际中心频率范围生成等宽 center-frequency 分箱。"""
    values = [float(row["center_freq_mhz"]) for row in truth if row.get("center_freq_mhz") not in ("", None)]
    if not values:
        return [1000, 1062.5, 1125, 1187.5, 1250, 1312.5, 1375, 1437.5, 1500]
    lo = math.floor(min(values) / 10.0) * 10.0
    hi = math.ceil(max(values) / 10.0) * 10.0
    if hi <= lo:
        hi = lo + 10.0
    return np.linspace(lo, hi, bins + 1).round(3).tolist()


def param_specs_for_truth(truth: list[dict]) -> dict:
    """复制默认参数规格，并用当前 truth 的频率范围补齐 center-frequency bins。"""
    specs = {key: dict(value) for key, value in PARAM_SPECS.items()}
    specs["center_freq_mhz"]["bins"] = infer_center_freq_bins(truth)
    return specs


def truth_key(row: dict) -> tuple[str, int, str]:
    return (str(row["quantization"]), int(row["batch"]), str(row["output_file_stem"]))


def candidate_key(row: dict) -> tuple[str, int, str]:
    return (str(row["quantization"]), int(row["batch"]), str(row["file_stem"]))


def event_key(row: dict) -> tuple[str, int, str]:
    return candidate_key(row)


def event_dt_ms(event: dict, truth: dict) -> float:
    dt = float(truth.get("time_reso_seconds") or event.get("dt_seconds") or 4.9152e-05)
    return (float(event["sample"]) - float(truth["highest_freq_toa_file_raw_sample"])) * dt * 1000.0


def event_ddm(event: dict, truth: dict) -> float:
    return float(event["dm_pc_cm3"]) - float(truth["dm_pc_cm3"])


def event_matches_truth(event: dict, truth: dict, dm_tol: float, time_tol_ms: float) -> bool:
    return abs(event_ddm(event, truth)) <= dm_tol and abs(event_dt_ms(event, truth)) <= time_tol_ms


def nearest_truth_with_distance(event: dict, truths: list[dict]) -> tuple[dict | None, float]:
    """按 time + 0.5 * DM 的经验距离找最近 truth。"""
    best = None
    best_dist = float("inf")
    for truth in truths:
        dist = abs(event_dt_ms(event, truth)) + 0.5 * abs(event_ddm(event, truth))
        if dist < best_dist:
            best = truth
            best_dist = dist
    return best, best_dist


def nearest_truth(event: dict, truths: list[dict]) -> dict | None:
    return nearest_truth_with_distance(event, truths)[0]


# ---------------------------------------------------------------------------
# 画图：所有 PRESTO 最终图只从这里生成
# ---------------------------------------------------------------------------

def metric_matrix(cells: list[dict], quant: str, metric: str, bins: list[float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(bins) - 1, len(SNR_BINS) - 1), np.nan)
    counts = np.zeros_like(values, dtype=int)
    for row in cells:
        if row["quantization"] != quant:
            continue
        y = int(row["param_bin"])
        x = int(row["snr_bin"])
        counts[y, x] = int(row["n_injected"])
        value = row.get(metric, "")
        if value != "":
            values[y, x] = float(value)
    return values, counts


def clean_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        which="both",
        top=True,
        right=True,
        direction="out",
        length=3.5,
        width=0.8,
        colors="black",
        labelsize=8.5,
    )
    ax.grid(False)


def draw_contours(ax, data: np.ndarray) -> None:
    if np.isfinite(data).sum() < 4:
        return
    filled = data.copy()
    filled[~np.isfinite(filled)] = float(np.nanmedian(filled))
    levels = [level for level in (0.9, 0.95, 0.98, 0.99) if np.nanmin(filled) <= level <= np.nanmax(filled)]
    if not levels:
        return
    contours = ax.contour(
        np.arange(data.shape[1]),
        np.arange(data.shape[0]),
        filled,
        levels=levels,
        colors="white",
        linewidths=0.75,
        alpha=0.85,
    )
    ax.clabel(contours, inline=True, fontsize=6.8, fmt="%.2f")


def annotate_cells(ax, data: np.ndarray, counts: np.ndarray) -> None:
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if counts[y, x] <= 0 or not np.isfinite(data[y, x]):
                continue
            color = "white" if data[y, x] < 0.965 else "black"
            ax.text(x, y, f"{data[y, x]:.2f}\nn={counts[y, x]}", ha="center", va="center", fontsize=6.3, color=color)


def plot_parameter_cells(
    parameter: str,
    spec: dict,
    cells: list[dict],
    out_dir: Path,
    threshold: float | None = None,
) -> None:
    """画某个参数的 recall/precision-proxy 热图，一次输出 raw8 和 packed2。"""
    bins = spec["bins"]
    labels = spec.get("labels")
    for quant in QUANTIZATIONS:
        recall, counts = metric_matrix(cells, quant, "recall", bins)
        precision, _ = metric_matrix(cells, quant, "precision_proxy", bins)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=False)
        for ax, data, title, cbar_label in [
            (axes[0], recall, "Recall", "recall"),
            (axes[1], precision, "Precision proxy", "precision proxy"),
        ]:
            finite = data[np.isfinite(data)]
            vmin = 0.0 if threshold is not None and finite.size and np.nanmin(finite) < 0.9 else 0.9
            image = ax.imshow(data, origin="lower", vmin=vmin, vmax=1.0, cmap=MAKO_LIKE, aspect="auto")
            ax.set_title(title, fontsize=11)
            ax.set_xticks(np.arange(len(SNR_BINS) - 1))
            ax.set_xticklabels(labels_for_bins(SNR_BINS), rotation=35, ha="right")
            ax.set_yticks(np.arange(len(bins) - 1))
            ax.set_yticklabels(labels_for_bins(bins, labels))
            ax.set_xlabel("Injected fluence S/N")
            ax.set_ylabel(spec["label"])
            annotate_cells(ax, data, counts)
            draw_contours(ax, data)
            clean_axes(ax)
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
            cbar.ax.set_ylabel(cbar_label)
        if threshold is None:
            title = f"PRESTO {quant}: source recovery over S/N and {spec['short']}"
        else:
            title = f"PRESTO {quant}: sigma >= {threshold:g}, S/N vs {spec['short']}"
        fig.suptitle(title, fontsize=12.5, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{parameter}_{quant}_recall_precision.png", dpi=300)
        fig.savefig(out_dir / f"{parameter}_{quant}_recall_precision.pdf")
        plt.close(fig)


def snr_rows_from_matches(matches: list[dict], false_positives: list[dict]) -> list[dict]:
    """把 source 级 matches 和误报事件压成按 S/N 分箱的折线图输入。"""
    rows = []
    for quant in QUANTIZATIONS:
        q_matches = [row for row in matches if row.get("quantization") == quant]
        q_fp = [row for row in false_positives if row.get("assigned_quantization") == quant]
        for x, (lo, hi) in enumerate(zip(SNR_BINS[:-1], SNR_BINS[1:])):
            subset = [row for row in q_matches if bin_index(float(row["snr"]), SNR_BINS) == x]
            fp_subset = [
                row
                for row in q_fp
                if row.get("assigned_snr") not in ("", None)
                and bin_index(float(row["assigned_snr"]), SNR_BINS) == x
            ]
            total = len(subset)
            detected = sum(1 for row in subset if bool(row.get("detected")))
            fp_count = len(fp_subset)
            rows.append(
                {
                    "quantization": quant,
                    "snr_low": lo,
                    "snr_high": hi,
                    "snr_center": math.sqrt(lo * hi),
                    "n_injected": total,
                    "n_detected": detected,
                    "n_false_positive_assigned": fp_count,
                    "recall": detected / total if total else np.nan,
                    "precision_proxy": detected / (detected + fp_count) if detected + fp_count else np.nan,
                }
            )
    return rows


def plot_snr_rows(snr_rows: list[dict], out_dir: Path, threshold: float | None = None) -> None:
    """画 PRESTO recall/precision-proxy 随注入 S/N 的摘要折线图。"""
    colors = {"raw8": "#0072B2", "packed2": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for metric, ax, ylabel in [("recall", axes[0], "Recall"), ("precision_proxy", axes[1], "Precision proxy")]:
        for quant in QUANTIZATIONS:
            part = [row for row in snr_rows if row["quantization"] == quant]
            ax.plot([row["snr_center"] for row in part], [row[metric] for row in part], marker="o", color=colors[quant], label=quant)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(SNR_BINS)
        ax.set_xticklabels([f"{x:g}" for x in SNR_BINS], rotation=35, ha="right")
        ax.set_xlabel("Injected fluence S/N")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        clean_axes(ax)
    if threshold is None:
        title = "PRESTO blind-search source-level performance versus S/N"
    else:
        title = f"PRESTO blind-search performance, sigma >= {threshold:g}"
    fig.suptitle(title, fontsize=12.5, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "snr_recall_precision.png", dpi=300)
    fig.savefig(out_dir / "snr_recall_precision.pdf")
    plt.close(fig)
