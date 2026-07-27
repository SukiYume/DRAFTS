"""把搜索候选匹配到注入真值，统计单批次的召回率（recall）与误报。

评估流程（数据流顺序，也是本文件函数的排列顺序）
------------------------------------------------------
1. 读取 truth_manifest.jsonl（注入真值）和各 section 的 candidate manifest。
2. 给每个候选算"全局原始采样" ``_pred_toa_global_raw_sample``，把分散在不同
   文件里的候选放到同一条时间轴上（见 ``candidate_file_index`` /
   ``match_candidates``）。
3. 用 ``build_candidate_events`` 把同一信号在不同 frequency split / DM chunk /
   邻近像素上产生的多个候选聚类成"检测事件"，并限制事件的整体 DM/时间直径，
   防止 single-linkage 桥接两个真实事件。后续 recall 和误报都以**事件**为单位。
4. ``match_candidates``：在"源级关联容差"内做最大基数、最小代价的 truth-event
   二分图匹配 → 命中即 recall 计入；同时记录是否满足更严格的定位容差。未匹配上的
   事件若不靠近任何注入源，则记为误报。
5. 输出 matches.csv / false_positives.csv、按参数分箱的 recall、若干 recall 曲线
   和 S/N–DM 召回热图、以及 summary_metrics.json。

关于误报 / precision 的重要说明（评估口径的已知局限，不影响 recall）
------------------------------------------------------------------------
* 背景是真实望远镜数据，里面本身含有真实暴发和 RFI。凡是不落在注入真值附近的
  检测都会被计成"误报"，因此这里的误报数把"算法真误报"与"背景里真实存在的信号"
  混在一起，不能等同于干净的算法 precision。
* 误报判定用 ``event_near_any_truth``：一个事件只要落在该批 **任意** 注入源的
  源级关联容差内就不算误报。每批注入很密集，会让一部分本应算误报的事件被漏算，
  使 precision 偏乐观。这两点方向相反，解读 precision 时需同时记住。
* 本文件只输出 ``false_positive_count``；真正的 precision_proxy 在
  aggregate_results.py 里跨批汇总计算。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from matching import maximum_cardinality_min_cost_matching


# FAST L 波段总带宽（MHz），用于把注入带宽换算成"有效通道数"
FAST_BANDWIDTH_MHZ = 500.0

# 源级关联比严格定位容差更宽。目标检测可能在相邻 DM chunk 上给出同一源的候选；
# 这种情况应算"找到源"，但在 matches.csv 中另记 localized_within_tolerance=False。
DEFAULT_SOURCE_DM_TOLERANCE = 60.0


@dataclass
class CandidateEvent:
    """搜索候选的事件级聚类。

    同一个注入信号可能在多个 frequency split、多个 DM chunk 或邻近像素上
    产生多个候选。评估准确率前先把这些候选合并成一个事件；后续 recall 和
    false positive 都以事件为单位，而不是以候选行为单位。
    """

    event_id: int
    representative: dict
    members: list[dict] = field(default_factory=list)

    def add(self, cand: dict) -> None:
        self.members.append(cand)
        if candidate_rank_key(cand) < candidate_rank_key(self.representative):
            self.representative = cand

    @property
    def size(self) -> int:
        return len(self.members)


# ---------------------------------------------------------------------------
# 读写工具
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    """读取 JSON Lines 文件；文件不存在时返回空列表。"""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def candidate_paths(paths: list[Path]) -> list[Path]:
    """把传入路径（目录或文件）展开成具体的 candidate manifest 文件列表。"""
    out = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.glob("**/*candidates*.jsonl")))
            out.extend(sorted(path.glob("**/candidate_manifest*.jsonl")))
        else:
            out.append(path)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# 文件索引与全局时间轴
#
# 真值和候选都按"全局原始采样数"对齐：global = file_index * samples_per_file +
# 文件内采样。候选 manifest 里只有文件名（fits_stem）和文件内 TOA，需要先把
# fits_stem 映射成 0 基的 file_index，才能和真值放在同一条时间轴上比较。
# ---------------------------------------------------------------------------

def candidate_time_raw(row: dict) -> int:
    """候选在所属 block 起始文件内的原始采样位置。"""
    return int(row["pred_toa_raw_sample"])


def file_number_from_stem(stem: str) -> int | None:
    """从文件名末尾的 ``_NNNN`` 取出 FITS 文件编号。"""
    match = re.search(r"_(\d+)$", stem)
    return int(match.group(1)) if match else None


def infer_file_number_base(truth_rows: list[dict], candidate_rows: list[dict]) -> int:
    """推断 file_index=0 对应的 FITS 文件编号（搜索窗口的第一个文件号）。

    真值行能唯一确定基号（文件号 - output_file_index）；这里同时扫描候选行的
    文件号取最小值做兜底，候选号不会低于搜索窗起始号，所以最终结果由真值锚定。
    """
    numbers = []
    for row in truth_rows:
        number = file_number_from_stem(str(row.get("output_file_stem", "")))
        if number is not None:
            numbers.append(number - int(float(row.get("output_file_index", 0))))
    for row in candidate_rows:
        number = file_number_from_stem(str(row.get("fits_stem", "")))
        if number is not None:
            numbers.append(number)
    return min(numbers) if numbers else 0


def candidate_file_index(row: dict, file_number_base: int, stem_to_index: dict[str, int]) -> int | None:
    """把候选的 fits_stem 映射成 0 基的 file_index。

    有注入的文件能直接查 ``stem_to_index``；否则用文件号减去基号兜底。
    """
    stem = str(row.get("fits_stem", ""))
    if stem in stem_to_index:
        return stem_to_index[stem]
    number = file_number_from_stem(stem)
    if number is None:
        return None
    return number - file_number_base


# ---------------------------------------------------------------------------
# 候选聚类成检测事件
# ---------------------------------------------------------------------------

def candidate_rank_key(cand: dict) -> tuple[float, float, int]:
    """事件代表候选的排序键：分类分高、检测分高者优先（元组按升序排列）。"""
    return (
        -float(cand.get("classifier_score", 0.0) or 0.0),
        -float(cand.get("detector_score", 0.0) or 0.0),
        int(cand.get("candidate_index", 0) or 0),
    )


def candidate_close(a: dict, b: dict, dm_tolerance: float, time_tolerance_samples: int) -> bool:
    """两个候选是否在 DM 和全局时间容差内（用于聚类判同源）。"""
    return (
        abs(float(a["pred_dm_pc_cm3"]) - float(b["pred_dm_pc_cm3"])) <= dm_tolerance
        and abs(int(a["_pred_toa_global_raw_sample"]) - int(b["_pred_toa_global_raw_sample"]))
        <= time_tolerance_samples
    )


def build_candidate_events(
    candidates_with_global: list[tuple[int, dict]],
    dm_tolerance: float,
    time_tolerance_samples: int,
) -> list[CandidateEvent]:
    """把分类后的候选去重成事件，并限制每个事件的整体 DM/时间直径。"""
    events: list[CandidateEvent] = []
    ordered = sorted(
        ((index, cand) for index, cand in candidates_with_global),
        key=lambda item: candidate_rank_key(item[1]),
    )
    for index, cand in ordered:
        cand = {**cand, "_candidate_index_global": index}
        compatible = [
            event
            for event in events
            if all(
                candidate_close(cand, member, dm_tolerance, time_tolerance_samples)
                for member in event.members
            )
        ]
        matched_event = min(
            compatible,
            key=lambda event: (
                abs(
                    float(cand["pred_dm_pc_cm3"])
                    - float(event.representative["pred_dm_pc_cm3"])
                )
                / max(float(dm_tolerance), 1e-6)
                + abs(
                    int(cand["_pred_toa_global_raw_sample"])
                    - int(event.representative["_pred_toa_global_raw_sample"])
                )
                / max(int(time_tolerance_samples), 1)
            ),
            default=None,
        )
        if matched_event is None:
            events.append(CandidateEvent(event_id=len(events), representative=cand, members=[cand]))
        else:
            matched_event.add(cand)
    return events


# ---------------------------------------------------------------------------
# 事件 ↔ 真值匹配
# ---------------------------------------------------------------------------

def event_match_to_truth(
    event: CandidateEvent,
    truth: dict,
    dm_tolerance: float,
    time_tolerance_samples: int,
) -> tuple[float, dict, float, int] | None:
    """返回某事件与某真值之间最优的成员级匹配（无匹配返回 None）。"""
    truth_global = int(float(truth["highest_freq_toa_global_raw_sample"]))
    best = None
    for cand in event.members:
        dm_error = float(cand["pred_dm_pc_cm3"]) - float(truth["dm_pc_cm3"])
        time_error_samples = int(cand["_pred_toa_global_raw_sample"]) - truth_global
        if abs(dm_error) > dm_tolerance or abs(time_error_samples) > time_tolerance_samples:
            continue
        rank = (
            abs(time_error_samples) / max(1, time_tolerance_samples)
            + abs(dm_error) / max(1e-6, dm_tolerance)
            - 0.05 * float(cand.get("classifier_score", 0.0) or 0.0)
        )
        if best is None or rank < best[0]:
            best = (rank, cand, dm_error, time_error_samples)
    return best


def normalize_source_tolerances(
    dm_tolerance: float,
    time_tolerance_ms: float,
    source_dm_tolerance: float | None,
    source_time_tolerance_ms: float | None,
) -> tuple[float, float]:
    """源级关联容差不得窄于严格定位容差，避免同源候选被误算成 FP。"""
    source_dm = dm_tolerance if source_dm_tolerance is None else source_dm_tolerance
    source_time = time_tolerance_ms if source_time_tolerance_ms is None else source_time_tolerance_ms
    return max(dm_tolerance, source_dm), max(time_tolerance_ms, source_time)


def event_near_any_truth(
    event: CandidateEvent,
    truth_rows: list[dict],
    source_dm_tolerance: float,
    source_time_tolerance_ms: float,
) -> bool:
    """事件是否落在任意注入源的源级关联容差内（用于排除误报）。"""
    for truth in truth_rows:
        dt_scale_ms = float(truth["time_reso_seconds"]) * 1e3
        tol_samples = max(1, int(round(source_time_tolerance_ms / dt_scale_ms)))
        if event_match_to_truth(event, truth, source_dm_tolerance, tol_samples) is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# 真值派生量 / 误报参数归因
# ---------------------------------------------------------------------------

def add_derived_truth_metrics(row: dict) -> dict:
    """给真值行补上搜索相关的派生幅度（有效通道数、每通道峰值 S/N）。"""
    out = dict(row)
    nchan = int(float(out.get("nchan", 4096)))
    channel_width_mhz = FAST_BANDWIDTH_MHZ / max(1, nchan)
    effective_channels = max(1.0, float(out["bandwidth_mhz_fwhm"]) / channel_width_mhz)
    out["effective_channels_fwhm"] = effective_channels
    out["per_channel_peak_snr"] = float(out["snr"]) / np.sqrt(effective_channels)
    return out


def annotate_false_positive_cell(cand: dict, truth_rows: list[dict]) -> dict:
    """把 unmatched candidate 归到最近注入源的参数 cell。

    false positive 本身没有 S/N、宽度等真值参数。为了画 S/N-参数空间的
    precision proxy，这里只做诊断性归因：按全局 TOA 和 DM 找最近的注入源，
    把该源参数写到 assigned_* 字段。整体 precision 仍以所有 unmatched
    candidates 计数；cell precision 用这些 assigned_* 字段分摊误报。
    """
    out = dict(cand)
    if not truth_rows:
        return out

    cand_time = int(float(out.get("_pred_toa_global_raw_sample", 0)))
    cand_dm = float(out.get("pred_dm_pc_cm3", 0.0))
    best_truth = None
    best_rank = None
    for truth in truth_rows:
        time_reso_ms = float(truth["time_reso_seconds"]) * 1e3
        dt_ms = (cand_time - int(float(truth["highest_freq_toa_global_raw_sample"]))) * time_reso_ms
        dm_error = cand_dm - float(truth["dm_pc_cm3"])
        rank = abs(dt_ms) / 1000.0 + abs(dm_error) / 100.0
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_truth = truth
            best_dt_ms = dt_ms
            best_dm_error = dm_error

    if best_truth is None:
        return out
    out["assigned_injection_id"] = best_truth.get("injection_id", "")
    out["assigned_run_label"] = best_truth.get("run_label", "")
    out["assigned_quantization"] = best_truth.get("quantization", "")
    out["assigned_toa_delta_ms"] = best_dt_ms
    out["assigned_dm_delta_pc_cm3"] = best_dm_error
    for key in (
        "snr",
        "dm_pc_cm3",
        "width_ms_fwhm",
        "bandwidth_mhz_fwhm",
        "scattering_ms_at_1ghz",
        "scattering_ms_at_highest_freq",
        "center_freq_mhz",
        "effective_channels_fwhm",
        "per_channel_peak_snr",
    ):
        out[f"assigned_{key}"] = best_truth.get(key, "")
    return out


# ---------------------------------------------------------------------------
# 核心：候选 ↔ 真值匹配，产出 matches / false_positives
# ---------------------------------------------------------------------------

def match_candidates(
    truth_rows: list[dict],
    candidate_rows: list[dict],
    dm_tolerance: float,
    time_tolerance_ms: float,
    event_dedup_dm_tolerance: float | None = None,
    event_dedup_time_tolerance_ms: float | None = None,
    source_dm_tolerance: float | None = None,
    source_time_tolerance_ms: float | None = None,
) -> tuple[list[dict], list[dict], int]:
    source_dm_tol, source_time_ms = normalize_source_tolerances(
        dm_tolerance,
        time_tolerance_ms,
        source_dm_tolerance,
        source_time_tolerance_ms,
    )

    # 1) 把每个候选放到全局时间轴上：_pred_toa_global_raw_sample
    stem_to_index = {
        str(row["output_file_stem"]): int(float(row["output_file_index"]))
        for row in truth_rows
        if "output_file_stem" in row and "output_file_index" in row
    }
    file_number_base = infer_file_number_base(truth_rows, candidate_rows)
    candidates_with_global = []
    for index, cand in enumerate(candidate_rows):
        file_index = candidate_file_index(cand, file_number_base, stem_to_index)
        if file_index is None:
            continue
        n_time = int(float(truth_rows[0]["rows_per_file"])) * int(float(truth_rows[0]["nsblk"])) if truth_rows else 0
        cand = {
            **cand,
            "_candidate_file_index": file_index,
            "_pred_toa_global_raw_sample": file_index * n_time + candidate_time_raw(cand),
        }
        candidates_with_global.append((index, cand))

    # 2) 把候选聚类成检测事件（去重容差默认等于匹配容差）
    if truth_rows:
        dt_scale_ms0 = float(truth_rows[0]["time_reso_seconds"]) * 1e3
    else:
        dt_scale_ms0 = 1.0
    event_dm_tol = dm_tolerance if event_dedup_dm_tolerance is None else event_dedup_dm_tolerance
    event_time_ms = time_tolerance_ms if event_dedup_time_tolerance_ms is None else event_dedup_time_tolerance_ms
    event_time_samples = max(1, int(round(event_time_ms / dt_scale_ms0)))
    candidate_events = build_candidate_events(candidates_with_global, event_dm_tol, event_time_samples)

    # 3) 构建 truth-event 合法边，先最大化命中数，再最小化总关联代价。
    derived_truths = [add_derived_truth_metrics(truth) for truth in truth_rows]
    match_edges: list[tuple[int, int, float]] = []
    edge_payload: dict[tuple[int, int], tuple[dict, float, int]] = {}
    for truth_index, truth in enumerate(derived_truths):
        dt_scale_ms = float(truth["time_reso_seconds"]) * 1e3
        source_tol_samples = max(1, int(round(source_time_ms / dt_scale_ms)))
        for event_index, event in enumerate(candidate_events):
            event_match = event_match_to_truth(
                event, truth, source_dm_tol, source_tol_samples
            )
            if event_match is None:
                continue
            rank, cand, dm_error, time_error_samples = event_match
            match_edges.append((truth_index, event_index, rank))
            edge_payload[(truth_index, event_index)] = (
                cand,
                dm_error,
                time_error_samples,
            )

    assignments = maximum_cardinality_min_cost_matching(
        len(derived_truths), len(candidate_events), match_edges
    )
    matches = []
    for truth_index, truth in enumerate(derived_truths):
        dt_scale_ms = float(truth["time_reso_seconds"]) * 1e3
        strict_tol_samples = max(1, int(round(time_tolerance_ms / dt_scale_ms)))
        event_index = assignments.get(truth_index)
        if event_index is None:
            matches.append({
                **truth,
                "detected": False,
                "event_id": "",
                "event_size": "",
                "candidate_index": "",
                "pred_dm_pc_cm3": "",
                "dm_error_pc_cm3": "",
                "pred_toa_raw_sample": "",
                "pred_toa_global_raw_sample": "",
                "candidate_file_index": "",
                "toa_error_ms": "",
                "localized_within_tolerance": "",
                "detector_score": "",
                "classifier_score": "",
                "frequency_split": "",
                "plot_path": "",
            })
        else:
            event = candidate_events[event_index]
            cand, dm_error, time_error_samples = edge_payload[
                (truth_index, event_index)
            ]
            localized = abs(dm_error) <= dm_tolerance and abs(time_error_samples) <= strict_tol_samples
            matches.append({
                **truth,
                "detected": True,
                "event_id": event.event_id,
                "event_size": event.size,
                "candidate_index": cand.get("candidate_index", ""),
                "pred_dm_pc_cm3": cand.get("pred_dm_pc_cm3", ""),
                "dm_error_pc_cm3": dm_error,
                "pred_toa_raw_sample": cand.get("pred_toa_raw_sample", ""),
                "pred_toa_global_raw_sample": cand.get("_pred_toa_global_raw_sample", ""),
                "candidate_file_index": cand.get("_candidate_file_index", ""),
                "toa_error_ms": time_error_samples * dt_scale_ms,
                "localized_within_tolerance": localized,
                "detector_score": cand.get("detector_score", ""),
                "classifier_score": cand.get("classifier_score", ""),
                "frequency_split": cand.get("frequency_split", ""),
                "plot_path": cand.get("plot_path", ""),
            })

    # 4) 剩余事件若不靠近任何注入源 → 误报
    false_positives = []
    for event in candidate_events:
        # 事件只要能落到任意注入源的匹配容差内，就不是判断错误；它可能只是
        # 未被选中的重复切片/重复 DM chunk 候选。
        if event_near_any_truth(event, truth_rows, source_dm_tol, source_time_ms):
            continue
        row = annotate_false_positive_cell(event.representative, truth_rows)
        row["event_id"] = event.event_id
        row["event_size"] = event.size
        row["event_candidate_indices"] = ",".join(
            str(member.get("candidate_index", "")) for member in event.members
        )
        false_positives.append(row)
    return matches, false_positives, len(candidate_events)


# ---------------------------------------------------------------------------
# 分箱统计与绘图
# ---------------------------------------------------------------------------

def bin_summary(rows: list[dict], key: str, bins: list[float]) -> list[dict]:
    """按某参数分箱统计 recall（n_detected / n_injected）。"""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        subset = [
            row for row in rows
            if key in row
            and row.get(key, "") != ""
            and (lo <= float(row[key]) < hi or (hi == bins[-1] and lo <= float(row[key]) <= hi))
        ]
        detected = sum(1 for row in subset if row["detected"])
        total = len(subset)
        out.append({
            "parameter": key,
            "bin_low": lo,
            "bin_high": hi,
            "n_injected": total,
            "n_detected": detected,
            "recall": detected / total if total else "",
        })
    return out


def plot_recall_curve(rows: list[dict], key: str, bins: list[float], output: Path, xlabel: str) -> None:
    """画单参数的 recall 曲线，点上标注每箱注入数。"""
    summary = bin_summary(rows, key, bins)
    centers = [(row["bin_low"] + row["bin_high"]) / 2.0 for row in summary]
    recall = [np.nan if row["recall"] == "" else float(row["recall"]) for row in summary]
    counts = [row["n_injected"] for row in summary]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(centers, recall, marker="o", color="#1f77b4")
    for x, y, n in zip(centers, recall, counts):
        if np.isfinite(y):
            ax.text(x, min(1.03, y + 0.04), str(n), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Recall")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_snr_dm_heatmap(rows: list[dict], output: Path) -> None:
    """画 S/N × DM 的二维 recall 热图。"""
    snr_bins = [3, 5, 8, 12, 20, 40, 100]
    dm_bins = [100, 500, 1000, 1500, 2000]
    heat = np.full((len(dm_bins) - 1, len(snr_bins) - 1), np.nan)
    counts = np.zeros_like(heat, dtype=int)
    for i, dm_lo in enumerate(dm_bins[:-1]):
        dm_hi = dm_bins[i + 1]
        for j, snr_lo in enumerate(snr_bins[:-1]):
            snr_hi = snr_bins[j + 1]
            subset = [
                row for row in rows
                if dm_lo <= float(row["dm_pc_cm3"]) <= dm_hi and snr_lo <= float(row["snr"]) <= snr_hi
            ]
            counts[i, j] = len(subset)
            if subset:
                heat[i, j] = sum(1 for row in subset if row["detected"]) / len(subset)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    image = ax.imshow(heat, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(snr_bins) - 1), [f"{snr_bins[i]}-{snr_bins[i+1]}" for i in range(len(snr_bins) - 1)])
    ax.set_yticks(range(len(dm_bins) - 1), [f"{dm_bins[i]}-{dm_bins[i+1]}" for i in range(len(dm_bins) - 1)])
    ax.set_xlabel("Nominal injected S/N")
    ax.set_ylabel("DM (pc cm$^{-3}$)")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            label = "n=0" if counts[i, j] == 0 else f"{heat[i, j]:.2f}\nn={counts[i, j]}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if np.nan_to_num(heat[i, j]) < 0.6 else "black")
    fig.colorbar(image, ax=ax, label="Recall")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DRAFTS injection-test candidates.")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dm-tolerance", type=float, default=25.0)
    parser.add_argument("--time-tolerance-ms", type=float, default=30.0)
    parser.add_argument(
        "--source-dm-tolerance",
        type=float,
        default=DEFAULT_SOURCE_DM_TOLERANCE,
        help=(
            "Source-level DM association tolerance for recall/FP suppression. "
            "Strict localization is still reported with --dm-tolerance."
        ),
    )
    parser.add_argument(
        "--source-time-tolerance-ms",
        type=float,
        default=None,
        help=(
            "Source-level time association tolerance for recall/FP suppression; "
            "defaults to --time-tolerance-ms."
        ),
    )
    parser.add_argument(
        "--event-dedup-dm-tolerance",
        type=float,
        default=None,
        help="DM tolerance used to merge slice/chunk candidates into one event; defaults to --dm-tolerance.",
    )
    parser.add_argument(
        "--event-dedup-time-tolerance-ms",
        type=float,
        default=None,
        help="Time tolerance used to merge slice/chunk candidates into one event; defaults to --time-tolerance-ms.",
    )
    args = parser.parse_args()
    source_dm_tolerance, source_time_tolerance_ms = normalize_source_tolerances(
        args.dm_tolerance,
        args.time_tolerance_ms,
        args.source_dm_tolerance,
        args.source_time_tolerance_ms,
    )

    # 读入真值与全部 section 的候选
    truth_rows = read_jsonl(args.truth)
    candidate_rows = []
    for path in candidate_paths(args.candidates):
        candidate_rows.extend(read_jsonl(path))

    matches, false_positives, candidate_event_count = match_candidates(
        truth_rows,
        candidate_rows,
        args.dm_tolerance,
        args.time_tolerance_ms,
        args.event_dedup_dm_tolerance,
        args.event_dedup_time_tolerance_ms,
        source_dm_tolerance,
        source_time_tolerance_ms,
    )
    detected = sum(1 for row in matches if row["detected"])
    localized = sum(1 for row in matches if row.get("localized_within_tolerance") is True)
    total = len(matches)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 逐注入的匹配明细
    match_fields = [
        "injection_id", "run_label", "quantization", "detected", "snr", "dm_pc_cm3",
        "width_ms_fwhm", "bandwidth_mhz_fwhm", "scattering_ms_at_1ghz",
        "scattering_ms_at_highest_freq",
        "center_freq_mhz", "effective_channels_fwhm", "per_channel_peak_snr",
        "highest_freq_toa_file_raw_sample", "highest_freq_toa_global_raw_sample",
        "event_id", "event_size", "pred_toa_raw_sample", "pred_toa_global_raw_sample", "candidate_file_index",
        "toa_error_ms", "pred_dm_pc_cm3", "dm_error_pc_cm3", "localized_within_tolerance", "detector_score",
        "classifier_score", "frequency_split", "output_file_stem", "plot_path",
    ]
    write_csv(args.output_dir / "matches.csv", matches, match_fields)
    write_csv(args.output_dir / "false_positives.csv", false_positives, sorted({key for row in false_positives for key in row}))

    # 按各物理参数分箱的 recall
    bin_specs = {
        "snr": [3, 5, 8, 12, 20, 40, 100],
        "dm_pc_cm3": [100, 500, 1000, 1500, 2000],
        "width_ms_fwhm": [0.5, 1, 2, 5, 10, 20],
        "bandwidth_mhz_fwhm": [50, 100, 170, 300, 500],
        "scattering_ms_at_1ghz": [0, 0.03, 0.1, 0.3, 1, 3, 10, 20],
        "scattering_ms_at_highest_freq": [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 5],
        "per_channel_peak_snr": [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.5],
    }
    all_bins = []
    for key, bins in bin_specs.items():
        rows = bin_summary(matches, key, bins)
        all_bins.extend(rows)
        write_csv(args.output_dir / f"recall_by_{key}.csv", rows, ["parameter", "bin_low", "bin_high", "n_injected", "n_detected", "recall"])
    write_csv(args.output_dir / "recall_by_parameter_bins.csv", all_bins, ["parameter", "bin_low", "bin_high", "n_injected", "n_detected", "recall"])

    # recall 曲线 + S/N-DM 热图
    plot_recall_curve(matches, "snr", bin_specs["snr"], args.output_dir / "recall_by_snr.png", "Nominal injected S/N")
    plot_recall_curve(matches, "dm_pc_cm3", bin_specs["dm_pc_cm3"], args.output_dir / "recall_by_dm.png", "DM (pc cm$^{-3}$)")
    plot_recall_curve(matches, "width_ms_fwhm", bin_specs["width_ms_fwhm"], args.output_dir / "recall_by_width.png", "Temporal FWHM (ms)")
    plot_recall_curve(matches, "scattering_ms_at_1ghz", bin_specs["scattering_ms_at_1ghz"], args.output_dir / "recall_by_scattering_1ghz.png", "Scattering at 1 GHz (ms)")
    plot_recall_curve(matches, "scattering_ms_at_highest_freq", bin_specs["scattering_ms_at_highest_freq"], args.output_dir / "recall_by_scattering_highest_freq.png", "Scattering at highest channel (ms)")
    plot_recall_curve(matches, "per_channel_peak_snr", bin_specs["per_channel_peak_snr"], args.output_dir / "recall_by_per_channel_peak_snr.png", "Peak S/N per FWHM-band channel")
    plot_snr_dm_heatmap(matches, args.output_dir / "recall_heatmap_snr_dm.png")

    # 汇总指标（recall / DM、TOA 误差分位 / 误报数）
    dm_errors = [float(row["dm_error_pc_cm3"]) for row in matches if row["detected"]]
    toa_errors = [float(row["toa_error_ms"]) for row in matches if row["detected"]]
    summary = {
        "truth_count": total,
        "candidate_count": len(candidate_rows),
        "candidate_event_count": candidate_event_count,
        "detected_count": detected,
        "recall": detected / total if total else 0.0,
        "localized_within_tolerance_count": localized,
        "localized_within_tolerance_fraction": localized / total if total else 0.0,
        "detected_count_snr_ge_10": sum(1 for row in matches if float(row["snr"]) >= 10.0 and row["detected"]),
        "truth_count_snr_ge_10": sum(1 for row in matches if float(row["snr"]) >= 10.0),
        "recall_snr_ge_10": (
            sum(1 for row in matches if float(row["snr"]) >= 10.0 and row["detected"])
            / max(1, sum(1 for row in matches if float(row["snr"]) >= 10.0))
        ),
        "detected_count_snr_ge_20": sum(1 for row in matches if float(row["snr"]) >= 20.0 and row["detected"]),
        "truth_count_snr_ge_20": sum(1 for row in matches if float(row["snr"]) >= 20.0),
        "recall_snr_ge_20": (
            sum(1 for row in matches if float(row["snr"]) >= 20.0 and row["detected"])
            / max(1, sum(1 for row in matches if float(row["snr"]) >= 20.0))
        ),
        "false_positive_count": len(false_positives),
        "false_positive_unit": "deduplicated_detection_event",
        "false_positive_definition": (
            "A post-classifier detection event is a false positive only if none of its "
            "slice/chunk candidates falls within the source-level DM/time association tolerance "
            "of any injected source."
        ),
        "dm_tolerance_pc_cm3": args.dm_tolerance,
        "time_tolerance_ms": args.time_tolerance_ms,
        "source_dm_tolerance_pc_cm3": source_dm_tolerance,
        "source_time_tolerance_ms": source_time_tolerance_ms,
        "event_dedup_dm_tolerance_pc_cm3": (
            args.dm_tolerance if args.event_dedup_dm_tolerance is None else args.event_dedup_dm_tolerance
        ),
        "event_dedup_time_tolerance_ms": (
            args.time_tolerance_ms
            if args.event_dedup_time_tolerance_ms is None
            else args.event_dedup_time_tolerance_ms
        ),
        "dm_error_abs_median_pc_cm3": float(np.median(np.abs(dm_errors))) if dm_errors else None,
        "dm_error_abs_p90_pc_cm3": float(np.percentile(np.abs(dm_errors), 90)) if dm_errors else None,
        "toa_error_abs_median_ms": float(np.median(np.abs(toa_errors))) if toa_errors else None,
        "toa_error_abs_p90_ms": float(np.percentile(np.abs(toa_errors), 90)) if toa_errors else None,
    }
    write_json(args.output_dir / "summary_metrics.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
