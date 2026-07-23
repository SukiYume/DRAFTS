#!/usr/bin/env bash
# Compare fixed-DM binary classifiers on a small curated set.
#
# Examples:
#   bash t-binary-bench.sh
#   bash t-binary-bench.sh --summarize-only
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-/path/to/drafts_runs/binary_output}"
CONDA_SH="${CONDA_SH:-/path/to/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pytorch}"
PROB="${PROB:-0.5}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
fi
conda activate "$CONDA_ENV"
PYTHON_BIN="$(command -v python)"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUT_ROOT/_logs/$RUN_ID"
mkdir -p "$LOG_DIR"

summarize_results() {
  ROOT_DIR="$ROOT_DIR" OUT_ROOT="$OUT_ROOT" "$PYTHON_BIN" - <<'PY'
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

out_root = Path(__import__("os").environ["OUT_ROOT"])

models = ["binary_conv_tiny", "binary_conv_small"]
observations = {
    "FRB180301A_20260430_M01": {
        "baseline_root": Path("/path/to/observations/FRB180301A/CalData/20260430/M01"),
        "rates": ["0008"],
    },
    "FRB20220912A_20230926_M01": {
        "baseline_root": Path("/path/to/observations/FRB20220912A/CalData/20230926/M01"),
        "rates": ["0008"],
    },
    "CHIMEJ0630+25_20260214_M01": {
        "baseline_root": Path("/path/to/observations/CHIMEJ0630+25/CalData/20260214/M01"),
        "rates": ["0002", "0004", "0008", "0016", "0032", "0064", "0128", "0256", "0512", "1024"],
    },
}

def parse_npy(path: Path):
    rate = path.parent.name
    stem = path.stem
    m = re.match(r"(.+)-(\d+)$", stem)
    if not m:
        return None
    return rate, m.group(1), int(m.group(2))

def load_keys(root: Path, rates):
    keys = set()
    by_rate = defaultdict(int)
    if not root.exists():
        return keys, by_rate
    for rate in rates:
        rate_dir = root / rate
        if not rate_dir.exists():
            continue
        for npy in rate_dir.glob("*.npy"):
            key = parse_npy(npy)
            if key is None:
                continue
            keys.add(key)
            by_rate[rate] += 1
    return keys, by_rate

def exact_metrics(pred, ref):
    hit = len(pred & ref)
    precision = hit / len(pred) if pred else 0.0
    recall = hit / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pred_total": len(pred),
        "ref_total": len(ref),
        "matches": hit,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def tolerant_match_count(pred, ref, tolerance=1):
    ref_by_group = defaultdict(list)
    for rate, file_id, block in ref:
        ref_by_group[(rate, file_id)].append(block)
    for group in ref_by_group:
        ref_by_group[group].sort()
    used = defaultdict(set)
    matches = 0
    for rate, file_id, block in sorted(pred):
        group = (rate, file_id)
        candidates = [
            rb for rb in ref_by_group.get(group, [])
            if rb not in used[group] and abs(rb - block) <= tolerance
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda rb: (abs(rb - block), rb))
        used[group].add(best)
        matches += 1
    return matches

def tolerant_metrics(pred, ref, tolerance=1):
    hit = tolerant_match_count(pred, ref, tolerance=tolerance)
    precision = hit / len(pred) if pred else 0.0
    recall = hit / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pred_total": len(pred),
        "ref_total": len(ref),
        "matches": hit,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

summary = {"counts": [], "comparisons": []}
keys = {}

for obs, meta in observations.items():
    rates = meta["rates"]
    baseline_keys, baseline_by_rate = load_keys(meta["baseline_root"], rates)
    keys[("baseline", obs)] = baseline_keys
    summary["counts"].append({
        "model": "baseline",
        "observation": obs,
        "total": len(baseline_keys),
        "by_rate": dict(sorted(baseline_by_rate.items())),
        "root": str(meta["baseline_root"]),
    })
    for model in models:
        root = out_root / model / obs
        model_keys, model_by_rate = load_keys(root, rates)
        keys[(model, obs)] = model_keys
        summary["counts"].append({
            "model": model,
            "observation": obs,
            "total": len(model_keys),
            "by_rate": dict(sorted(model_by_rate.items())),
            "root": str(root),
        })

for obs in observations:
    pairs = [
        ("binary_conv_tiny", "binary_conv_small"),
        ("binary_conv_small", "binary_conv_tiny"),
        ("binary_conv_tiny", "baseline"),
        ("binary_conv_small", "baseline"),
    ]
    for pred_name, ref_name in pairs:
        pred = keys[(pred_name, obs)]
        ref = keys[(ref_name, obs)]
        row = {
            "observation": obs,
            "prediction": pred_name,
            "reference": ref_name,
            "exact": exact_metrics(pred, ref),
            "block_pm1": tolerant_metrics(pred, ref, tolerance=1),
        }
        summary["comparisons"].append(row)

out_root.mkdir(parents=True, exist_ok=True)
summary_json = out_root / "benchmark_summary.json"
summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

counts_csv = out_root / "benchmark_counts.csv"
with counts_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "observation", "total", "by_rate", "root"])
    writer.writeheader()
    for row in summary["counts"]:
        writer.writerow({
            **row,
            "by_rate": json.dumps(row["by_rate"], sort_keys=True),
        })

comparisons_csv = out_root / "benchmark_comparisons.csv"
with comparisons_csv.open("w", newline="", encoding="utf-8") as f:
    fields = [
        "observation", "prediction", "reference",
        "exact_pred_total", "exact_ref_total", "exact_matches", "exact_precision", "exact_recall", "exact_f1",
        "pm1_pred_total", "pm1_ref_total", "pm1_matches", "pm1_precision", "pm1_recall", "pm1_f1",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in summary["comparisons"]:
        exact = row["exact"]
        pm1 = row["block_pm1"]
        writer.writerow({
            "observation": row["observation"],
            "prediction": row["prediction"],
            "reference": row["reference"],
            "exact_pred_total": exact["pred_total"],
            "exact_ref_total": exact["ref_total"],
            "exact_matches": exact["matches"],
            "exact_precision": f"{exact['precision']:.6f}",
            "exact_recall": f"{exact['recall']:.6f}",
            "exact_f1": f"{exact['f1']:.6f}",
            "pm1_pred_total": pm1["pred_total"],
            "pm1_ref_total": pm1["ref_total"],
            "pm1_matches": pm1["matches"],
            "pm1_precision": f"{pm1['precision']:.6f}",
            "pm1_recall": f"{pm1['recall']:.6f}",
            "pm1_f1": f"{pm1['f1']:.6f}",
        })

print(f"Wrote {summary_json}")
print(f"Wrote {counts_csv}")
print(f"Wrote {comparisons_csv}")
print("\nCounts:")
for row in summary["counts"]:
    print(f"  {row['model']:18s} {row['observation']:30s} total={row['total']:4d} by_rate={row['by_rate']}")
print("\nComparisons, exact / block_pm1:")
for row in summary["comparisons"]:
    exact = row["exact"]
    pm1 = row["block_pm1"]
    print(
        f"  {row['observation']:30s} {row['prediction']:18s} vs {row['reference']:18s} "
        f"exact P={exact['precision']:.3f} R={exact['recall']:.3f} M={exact['matches']}; "
        f"pm1 P={pm1['precision']:.3f} R={pm1['recall']:.3f} M={pm1['matches']}"
    )
PY
}

if [[ "${1:-}" == "--summarize-only" ]]; then
  summarize_results
  exit 0
fi

run_task() {
  local gpu="$1"
  local model_label="$2"
  local model_name="$3"
  local ckpt="$4"
  local obs_id="$5"
  local data_path="$6"
  local dm="$7"
  local rates_csv="$8"
  local beam="$9"

  local out_dir="$OUT_ROOT/$model_label/$obs_id"
  local log_file="$LOG_DIR/${model_label}__${obs_id}.log"
  local status_file="$LOG_DIR/${model_label}__${obs_id}.status"

  if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    case "$out_dir" in
      "$OUT_ROOT"/binary_conv_tiny/*|"$OUT_ROOT"/binary_conv_small/*)
        rm -rf "$out_dir"
        ;;
      *)
        echo "Refusing to clean suspicious output path: $out_dir" >&2
        exit 2
        ;;
    esac
  fi
  mkdir -p "$out_dir"

  (
    set +e
    local start_ts end_ts rc
    start_ts="$(date +%s)"
    echo "START $(date -Is)" > "$status_file"
    echo "gpu=$gpu model=$model_label obs=$obs_id dm=$dm rates=$rates_csv out=$out_dir" >> "$status_file"

    CUDA_VISIBLE_DEVICES="$gpu" \
    ROOT_DIR="$ROOT_DIR" MODEL_LABEL="$model_label" MODEL_NAME="$model_name" CKPT="$ckpt" \
    OBS_ID="$obs_id" DATA_PATH="$data_path" DM_VALUE="$dm" RATES_CSV="$rates_csv" BEAM="$beam" \
    OUT_DIR="$out_dir" PROB="$PROB" \
    "$PYTHON_BIN" -u - <<'PY' > "$log_file" 2>&1
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

root_dir = Path(os.environ["ROOT_DIR"])
module_path = root_dir / "d-dm-time-predown.py"
spec = importlib.util.spec_from_file_location("d_dm_time_predown", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to import {module_path}")
predown = importlib.util.module_from_spec(spec)
sys.modules["d_dm_time_predown"] = predown
spec.loader.exec_module(predown)

model_label = os.environ["MODEL_LABEL"]
model_name = os.environ["MODEL_NAME"]
ckpt = os.environ["CKPT"]
obs_id = os.environ["OBS_ID"]
data_path = os.environ["DATA_PATH"]
beam = os.environ["BEAM"]
out_dir = Path(os.environ["OUT_DIR"])
rates = np.array([int(x) for x in os.environ["RATES_CSV"].split(",") if x], dtype=int)
dm = float(os.environ["DM_VALUE"])
prob = float(os.environ["PROB"])

print(json.dumps({
    "event": "task_start",
    "model_label": model_label,
    "model_name": model_name,
    "ckpt": ckpt,
    "obs_id": obs_id,
    "data_path": data_path,
    "beam": beam,
    "dm": dm,
    "rates": rates.tolist(),
    "prob": prob,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_device_count": torch.cuda.device_count(),
}, indent=2))

model = predown.load_binary_classifier(
    ckpt,
    model_type="ConvNeXtNet",
    model_name=model_name,
)
config = predown.ProcessConfig(
    DM=dm,
    prob=prob,
    block_size=512,
    gpu_num=1,
    down_sampling_rate_list=rates,
)

file_lists = predown.organize_file_lists(data_path, beam)
print(f"organized_file_lists={len(file_lists)}")
if not file_lists:
    raise RuntimeError(f"No FITS files found for data_path={data_path!r}, beam={beam!r}")

started = time.time()
for identifier, fits_list, info in file_lists:
    print(f"processing identifier={identifier} files={len(fits_list)} info={info}")
    predown.process_fits_list(fits_list, model, config, str(out_dir) + "/")

counts = {}
for rate in rates:
    rate_dir = out_dir / f"{int(rate):04d}"
    counts[f"{int(rate):04d}"] = {
        "npy": len(list(rate_dir.glob("*.npy"))) if rate_dir.exists() else 0,
        "jpg": len(list(rate_dir.glob("*.jpg"))) if rate_dir.exists() else 0,
    }

summary = {
    "model_label": model_label,
    "model_name": model_name,
    "obs_id": obs_id,
    "data_path": data_path,
    "beam": beam,
    "dm": dm,
    "rates": rates.tolist(),
    "prob": prob,
    "elapsed_sec": time.time() - started,
    "counts": counts,
}
(out_dir / "task_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"event": "task_done", **summary}, indent=2))
PY
    rc=$?
    end_ts="$(date +%s)"
    {
      echo "END $(date -Is)"
      echo "exit_code=$rc"
      echo "elapsed_sec=$((end_ts - start_ts))"
      echo "log=$log_file"
    } >> "$status_file"
    exit "$rc"
  ) &
  local pid="$!"
  PIDS+=("$pid")
  TASK_NAMES+=("$model_label/$obs_id")
}

declare -a PIDS=()
declare -a TASK_NAMES=()

TINY_CKPT="$ROOT_DIR/models/binary_best_model_conv_tiny_ema.pth"
SMALL_CKPT="$ROOT_DIR/models/binary_best_model_conv_small_ema.pth"

run_task 0 binary_conv_tiny  convnext_tiny  "$TINY_CKPT"  observation_a /path/to/observations/source_a/date 510 8 M01
run_task 1 binary_conv_tiny  convnext_tiny  "$TINY_CKPT"  observation_b /path/to/observations/source_b/date 220 8 M01
run_task 2 binary_conv_tiny  convnext_tiny  "$TINY_CKPT"  observation_c /path/to/observations/source_c/date 22  2,4,8,16,32,64,128,256,512,1024 M01

run_task 3 binary_conv_small convnext_small "$SMALL_CKPT" observation_a /path/to/observations/source_a/date 510 8 M01
run_task 4 binary_conv_small convnext_small "$SMALL_CKPT" observation_b /path/to/observations/source_b/date 220 8 M01
run_task 5 binary_conv_small convnext_small "$SMALL_CKPT" observation_c /path/to/observations/source_c/date 22  2,4,8,16,32,64,128,256,512,1024 M01

printf '%s\n' "${PIDS[@]}" > "$LOG_DIR/pids.txt"
printf '%s\n' "${TASK_NAMES[@]}" > "$LOG_DIR/tasks.txt"

echo "Started ${#PIDS[@]} tasks. Log dir: $LOG_DIR"
for idx in "${!PIDS[@]}"; do
  echo "  pid=${PIDS[$idx]} task=${TASK_NAMES[$idx]}"
done

failed=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    echo "DONE ${TASK_NAMES[$idx]}"
  else
    rc=$?
    echo "FAILED ${TASK_NAMES[$idx]} exit=$rc" >&2
    failed=1
  fi
done

summarize_results | tee "$LOG_DIR/summary.log"

if [[ "$failed" != "0" ]]; then
  echo "One or more tasks failed. See $LOG_DIR" >&2
  exit 1
fi

echo "All tasks finished. Summary: $OUT_ROOT/benchmark_summary.json"
