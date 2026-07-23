#!/usr/bin/env bash
# Run detector-only backend/model combinations across GPU sections.
#
# Example:
#   ROOT=/path/to/drafts_runs/data_searching \
#   DATA_PATH=/path/to/observations/source/date \
#   OUT_ROOT=/path/to/drafts_runs/object_backend \
#   GPU_NUM=8 BACKENDS="cupy" MODELS="centernet_conv_tiny" SAVE_PLOT=0 \
#     bash t-object-matrix.sh
set -Eeuo pipefail

ROOT="${ROOT:-/path/to/drafts_runs/runcode}"
DATA_PATH="${DATA_PATH:-/path/to/observations/source/date}"
OUT_ROOT="${OUT_ROOT:-/path/to/drafts_runs/object_backend}"
CONDA_SH="${CONDA_SH:-/path/to/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pytorch}"
BEAM="${BEAM:-M01}"
GPU_NUM="${GPU_NUM:-8}"
MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-4}"
SAVE_PLOT="${SAVE_PLOT:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-object_matrix_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/_logs/$RUN_LABEL"
SUMMARY_CSV="$LOG_DIR/summary.csv"

BACKENDS=(${BACKENDS:-cupy numba})
MODELS=(${MODELS:-centernet_conv_tiny})

if (( ${#BACKENDS[@]} * ${#MODELS[@]} > MAX_PROCS_PER_GPU )); then
  echo "[Error] combinations exceed MAX_PROCS_PER_GPU=${MAX_PROCS_PER_GPU}" >&2
  echo "        backends=${BACKENDS[*]} models=${MODELS[*]}" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$OUT_ROOT"
cd "$ROOT"

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
else
  # shellcheck disable=SC1090
  source ~/.bashrc
fi
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "[Launch] root=$ROOT"
echo "[Launch] data=$DATA_PATH"
echo "[Launch] out=$OUT_ROOT"
echo "[Launch] run_label=$RUN_LABEL"
echo "[Launch] log_dir=$LOG_DIR"
echo "[Launch] backends=${BACKENDS[*]}"
echo "[Launch] models=${MODELS[*]}"
echo "[Launch] gpu_num=$GPU_NUM max_procs_per_gpu=$MAX_PROCS_PER_GPU save_plot=$SAVE_PLOT dry_run=$DRY_RUN"
echo "[Launch] python=$(command -v python)"

python - <<'PY'
import torch
print(f"[Launch] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"[Launch] gpu{idx}={torch.cuda.get_device_name(idx)}", flush=True)
PY

echo "[Launch] matching FITS count:"
find "$DATA_PATH" -maxdepth 1 -type f -name "*-${BEAM}_*.fits" | sort | wc -l
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee "$LOG_DIR/nvidia-smi-start.txt"

COMMON_ARGS=(
  --data-path "$DATA_PATH"
  --output-root "$OUT_ROOT"
  --run-label "$RUN_LABEL"
  --beam "$BEAM"
  --gpu-num "$GPU_NUM"
)

if [[ "$SAVE_PLOT" == "0" ]]; then
  COMMON_ARGS+=(--no-save-plot)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

pids=()
labels=()

for ((section = 0; section < GPU_NUM; section++)); do
  gpu="$section"
  for backend in "${BACKENDS[@]}"; do
    for model in "${MODELS[@]}"; do
      label="s${section}_${backend}_${model}"
      log_file="$LOG_DIR/${label}.log"
      echo "[Launch] $label gpu=$gpu log=$log_file"
      CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/t-object-bench.py" \
        --backend "$backend" \
        --detector-type "$model" \
        --section "$section" \
        "${COMMON_ARGS[@]}" \
        >"$log_file" 2>&1 &
      pids+=("$!")
      labels+=("$label")
    done
  done
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  label="${labels[$idx]}"
  if wait "$pid"; then
    echo "[Done] $label"
  else
    rc=$?
    echo "[Fail] $label rc=$rc log=$LOG_DIR/${label}.log" >&2
    status=1
  fi
done

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee "$LOG_DIR/nvidia-smi-end.txt"

LOG_DIR="$LOG_DIR" SUMMARY_CSV="$SUMMARY_CSV" python - <<'PY'
import csv
import os
import re
from pathlib import Path

log_dir = Path(os.environ["LOG_DIR"])
summary_csv = Path(os.environ["SUMMARY_CSV"])
time_re = re.compile(
    r"\[Time\].*?load=(?P<load>[0-9.]+)s .*?"
    r"freq_ds\+norm=(?P<prepare>[0-9.]+)s "
    r"dedisp=(?P<dedisp>[0-9.]+)s "
    r"detect\+save=(?P<detect>[0-9.]+)s "
    r"total=(?P<total>[0-9.]+)s .*?bursts=(?P<bursts>\d+)"
)
done_re = re.compile(
    r"\[Done\] total_bursts=(?P<total_bursts>\d+) "
    r"total_time=(?P<total_time>[0-9.]+)s "
    r"avg_per_block=(?P<avg_per_block>[0-9.]+)s"
)
name_re = re.compile(r"s(?P<section>\d+)_(?P<backend>cupy|numba)_(?P<model>.+)\.log$")

rows = []
for log in sorted(log_dir.glob("s*_*.log")):
    m_name = name_re.match(log.name)
    if not m_name:
        continue
    metrics = {
        "section": m_name.group("section"),
        "backend": m_name.group("backend"),
        "model": m_name.group("model"),
        "log": log.name,
        "blocks": 0,
        "load_sum": 0.0,
        "prepare_sum": 0.0,
        "dedisp_sum": 0.0,
        "detect_sum": 0.0,
        "block_total_sum": 0.0,
        "block_bursts": 0,
        "total_bursts": "",
        "total_time": "",
        "avg_per_block": "",
        "status": "missing_done",
    }
    text = log.read_text(errors="replace")
    if "Traceback" in text or "[Fail]" in text:
        metrics["status"] = "failed"
    for line in text.splitlines():
        m = time_re.search(line)
        if m:
            metrics["blocks"] += 1
            metrics["load_sum"] += float(m.group("load"))
            metrics["prepare_sum"] += float(m.group("prepare"))
            metrics["dedisp_sum"] += float(m.group("dedisp"))
            metrics["detect_sum"] += float(m.group("detect"))
            metrics["block_total_sum"] += float(m.group("total"))
            metrics["block_bursts"] += int(m.group("bursts"))
        m = done_re.search(line)
        if m:
            metrics["total_bursts"] = int(m.group("total_bursts"))
            metrics["total_time"] = float(m.group("total_time"))
            metrics["avg_per_block"] = float(m.group("avg_per_block"))
            metrics["status"] = "ok"
    blocks = max(metrics["blocks"], 1)
    for key in ("load", "prepare", "dedisp", "detect", "block_total"):
        metrics[f"{key}_avg"] = metrics[f"{key}_sum"] / blocks
    rows.append(metrics)

fieldnames = [
    "section", "backend", "model", "status", "blocks",
    "total_time", "avg_per_block", "total_bursts", "block_bursts",
    "load_avg", "prepare_avg", "dedisp_avg", "detect_avg", "block_total_avg",
    "log",
]
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})

print(f"[Summary] wrote {summary_csv}")
for key in sorted({(r["backend"], r["model"]) for r in rows}):
    subset = [r for r in rows if (r["backend"], r["model"]) == key and r["status"] == "ok"]
    if not subset:
        print(f"[Summary] {key[0]} {key[1]} no ok sections")
        continue
    max_time = max(float(r["total_time"]) for r in subset)
    avg_block = sum(float(r["avg_per_block"]) for r in subset) / len(subset)
    avg_dedisp = sum(float(r["dedisp_avg"]) for r in subset) / len(subset)
    avg_detect = sum(float(r["detect_avg"]) for r in subset) / len(subset)
    print(
        f"[Summary] {key[0]} {key[1]} sections={len(subset)} "
        f"wall_est={max_time:.1f}s avg_block={avg_block:.2f}s "
        f"avg_dedisp={avg_dedisp:.2f}s avg_detect={avg_detect:.2f}s"
    )
PY

exit "$status"
