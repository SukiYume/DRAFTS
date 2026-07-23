#!/usr/bin/env bash
# Run t-blind-section.py across several datasets and GPU sections.
#
# Example, production-like CRAFTS search parameters:
#   ROOT=/path/to/drafts_runs/data_searching \
#   OUTPUT_ROOT=/path/to/drafts_runs/blind \
#   BEAM=all GPU_NUM=8 DM_THRESHOLD=10 BLOCK_SIZE=4096 DM_SPAN=1024 DET_PROB=0.40 \
#     bash t-blind-batch.sh /data31/ZD2024_1_1_2bit/
#
# This direct launcher is for development nodes. On mu01/gate nodes use
# s-pbsspt.py instead of running this script directly.
set -euo pipefail

ROOT="/path/to/drafts_runs/data_searching"
PY_ENV="${PY_ENV:-pytorch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
RUN_LABEL="${RUN_LABEL:-blind_centernet_conv_tiny_binary_conv_small_v10}"
LOG_ROOT="$ROOT/logs/${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)"
DETECTOR_TYPE="${DETECTOR_TYPE:-centernet_conv_tiny}"
DETECTOR_CKPT="${DETECTOR_CKPT:-models/object_best_model_centernet_conv_tiny_ema_v10.pth}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-models/binary_best_model_conv_small_ema.pth}"
BEAM="${BEAM:-M01}"
GPU_NUM="${GPU_NUM:-8}"
DM_RANGE="${DM_RANGE:-4096}"
DM_SCALE="${DM_SCALE:-1}"
DM_OFFSET="${DM_OFFSET:-0}"
DM_THRESHOLD="${DM_THRESHOLD:-50}"
BLOCK_SIZE="${BLOCK_SIZE:-8192}"
DM_SPAN="${DM_SPAN:-2048}"
DET_PROB="${DET_PROB:-0.45}"
TIME_FACTOR="${TIME_FACTOR:-8}"

if (($# > 0)); then
  DATASETS=("$@")
else
  DATASETS=(
    "/data31/ZD2023_5/FRB20220912A/20230926"
    "/data31/ZD2024_5/FRB20190417A/20250723"
  )
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

set +u
source ~/.bashrc
conda activate "$PY_ENV"
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "[Launch] root=$ROOT"
echo "[Launch] output=$OUTPUT_ROOT"
echo "[Launch] logs=$LOG_ROOT"
printf '[Launch] datasets=%s\n' "${DATASETS[@]}"
echo "[Launch] detector=$DETECTOR_TYPE ckpt=$DETECTOR_CKPT"
echo "[Launch] classifier_ckpt=$CLASSIFIER_CKPT"
echo "[Launch] beam=$BEAM gpu_num=$GPU_NUM"
echo "[Launch] dm_range=$DM_RANGE dm_scale=$DM_SCALE dm_offset=$DM_OFFSET dm_threshold=$DM_THRESHOLD block_size=$BLOCK_SIZE dm_span=$DM_SPAN time_factor=$TIME_FACTOR"
echo "[Launch] python=$(command -v python)"
python - <<'PY'
import torch
print(f"[Launch] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    print(f"[Launch] gpu0={torch.cuda.get_device_name(0)}", flush=True)
PY

for data_path in "${DATASETS[@]}"; do
  dataset_label="$(basename "$(dirname "$data_path")")_$(basename "$data_path")"
  for ((gpu = 0; gpu < GPU_NUM; gpu++)); do
    log_file="$LOG_ROOT/${dataset_label}_gpu${gpu}.log"
    echo "[Launch] dataset=$dataset_label gpu=$gpu log=$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/t-blind-section.py" \
      --section "$gpu" \
      --data-path "$data_path" \
      --output-root "$OUTPUT_ROOT" \
      --detector-type "$DETECTOR_TYPE" \
      --detector-ckpt "$DETECTOR_CKPT" \
      --classifier-ckpt "$CLASSIFIER_CKPT" \
      --beam "$BEAM" \
      --gpu-num "$GPU_NUM" \
      --dm-range "$DM_RANGE" \
      --dm-scale "$DM_SCALE" \
      --dm-offset "$DM_OFFSET" \
      --dm-threshold "$DM_THRESHOLD" \
      --block-size "$BLOCK_SIZE" \
      --dm-span "$DM_SPAN" \
      --det-prob "$DET_PROB" \
      --time-factor "$TIME_FACTOR" \
      >"$log_file" 2>&1 &
  done
done

wait
echo "[Done] all jobs completed"
