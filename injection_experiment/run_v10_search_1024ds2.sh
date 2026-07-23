#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f /path/to/miniforge3/etc/profile.d/conda.sh ]]; then
  # Remote pg13/pg15 environment used for the injection search.
  source /path/to/miniforge3/etc/profile.d/conda.sh
  conda activate pytorch
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate pytorch
fi

PY="${PY:-python}"
BASE_DIR="${BASE_DIR:-${SCRIPT_DIR}}"
WORK_ROOT="${WORK_ROOT:-${BASE_DIR}/runs}"
SIM_ROOT="${SIM_ROOT:-${BASE_DIR}/simdata}"
TRUTH_ROOT="${TRUTH_ROOT:-${BASE_DIR}/truth_archive}"
RUN_LABEL="${RUN_LABEL:-v10_det03_injection_10000}"
CAMPAIGN_DIR="${WORK_ROOT}/${RUN_LABEL}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_NUM="${GPU_NUM:-8}"
DET_PROB="${DET_PROB:-0.3}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-64}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "${WORK_ROOT}"

echo "[launch] run_label=${RUN_LABEL}"
echo "[launch] campaign_dir=${CAMPAIGN_DIR}"
echo "[launch] sim_root=${SIM_ROOT}"
echo "[launch] truth_root=${TRUTH_ROOT}"
echo "[launch] detector=v10 det_prob=${DET_PROB} classifier_input=1024x512"
echo "[launch] gpu_ids=${GPU_IDS} gpu_num=${GPU_NUM}"

"${PY}" run_injection_campaign.py \
  --work-root "${WORK_ROOT}" \
  --sim-root "${SIM_ROOT}" \
  --truth-root "${TRUTH_ROOT}" \
  --run-label "${RUN_LABEL}" \
  --batches "${BATCHES:-20}" \
  --count-per-batch "${COUNT_PER_BATCH:-500}" \
  --search-only \
  --overwrite-search \
  --runtime-dir "${SCRIPT_DIR}/search_runtime" \
  --gpu-num "${GPU_NUM}" \
  --gpu-ids "${GPU_IDS}" \
  --det-prob "${DET_PROB}" \
  --detector-type centernet_conv_tiny \
  --detector-ckpt models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  --classifier-ckpt models/binary_best_model_conv_small_ema.pth \
  --classifier-model-name convnext_small \
  --class-block-size 1024 \
  --class-time-downsample 2 \
  --classifier-batch-size "${CLASSIFIER_BATCH_SIZE}" \
  --source-dm-tolerance 60 \
  --source-time-tolerance-ms 30 \
  --poll-seconds "${POLL_SECONDS}"
