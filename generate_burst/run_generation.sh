#!/usr/bin/env bash
set -euo pipefail

# 全部路径以脚本所在目录为根，自包含。把整个 gendata 目录拷到哪都能直接跑：
#   gendata/ 下需有：本脚本、generate_dataset.py、merge_shards.py、
#                    inspect_dataset.py、simulation_utils.py、d-center-binary-core.py、
#                    rawdata/（背景 FITS）。中间产物/输出也都写在 gendata 下。
# 需要时可用环境变量覆盖任意一项（如 GEN_ROOT=/somewhere RAW_DIR=... PY=python3）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_ROOT="${GEN_ROOT:-${SCRIPT_DIR}}"
RAW_DIR="${RAW_DIR:-${GEN_ROOT}/rawdata}"
# simulation_utils.py、d-center-binary-core.py 固定与生成脚本同目录，不再传 --injection-dir / --runtime-dir。
BASE_DIR="${GEN_ROOT}/shards_50000"
FINAL_H5="${GEN_ROOT}/centernet_dataset_sim50000_max3.h5"
PY="${PY:-python}"                                   # 用当前环境的 python；可用 PY=... 覆盖

# 50000 唯一信号 x 4 crops = 200000 张图
TOTAL_UNIQUE=50000
SHARD_COUNT=40
# A full-generation benchmark found that four concurrent shards avoids FITS I/O thrash.
SHARDS_PER_WAVE="${SHARDS_PER_WAVE:-4}"
UNIQUE_PER_SHARD=$((TOTAL_UNIQUE / SHARD_COUNT))
CROPS_PER_SIGNAL=4
SIGNALS_PER_SCENE=3
MAX_OBJECTS_PER_IMAGE=3
SIGNALS_PER_BATCH_IMAGES=2000
STD_CACHE="${GEN_ROOT}/channel_std_raw8_8f_16384.npy"
MIN_SPLIT_EFFECTIVE_SNR=1.5
PHYSICAL_FALLBACK_MIN_SNR=3.0

if (( TOTAL_UNIQUE % SHARD_COUNT != 0 )); then
  echo "TOTAL_UNIQUE=${TOTAL_UNIQUE} must be divisible by SHARD_COUNT=${SHARD_COUNT}" >&2
  exit 1
fi
if (( SHARDS_PER_WAVE < 1 )); then
  echo "SHARDS_PER_WAVE=${SHARDS_PER_WAVE} must be positive" >&2
  exit 1
fi

mkdir -p "${BASE_DIR}"
cd "${SCRIPT_DIR}"

run_shard() {
  local shard="$1"
  local gpu="$2"
  local scene_output="$3"
  local shard_id
  shard_id="$(printf "%02d" "${shard}")"
  local out="${BASE_DIR}/shard_${shard_id}.h5"
  local log="${BASE_DIR}/shard_${shard_id}.log"
  local seed=$((2026061400 + shard))

  rm -f "${out}" "${out%.h5}.config.json" "${out%.h5}.metadata.jsonl" "${log}"
  echo "[launch] shard=${shard_id} gpu=${gpu} scene_output=${scene_output} seed=${seed} out=${out}" | tee "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" generate_dataset.py \
    --rawdata-dir "${RAW_DIR}" \
    --work-dir "${GEN_ROOT}" \
    --output "${out}" \
    --run-label "objdet_multifit_50000_s${shard_id}" \
    --unique-signals "${UNIQUE_PER_SHARD}" \
    --signals-per-batch "${SIGNALS_PER_BATCH_IMAGES}" \
    --signals-per-scene "${SIGNALS_PER_SCENE}" \
    --crops-per-signal "${CROPS_PER_SIGNAL}" \
    --max-objects-per-image "${MAX_OBJECTS_PER_IMAGE}" \
    --file-first 11 --file-last 275 \
    --scene-output-down "${scene_output}" \
    --tail-guard-down 1024 \
    --raw-chunk-down 8192 \
    --std-cache "${STD_CACHE}" \
    --std-file-limit 8 \
    --std-samples-per-file 16384 \
    --std-workers 4 \
    --min-split-effective-snr "${MIN_SPLIT_EFFECTIVE_SNR}" \
    --physical-fallback-min-snr "${PHYSICAL_FALLBACK_MIN_SNR}" \
    --seed "${seed}" \
    --gzip-level 1 >> "${log}" 2>&1
  echo "[done-shard] shard=${shard_id}" | tee -a "${log}"
}

run_wave() {
  local wave="$1"
  shift
  local shards=("$@")
  local gpus=(0 1 2 3 4 5 6 7)
  local pids=()
  echo "[wave] start ${wave} shards=${shards[*]}"
  for idx in "${!shards[@]}"; do
    local shard="${shards[$idx]}"
    local gpu="${gpus[$((idx % ${#gpus[@]}))]}"
    # 三种 scene 长度循环，覆盖更宽的时间窗（16384 才能用到 16384 的 time crop）：
    #   shard%4==0 -> 4096, ==2 -> 16384, 其余 -> 8192
    # 16384 的 input_down/显存更大，故只占 1/4 的 shard。
    local scene_output=8192
    if (( shard % 4 == 0 )); then
      scene_output=4096
    elif (( shard % 4 == 2 )); then
      scene_output=16384
    fi
    run_shard "${shard}" "${gpu}" "${scene_output}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "[wave] failed ${wave}" >&2
    exit 1
  fi
  echo "[wave] done ${wave}"
}

all_shards=()
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  all_shards+=("${shard}")
done
for ((wave_start = 0; wave_start < SHARD_COUNT; wave_start += SHARDS_PER_WAVE)); do
  wave_shards=()
  for ((shard = wave_start; shard < SHARD_COUNT && shard < wave_start + SHARDS_PER_WAVE; shard++)); do
    wave_shards+=("${shard}")
  done
  run_wave $((wave_start / SHARDS_PER_WAVE)) "${wave_shards[@]}"
done

inputs=()
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  inputs+=("${BASE_DIR}/shard_$(printf "%02d" "${shard}").h5")
done

rm -f "${FINAL_H5}" "${FINAL_H5%.h5}.config.json" "${FINAL_H5%.h5}.metadata.jsonl" "${FINAL_H5%.h5}.inspect.json"
"${PY}" merge_shards.py \
  --output "${FINAL_H5}" \
  --inputs "${inputs[@]}" \
  --gzip-level 1 \
  --max-objects-per-image "${MAX_OBJECTS_PER_IMAGE}" | tee "${FINAL_H5%.h5}.merge.log"

"${PY}" inspect_dataset.py \
  "${FINAL_H5}" \
  --samples 36 \
  --seed 789 \
  --contact-sheet "${FINAL_H5%.h5}_contact.png" \
  --json "${FINAL_H5%.h5}_visual_inspect.json"

echo "[all-done] ${FINAL_H5}"
