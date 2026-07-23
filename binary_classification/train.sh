#!/usr/bin/env bash
# 单次拉起一个 binary 分类器的多卡 DDP 训练。
#
# Usage:
#   ./train.sh "<gpu_ids>" <model>
#
# Examples:
#   ./train.sh "0,1,2,3" tiny
#   ./train.sh "4,5,6,7" base
#   BATCH_SIZE=32 EPOCHS=100 ./train.sh "0,1" small
#   ./train.sh "0,1" spp-tiny       # 变尺寸训练（128~512）
#
# Models:
#   tiny       ->  binary_train.py --model-name convnext_tiny  (fix 512)
#   small      ->  binary_train.py --model-name convnext_small (fix 512)
#   base       ->  binary_train.py --model-name convnext_base  (fix 512)
#   large      ->  binary_train.py --model-name convnext_large (fix 512)
#   spp-tiny   ->  binary_train.py --model-type SPPConvNeXt --model-name convnext_tiny  (random 128~512)
#   spp-small  ->  binary_train.py --model-type SPPConvNeXt --model-name convnext_small (random 128~512)

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 \"<gpu_ids>\" <model>" >&2
  echo "  model: tiny | small | base | large | spp-tiny | spp-small" >&2
  exit 1
fi

gpu_ids=$1
model=$2

# 可通过环境变量覆盖
data_path=${DATA_PATH:-./Data/}
batch_size=${BATCH_SIZE:-32}            # 0 → 让 binary_train.py 按 fix_size / DDP 自动选
epochs=${EPOCHS:-50}
num_workers=${NUM_WORKERS:-16}
master_port=${MASTER_PORT:-$((RANDOM % 500 + 29500))}

IFS=',' read -ra gpu_arr <<< "$gpu_ids"
nproc_per_node=${#gpu_arr[@]}
# 让 CUDA_VISIBLE_DEVICES 的编号和 nvidia-smi 一致（默认 FASTEST_FIRST 会错位）
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_ids"

case "$model" in
  tiny)
    log_dir=./logs/logs_binary_convnext_tiny
    model_args=(--model-type ConvNeXtNet --model-name convnext_tiny  --fix-size true)
    ;;
  small)
    log_dir=./logs/logs_binary_convnext_small
    model_args=(--model-type ConvNeXtNet --model-name convnext_small --fix-size true)
    ;;
  base)
    log_dir=./logs/logs_binary_convnext_base
    model_args=(--model-type ConvNeXtNet --model-name convnext_base  --fix-size true)
    ;;
  large)
    log_dir=./logs/logs_binary_convnext_large
    model_args=(--model-type ConvNeXtNet --model-name convnext_large --fix-size true)
    ;;
  spp-tiny)
    log_dir=./logs/logs_binary_spp_tiny
    model_args=(--model-type SPPConvNeXt --model-name convnext_tiny  --fix-size false)
    ;;
  spp-small)
    log_dir=./logs/logs_binary_spp_small
    model_args=(--model-type SPPConvNeXt --model-name convnext_small --fix-size false)
    ;;
  *)
    echo "Unknown model: $model" >&2
    echo "Valid: tiny | small | base | large | spp-tiny | spp-small" >&2
    exit 1
    ;;
esac

mkdir -p "$log_dir"

echo "================================================================"
echo "  Model        : $model"
echo "  GPUs         : $gpu_ids  (nproc_per_node=$nproc_per_node)"
echo "  Log dir      : $log_dir"
echo "  Master port  : $master_port"
echo "  Batch/Epochs : $batch_size / $epochs   (batch=0 → auto-pick in binary_train.py)"
echo "  Started      : $(date)"
echo "================================================================"

torchrun \
  --nproc_per_node "$nproc_per_node" \
  --master_port "$master_port" \
  binary_train.py \
  "${model_args[@]}" \
  --data-path "$data_path" \
  --log-dir "$log_dir" \
  --batch-size "$batch_size" \
  --epochs "$epochs" \
  --num-workers "$num_workers" \
  --amp

echo "[done] $model finished at $(date)"
