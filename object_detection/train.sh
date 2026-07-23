#!/usr/bin/env bash
# 单次拉起一个 CenterNet 模型的多卡 DDP 训练（本套件只做 centernet）。
#
# Usage:
#   ./train.sh "<gpu_ids>" <model>
#
# Examples:
#   ./train.sh "4,5,6,7" centernet-resnet18
#   ./train.sh "4,5,6,7" centernet-conv-tiny
#   ./train.sh "4,5,6,7" centernet-conv-small
#   BATCH_SIZE=32 EPOCHS=100 ./train.sh "0,1" centernet-resnet18
#
# Models:
#   centernet-resnet18   ->  centernet_train.py --backbone resnet18
#   centernet-conv-tiny  ->  centernet_train.py --backbone convnext_tiny
#   centernet-conv-small ->  centernet_train.py --backbone convnext_small

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 \"<gpu_ids>\" <model>" >&2
  echo "  model: centernet-resnet18 | centernet-conv-tiny | centernet-conv-small" >&2
  exit 1
fi

gpu_ids=$1
model=$2
extra_args=("${@:3}")

# 可通过环境变量覆盖
data_path=${DATA_PATH:-./Data/}
epochs=${EPOCHS:-50}
num_workers=${NUM_WORKERS:-16}
master_port=${MASTER_PORT:-$((RANDOM % 500 + 29500))}

IFS=',' read -ra gpu_arr <<< "$gpu_ids"
nproc_per_node=${#gpu_arr[@]}

# BATCH_SIZE 自适应：目标 effective_batch = 128（与 4 卡 batch=32 的设定一致），
# 单卡 batch 上限 32 防 OOM。env BATCH_SIZE 显式设值时直接覆盖。
default_bs=$(( 128 / nproc_per_node ))
if [ "$default_bs" -gt 32 ]; then default_bs=32; fi
if [ "$default_bs" -lt 1 ];  then default_bs=1;  fi
batch_size=${BATCH_SIZE:-$default_bs}
# 让 CUDA_VISIBLE_DEVICES 的编号和 nvidia-smi 一致（默认 FASTEST_FIRST 会错位）
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_ids"

case "$model" in
  centernet-resnet18)
    log_dir=./logs/logs_centernet_resnet18
    model_args=(--backbone resnet18)
    ;;
  centernet-conv-tiny)
    log_dir=./logs/logs_centernet_conv_tiny
    model_args=(--backbone convnext_tiny)
    ;;
  centernet-conv-small)
    log_dir=./logs/logs_centernet_conv_small
    model_args=(--backbone convnext_small)
    ;;
  *)
    echo "Unknown model: $model" >&2
    echo "Valid: centernet-resnet18 | centernet-conv-tiny | centernet-conv-small" >&2
    exit 1
    ;;
esac

mkdir -p "$log_dir"

echo "================================================================"
echo "  Model        : $model"
echo "  GPUs         : $gpu_ids  (nproc_per_node=$nproc_per_node)"
echo "  Log dir      : $log_dir"
echo "  Master port  : $master_port"
echo "  Batch/Epochs : $batch_size / $epochs"
echo "  Started      : $(date)"
echo "================================================================"

torchrun \
  --nproc_per_node "$nproc_per_node" \
  --master_port "$master_port" \
  centernet_train.py \
  "${model_args[@]}" \
  --data-path "$data_path" \
  --log-dir "$log_dir" \
  --batch-size "$batch_size" \
  --epochs "$epochs" \
  --num-workers "$num_workers" \
  --amp \
  "${extra_args[@]}"

echo "[done] $model finished at $(date)"
