#!/usr/bin/env bash
set -euo pipefail

# Edit this compact configuration block before launching a run.
H5_PATH="./FRB20220912A_PrepData.h5"
DOWN_TIMES=(8 16 32 64 128 256)
MODEL="vae"                 # ae or vae
ENCODER="convnext"         # convnext or resnet18
PRED_MODE=false             # predict x[t] from x[t-1]
USE_PRETRAINED=false
USE_AUGMENTATION=true

CUDA_VISIBLE_DEVICES="0,1,2,3"
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-4
OUTPUT_DIR="./outputs"
NUM_WORKERS=8
WARMUP_EPOCHS=2

LATENT_DIM=128
KL_WEIGHT=0.05
KL_WARMUP_EPOCHS=5

if [[ ! -f "$H5_PATH" ]]; then
    echo "Error: HDF5 file not found: $H5_PATH" >&2
    echo "Create it with prepare_data.py before training." >&2
    exit 1
fi
if [[ "$MODEL" != "ae" && "$MODEL" != "vae" ]]; then
    echo "Error: MODEL must be ae or vae" >&2
    exit 1
fi
if [[ "$ENCODER" != "convnext" && "$ENCODER" != "resnet18" ]]; then
    echo "Error: ENCODER must be convnext or resnet18" >&2
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_IDS[@]}
if (( NUM_GPUS == 0 )); then
    echo "Error: CUDA_VISIBLE_DEVICES must list at least one GPU" >&2
    exit 1
fi

TRAIN_ARGS=(
    train_ddp.py
    --h5_path "$H5_PATH"
    --down_times "${DOWN_TIMES[@]}"
    --model "$MODEL"
    --encoder "$ENCODER"
    --epochs "$EPOCHS"
    --warmup_epochs "$WARMUP_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LEARNING_RATE"
    --output_dir "$OUTPUT_DIR"
    --num_workers "$NUM_WORKERS"
)

if [[ "$USE_PRETRAINED" == false ]]; then
    TRAIN_ARGS+=(--no_pretrained)
fi
if [[ "$USE_AUGMENTATION" == true ]]; then
    TRAIN_ARGS+=(--augment)
fi
if [[ "$PRED_MODE" == true ]]; then
    TRAIN_ARGS+=(--pred)
fi
if [[ "$MODEL" == "vae" ]]; then
    TRAIN_ARGS+=(
        --latent_dim "$LATENT_DIM"
        --kl_weight "$KL_WEIGHT"
        --kl_warmup_epochs "$KL_WARMUP_EPOCHS"
    )
fi

echo "============================================================"
echo "Radio anomaly training"
echo "HDF5:      $H5_PATH"
echo "Down time: ${DOWN_TIMES[*]}"
echo "Mode:      $MODEL (prediction=$PRED_MODE)"
echo "Encoder:   $ENCODER (pretrained=$USE_PRETRAINED)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Batch:     $BATCH_SIZE x $NUM_GPUS = $((BATCH_SIZE * NUM_GPUS))"
echo "Epochs:    $EPOCHS"
echo "============================================================"

export CUDA_VISIBLE_DEVICES
if (( NUM_GPUS > 1 )); then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" "${TRAIN_ARGS[@]}"
else
    python "${TRAIN_ARGS[@]}"
fi

echo "Training complete. Output root: $OUTPUT_DIR"
