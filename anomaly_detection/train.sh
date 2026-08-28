#!/bin/bash
# Training script for Radio Anomaly Detection (HDF5 version)
#
# Usage: bash train.sh

# ==================== 配置参数 ====================

H5_PATH="./FRB20220912A_PrepData.h5"
DOWN_TIMES="8 16 32 64 128 256"

EPOCHS=50
BATCH_SIZE=32
LR=1e-4
MODEL="vae"          # ae 或 vae
ENCODER="convnext"
PRED_MODE=false     # 是否使用预测模式 (f(x_t-1) = x_t)

# GPU 配置
CUDA_VISIBLE_DEVICES="0,1,2,3"  # 指定使用哪几张卡
NUM_GPUS=4                        # GPU 数量需要与上面指定的卡数一致
OUTPUT_DIR="./outputs"
NUM_WORKERS=16
WARMUP_EPOCHS=2

# VAE 专用参数
KL_WEIGHT=0.05
KL_WARMUP_EPOCHS=5

# ==================== 构建参数 ====================

PRED_FLAG=""
if [ "$PRED_MODE" = true ]; then
    PRED_FLAG="--pred"
fi

# ==================== 运行训练 ====================

echo "============================================"
echo "Radio Anomaly Detection Training (HDF5)"
echo "============================================"
echo "H5 Path:    $H5_PATH"
echo "Down times: $DOWN_TIMES"
echo "Model:      $MODEL"
echo "Pred mode:  $PRED_MODE"
echo "Encoder:    $ENCODER"
echo "GPUs:       $NUM_GPUS"
echo "Batch:      $BATCH_SIZE x $NUM_GPUS = $((BATCH_SIZE * NUM_GPUS))"
echo "Epochs:     $EPOCHS"
echo "============================================"

if [ ! -f "$H5_PATH" ]; then
    echo "Error: H5 file not found: $H5_PATH"
    echo "Run: python prepare_data.py first"
    exit 1
fi

if [ $NUM_GPUS -gt 1 ]; then
    echo "Starting DDP training with $NUM_GPUS GPUs..."
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

    if [ "$MODEL" = "vae" ]; then
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --standalone --nproc_per_node=$NUM_GPUS train_ddp.py \
            --no_pretrained \
            --h5_path "$H5_PATH" \
            --down_times $DOWN_TIMES \
            --model vae \
            --encoder $ENCODER \
            $PRED_FLAG \
            --epochs $EPOCHS \
            --warmup_epochs $WARMUP_EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --kl_weight $KL_WEIGHT \
            --kl_warmup_epochs $KL_WARMUP_EPOCHS \
            --output_dir "$OUTPUT_DIR" \
            --num_workers $NUM_WORKERS \
            --augment
    else
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --standalone --nproc_per_node=$NUM_GPUS train_ddp.py \
            --no_pretrained \
            --h5_path "$H5_PATH" \
            --down_times $DOWN_TIMES \
            --model ae \
            --encoder $ENCODER \
            $PRED_FLAG \
            --epochs $EPOCHS \
            --warmup_epochs $WARMUP_EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --output_dir "$OUTPUT_DIR" \
            --num_workers $NUM_WORKERS \
            --augment
    fi
else
    echo "Starting single GPU training..."
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_ddp.py \
        --h5_path "$H5_PATH" \
        --down_times $DOWN_TIMES \
        --model $MODEL \
        --encoder $ENCODER \
        $PRED_FLAG \
        --epochs $EPOCHS \
        --warmup_epochs $WARMUP_EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --output_dir "$OUTPUT_DIR" \
        --num_workers $NUM_WORKERS \
        --augment
fi

echo "============================================"
echo "Training complete!"
echo "Output: $OUTPUT_DIR"
echo "============================================"
