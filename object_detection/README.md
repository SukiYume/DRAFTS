# CenterNet 检测器训练目录

本目录训练 DRAFTS 第一阶段目标检测器。当前只维护 CenterNet 线，用于在 512 x 512 的 time-DM 图上定位候选中心；旧 YOLO 实验在上级 `bslocate/` 中保留，不属于当前搜索主线。

## 文件职责

| 文件或目录 | 作用 |
|---|---|
| `centernet_train.py` | 单卡/DDP 训练入口，支持 EMA、AMP、resume、SWA 和日志落盘。 |
| `centernet_data.py` | 读取 CenterNet H5 数据、构造 heatmap/offset target、训练/验证划分和增强。 |
| `centernet_model.py` | CenterNet 模型构建，支持 `resnet18`、`convnext_tiny`、`convnext_small` backbone。 |
| `centernet_eval.py` | heatmap 解码和中心距离匹配指标。 |
| `centernet_infer.py` | 从验证集抽样推理并可视化检测中心。 |
| `train.sh` | 多卡 DDP 包装脚本。 |
| `Data/` | 当前训练 H5 与 annotations JSON。 |
| `logs_v10/` | 当前保留的 v10 训练产物。 |

## 数据约定

训练脚本默认读取 `./Data/`，可以用 `DATA_PATH` 或 `--data-path` 覆盖。当前本地数据包含：

```text
centernet_dataset_sim50000_max3.h5
centernet_dataset_sim50000_max3_annotations.json
centernet_dataset_sim20000_max3.h5
centernet_dataset_sim20000_max3_annotations.json
centernet_dataset_crafts_2020..2024.h5
centernet_dataset_crafts_2020..2024_annotations.json
```

生成新模拟数据时从 `../generate_burst/` 产生 H5 和 annotations JSON，再复制或链接到本目录 `Data/`。

## 训练

包装脚本用法：

```bash
./train.sh "<gpu_ids>" <model>
```

可选模型：

| 名称 | 传给训练脚本的 backbone |
|---|---|
| `centernet-resnet18` | `resnet18` |
| `centernet-conv-tiny` | `convnext_tiny` |
| `centernet-conv-small` | `convnext_small` |

v10 训练口径示例：

```bash
DATA_PATH=./Data/v10_sim50000_only \
BATCH_SIZE=24 \
EPOCHS=50 \
./train.sh "0,1,2,3,4,5,6,7" centernet-conv-tiny
```

直接调用训练脚本：

```bash
torchrun --nproc_per_node=8 centernet_train.py \
  --backbone convnext_tiny \
  --data-path ./Data/v10_sim50000_only \
  --log-dir logs_v10/logs_centernet_conv_tiny_sim50000 \
  --batch-size 24 \
  --epochs 50 \
  --amp
```

## 当前保留的 v10 产物

`logs_v10/args.json` 记录的关键配置：

| 项 | 值 |
|---|---|
| data path | `./Data/v10_sim50000_only` |
| backbone | `convnext_tiny` |
| epochs | 50 |
| world size | 8 |
| batch size | 24 per GPU |
| effective batch size | 192 |
| eval distance threshold | 8 px |

保留文件：

```text
logs_v10/best_model_ema.pth
logs_v10/best_model.pth
logs_v10/last_checkpoint.pth
logs_v10/swa_model_ema.pth
logs_v10/logs_centernet.json
logs_v10/args.json
```

最近日志尾部的验证指标约为 `f1_ema=0.943`，中心距离中位数约 `0.43 px`，`p90` 约 `2.30 px`。最终模型选择仍应结合真实搜索或注入评估表现。

## 推理抽查

```bash
python centernet_infer.py \
  --weights logs_v10/best_model_ema.pth \
  --data-path ./Data/v10_sim50000_only \
  --backbone convnext_tiny \
  --conf 0.3 \
  --start 0 \
  --end 30
```

## 部署到搜索入口

训练目录里的 checkpoint 不会自动被 `runcode/` 使用。部署时显式复制并命名：

```bash
cp logs_v10/best_model_ema.pth ../runcode/models/object_best_model_centernet_conv_tiny_ema_v10.pth
```

如果要替换生产默认别名，再复制为：

```bash
cp logs_v10/best_model_ema.pth ../runcode/models/object_best_model_centernet_conv_tiny_ema.pth
```

替换默认别名前，先确认 `runcode/` 盲搜、`injection_experiment/search_runtime/` 和任何远端部署脚本都指向同一权重版本。
