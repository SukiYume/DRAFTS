# Binary 分类训练目录

本目录用于训练单脉冲搜索中的二分类过滤器。它接收候选 burst 图，输出 positive / negative 概率，用于过滤目标检测阶段的误报。

## 文件职责

| 文件或目录 | 作用 |
|---|---|
| `binary_train.py` | 单卡、DDP、CPU 通用训练入口。支持 EMA、cosine schedule、resume 和 finetune。 |
| `binary_infer.py` | 用训练好的权重对数据集推理并导出预测表。 |
| `binary_data.py` | 数据集读取、训练/验证划分和预处理。 |
| `binary_model.py` | ConvNeXt 和 SPPConvNeXt 二分类模型定义。 |
| `train.sh` | shell 包装脚本，便于设置 GPU 和常用超参数。 |
| `logs/` | 训练日志、checkpoint 和模型对比报告。其中包含逐 epoch 的 `logs.json` 和 `last_checkpoint.pth`。 |

## 数据约定

训练脚本默认读取 `./Data/`。该目录应包含正负样本图或 H5 数据，具体格式由 `binary_data.py` 解析。数据目录不作为代码目录说明的一部分。

## 训练示例

单卡：

```bash
python binary_train.py \
  --data-path ./Data/ \
  --model-name convnext_tiny \
  --epochs 50 \
  --device 0
```

多卡：

```bash
torchrun --nproc_per_node=8 binary_train.py \
  --data-path ./Data/ \
  --model-name convnext_small \
  --epochs 50
```

包装脚本：

```bash
MODEL_NAME=convnext_tiny EPOCHS=50 ./train.sh "0,1,2,3"
```

## 推理示例

```bash
python binary_infer.py \
  --weights logs/logs_binary_convnext_tiny/best_model_ema.pth \
  --data-path ./Data/ \
  --output binary_predictions.csv \
  --model-name convnext_tiny
```

## 选择 checkpoint

训练脚本默认使用 EMA 验证 `f1` 保存 best checkpoint：

```text
best_model_ema.pth
```

`logs/model_comparison.md` 是当前日志集的人类可读对比。默认部署选择需要结合 F1、precision、recall、loss、模型大小和真实搜索表现。训练目录里的 checkpoint 不会自动被搜索入口使用，需要复制到运行模型目录并修改入口脚本。

## 模型对比结论

`logs/model_comparison.md` 比较了 ConvNeXt Tiny 与 ConvNeXt Small 两个 50-epoch 训练 run，主口径是训练脚本实际保存 best checkpoint 的 `f1_ema`。两者整体非常接近，核心指标差距在 `1e-4` 量级。

| 模型 | best epoch | f1_ema | precision_ema | recall_ema | loss_ema |
|---|---:|---:|---:|---:|---:|
| ConvNeXt Tiny | 37 | 0.986434 | 0.987835 | 0.986434 | 0.028819 |
| ConvNeXt Small | 47 | 0.986444 | 0.985733 | 0.987155 | 0.028427 |

- **Small** 的 best `f1_ema` 高 `0.000010`，默认阈值下后期 recall 略高。
- **Tiny** 的 validation loss、precision、收敛速度（raw `f1_val≥0.985` 早约 10 个 epoch）和模型规格都更优。

**训练日志默认推荐：ConvNeXt Tiny**（`logs/logs_binary_convnext_tiny/best_model_ema.pth`）——综合 loss、precision、收敛速度和体积最均衡。需要更高默认阈值 recall、且接受更大模型时选 Small（`logs/logs_binary_convnext_small/best_model_ema.pth`）。

> **与部署默认的关系**：`runcode/` 盲搜与固定 DM 入口当前默认使用 **ConvNeXt-Small**（`CLASSIFIER_MODEL_NAME='convnext_small'`），偏向真实搜索中的 recall；训练日志的纸面默认则是 Tiny。两套权重都已放在 `runcode/models/`。binary 分类器的核心风险是漏检（false negative），最终默认值应结合真实搜索输出的漏检统计确定（见报告「补充验证建议」）。

## 排错

| 现象 | 检查项 |
|---|---|
| 训练找不到数据 | 检查 `--data-path` 和数据格式。 |
| checkpoint 加载失败 | 检查 `--model-name`、`--model-type` 是否与训练时一致。 |
| DDP 卡住 | 检查 `torchrun` 进程数、可见 GPU、NCCL 环境。 |
| 指标和部署表现不一致 | 用真实搜索输出复查 false positive 和 missed candidate。 |
