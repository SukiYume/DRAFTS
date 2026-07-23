# DRAFTS 二分类器训练

本目录训练 DRAFTS 搜索链路的第二阶段模型：CenterNet 给出候选位置后，
ConvNeXt 二分类器判断候选是否更像真实 burst。默认部署模型是
`convnext_small`，对应的公开权重可以直接从
[DRAFTS model repository](https://huggingface.co/TorchLight/DRAFTS) 下载。

## 目录职责

| 文件 | 作用 |
|---|---|
| `binary_data.py` | 读取正负样本 H5，完成分层切分、过采样和训练增强。 |
| `binary_model.py` | 定义 `ConvNeXtNet` 与 `SPPConvNeXt`。 |
| `binary_train.py` | 单卡或 DDP 训练入口，支持 AMP、EMA、resume 和 finetune。 |
| `binary_infer.py` | 对 H5 数据集推理并导出逐样本预测表。 |
| `train.sh` | 统一多卡启动参数和模型别名。 |

训练产生的 checkpoint、逐 epoch 指标和推理表写入运行目录；它们不是运行本目录代码的
前置文件。

## 数据格式

默认从 `./Data/` 读取以下文件，或通过 `--data-path` / `DATA_PATH` 指向其他目录：

```text
True_FRB.h5
True_LPT.h5
False_FRB.h5
False_LPT.h5
False_FRB20240114A.h5
```

每个 H5 至少包含：

| 数据集 | 含义 |
|---|---|
| `images` | time–DM 图像数组。 |
| `labels` | 二分类标签；`1` 为真实 burst，`0` 为负样本。 |
| `idx` | `images` 中的样本索引。 |
| `names` | 样本名称。 |
| `sources` | 可选来源字段，用于区分 LPT 样本。 |

代码先按标签分层切分 train/validation，再仅对训练集过采样，避免同一样本同时出现在
两侧。训练增强包括 max-mixup、随机拼图、旋转、翻转和合成干扰。

## 安装

从仓库根目录安装通用依赖：

```bash
python -m pip install -r requirements.txt
```

PyTorch 和 torchvision 应使用与目标 GPU/CUDA 环境匹配的版本。

## 训练

推荐通过 `train.sh` 启动：

```bash
cd binary_classification
DATA_PATH=/path/to/binary_h5 \
MODEL_NAME=convnext_small \
BATCH_SIZE=32 \
EPOCHS=50 \
./train.sh "0,1,2,3"
```

第二个位置参数也可以直接选择模型别名：

```bash
./train.sh "0,1" small
./train.sh "0,1" tiny
./train.sh "0,1" spp-small
```

底层 Python 入口适合精细控制：

```bash
python binary_train.py \
  --data-path /path/to/binary_h5 \
  --model-type ConvNeXtNet \
  --model-name convnext_small \
  --batch-size 32 \
  --epochs 50 \
  --log-dir /path/to/training_runs/binary-convnext-small
```

主要训练产物包括：

```text
best_model_ema.pth
best_model.pth
last_checkpoint.pth
logs.json
```

生产搜索通常优先使用 EMA 权重。模型选择应结合独立验证集以及真实观测或注入实验中的
漏检与误报表现，不应只根据单个训练指标决定。

## 推理

先下载当前默认分类器：

```bash
curl -L \
  -o /path/to/models/binary_best_model_conv_small_ema.pth \
  https://huggingface.co/TorchLight/DRAFTS/resolve/main/binary_best_model_conv_small_ema.pth
```

然后运行：

```bash
python binary_infer.py \
  --weights /path/to/models/binary_best_model_conv_small_ema.pth \
  --data-path /path/to/binary_h5 \
  --model-type ConvNeXtNet \
  --model-name convnext_small \
  --output /path/to/evaluation/binary_predictions.csv
```

`--model-name` 必须与 checkpoint 的 backbone 一致。输出表包含样本名、真值、预测类别和
正类概率，可用于选择阈值或定位误分类样本。

## 部署到 DRAFTS

真实观测搜索默认读取：

```text
runcode/models/binary_best_model_conv_small_ema.pth
```

注入实验默认读取：

```text
injection_experiment/search_runtime/models/binary_best_model_conv_small_ema.pth
```

下载后保持文件名不变，或在对应入口中显式传入 `--classifier-ckpt`；同时保持
`classifier_model_name=convnext_small`。

## 常见问题

| 现象 | 检查项 |
|---|---|
| 找不到 H5 | 确认 `--data-path` 下至少存在一个约定名称的 H5。 |
| checkpoint 形状不匹配 | 检查 `ConvNeXtNet` / `SPPConvNeXt` 和 backbone 名称。 |
| 显存不足 | 减小 `--batch-size`，或使用 `convnext_tiny`。 |
| 多卡没有启动 | 用逗号分隔 GPU ID，并确认 `torchrun` 可用。 |
| resume 失败 | `--resume` 需要完整训练 checkpoint；仅模型权重请用 `--finetune`。 |
