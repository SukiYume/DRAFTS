# DRAFTS CenterNet 检测器训练

本目录训练 DRAFTS 搜索链路的第一阶段模型：从 512 × 512 time–DM 图中定位暂现源
候选中心，并将中心坐标转换为候选 TOA 与 DM。当前默认部署模型使用
CenterNet + ConvNeXt-Tiny；公开权重位于
[DRAFTS model repository](https://huggingface.co/TorchLight/DRAFTS)。

## 目录职责

| 文件 | 作用 |
|---|---|
| `centernet_data.py` | 扫描训练 H5，构造中心点监督、过采样和 mosaic/几何增强。 |
| `centernet_model.py` | CenterNet、backbone、heatmap 与 offset loss。 |
| `centernet_eval.py` | 解码中心点并计算 precision、recall、F1 和定位误差。 |
| `centernet_train.py` | 单卡/DDP 训练入口，支持 AMP、EMA、resume 和 SWA。 |
| `centernet_infer.py` | 对 validation 样本推理并绘制真值/预测中心。 |
| `train.sh` | 统一多卡启动参数和模型别名。 |

## 数据格式

`centernet_data.py` 会扫描 `--data-path` 下所有 `.h5`。每个文件至少包含：

```text
images:      (N, 512, 512)
annotations: (M, 5)
```

`annotations` 每行格式为：

```text
image_index, left, top, width, height
```

CenterNet 只预测中心 heatmap 与亚像素 offset；边框宽高用于生成中心监督和高斯半径，
不会作为模型输出。训练集会按每帧目标数量及小目标存在情况过采样，validation 集保持
原始分布。

训练数据可由 [`../generate_burst/`](../generate_burst/) 生成。

## 安装

从仓库根目录安装通用依赖：

```bash
python -m pip install -r requirements.txt
```

PyTorch、torchvision 和 CUDA 版本需要与训练节点匹配。

## 训练

推荐通过 `train.sh` 启动：

```bash
cd object_detection
DATA_PATH=/path/to/centernet_h5 \
BATCH_SIZE=24 \
EPOCHS=50 \
./train.sh "0,1,2,3,4,5,6,7" centernet-conv-tiny
```

可选别名：

```text
centernet-resnet18
centernet-conv-tiny
centernet-conv-small
```

底层 Python 入口：

```bash
python centernet_train.py \
  --data-path /path/to/centernet_h5 \
  --backbone convnext_tiny \
  --batch-size 24 \
  --epochs 50 \
  --log-dir /path/to/training_runs/centernet-conv-tiny
```

主要产物包括：

```text
best_model_ema.pth
best_model.pth
last_checkpoint.pth
swa_model_ema.pth
logs_centernet.json
args.json
```

`best_model_ema.pth` 是常用部署候选。最终选择仍应结合独立 validation、真实观测搜索
和注入实验的召回率、误报率及 TOA/DM 定位误差。

## 推理抽查

先下载当前默认检测器：

```bash
curl -L \
  -o /path/to/models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  https://huggingface.co/TorchLight/DRAFTS/resolve/main/object_best_model_centernet_conv_tiny_ema_v10.pth
```

然后抽查 validation 图：

```bash
python centernet_infer.py \
  --weights /path/to/models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  --data-path /path/to/centernet_h5 \
  --backbone convnext_tiny \
  --conf 0.3 \
  --start 0 \
  --end 30
```

权重和 `--backbone` 必须一致。推理图用于检查中心偏移、漏检和重复候选，不能替代完整
validation 指标。

## 部署到 DRAFTS

真实观测搜索默认读取：

```text
runcode/models/object_best_model_centernet_conv_tiny_ema_v10.pth
```

注入实验默认读取：

```text
injection_experiment/search_runtime/models/object_best_model_centernet_conv_tiny_ema_v10.pth
```

下载后保持文件名不变，或在入口中显式传入 `--detector-ckpt`，并使用
`--detector-type centernet_conv_tiny`。

## 常见问题

| 现象 | 检查项 |
|---|---|
| 找不到训练样本 | 确认数据目录中存在含 `images` 与 `annotations` 的 H5。 |
| checkpoint 形状不匹配 | 检查 `resnet18`、`convnext_tiny`、`convnext_small` 是否一致。 |
| validation 候选过多 | 同时检查 `--conf`、`--topk` 和训练数据中的空背景比例。 |
| 显存不足 | 减小 batch size；`train.sh` 也支持用 `BATCH_SIZE` 覆盖。 |
| resume 失败 | `--resume` 需要包含 optimizer/scheduler 状态的完整 checkpoint。 |
