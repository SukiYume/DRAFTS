# Radio Telescope Anomaly Detection

基于自编码器的射电望远镜时频谱图异常检测系统。使用 ConvNeXt 编码器和 U-Net 风格解码器，通过重建误差识别异常信号。

## 本地保留范围

本目录是从历史实验目录精简后的可复现快照，只保留源码、说明和最佳权重：

- `best_loss_model.pth`：ConvNeXt 编码器、128 通道潜变量的 VAE 最低验证损失权重；
- `prepare_data.py`：从外部 FITS 重新生成 H5 输入；
- `run_inference.py`：从 H5 或 FITS 目录运行异常检测；
- 训练、数据集和潜变量分析代码。

历史 `inference_data/`、`inference_results/`、缓存和临时 H5 没有随项目迁移。新的输入和输出应放在仓库外的数据/运行目录，或使用下方参数显式指定。

保留权重的典型推理命令：

```bash
python run_inference.py \
    --fits_dir /path/to/fits_directory \
    --model_path ./best_loss_model.pth \
    --model vae \
    --encoder convnext \
    --output_dir /path/to/anomaly_results \
    --plot_anomalies
```

## 📁 项目结构

```
RadioAnomalyDetection/
├── models.py           # 模型定义 (AE/VAE) + Loss 函数
├── dataset_h5.py       # HDF5 数据集 + 数据增强
├── prepare_data.py     # FITS → HDF5 数据转换
├── train_ddp.py        # DDP 多卡训练脚本
├── run_inference.py    # 推理 + 异常检测
├── train.sh            # 训练启动脚本
└── requirements.txt    # 依赖包
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将 FITS 文件转换为 HDF5 格式：

```bash
python prepare_data.py
```

或在 Python 中：

```python
from prepare_data import build_h5_from_fits_paths

h5_path, _, _ = build_h5_from_fits_paths(
    fits_paths=["/path/to/fits/directory"],
    out_h5_path="data.h5",
    down_time_list=(8, 16, 32, 64, 128, 256),
    num_workers=8,
)
```

### 3. 训练模型

**单卡训练：**
```bash
python train_ddp.py \
    --h5_path ./data.h5 \
    --model ae \
    --epochs 50 \
    --batch_size 32 \
    --augment
```

**多卡 DDP 训练：**
```bash
torchrun --nproc_per_node=8 train_ddp.py \
    --h5_path ./data.h5 \
    --model ae \
    --epochs 50 \
    --batch_size 32 \
    --augment
```

**VAE 训练：**
```bash
torchrun --nproc_per_node=8 train_ddp.py \
    --h5_path ./data.h5 \
    --model vae \
    --epochs 50 \
    --kl_weight 0.1 \
    --kl_warmup_epochs 10 \
    --augment
```

### 4. 推理

```bash
python run_inference.py \
    --h5_path ./data.h5 \
    --model_path ./outputs/best.pth \
    --model ae \
    --plot_anomalies
```

---

## 🔄 数据流 (Data Flow)

本节详细描述数据如何从原始 FITS 时频数组，经过预处理、模型推理，最终判断是否为异常。

### 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │  FITS File   │────▶│  Time-Freq   │────▶│   512×512    │                │
│  │ (Raw Data)   │     │   Array      │     │   Patches    │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│         │                    │                    │                          │
│         │              下采样 + 归一化            │                          │
│         ▼                    ▼                    ▼                          │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                      HDF5 File                            │              │
│  │  images: (N, 512, 512) float32 [0,1]                     │              │
│  │  names:  原始文件名                                       │              │
│  │  down_time: 时间下采样率                                  │              │
│  │  slice_idx: patch 序号                                    │              │
│  └──────────────────────────────────────────────────────────┘              │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                   TRAINING PHASE                          │              │
│  │                                                           │              │
│  │   Input ─▶ Encoder ─▶ Latent ─▶ Decoder ─▶ Recon        │              │
│  │   (1,512,512)  (768,16,16)        (1,512,512)            │              │
│  │                                                           │              │
│  │   Loss = SmoothL1(Recon, Input) [+ β×KL for VAE]         │              │
│  └──────────────────────────────────────────────────────────┘              │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                   INFERENCE PHASE                         │              │
│  │                                                           │              │
│  │   Input ─▶ Model ─▶ Recon                                │              │
│  │              │                                            │              │
│  │              ▼                                            │              │
│  │   Error Map = |Input - Recon|                            │              │
│  │              │                                            │              │
│  │              ▼                                            │              │
│  │   Score = max(mean(Error), p99(Error))                   │              │
│  │              │                                            │              │
│  │              ▼                                            │              │
│  │   Anomaly = (Score > Threshold)                          │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 阶段 1: FITS → HDF5 预处理

**输入**: 原始 FITS 文件，包含时间-频率数据 `(T, F)`

**处理步骤**:

1. **读取 FITS**: 提取 DATA 字段，合并极化通道
   ```
   (NAXIS2 × NSBLK, NPOL, NCHAN) → mean(axis=1) → (T, F)
   ```

2. **频率下采样**: 将频率维压缩到 512
   ```
   (T, F) → reshape(T, 512, F/512) → mean(axis=2) → (T, 512)
   ```

3. **时间下采样**: 按多个倍率（如 8, 16, 32, 64...）下采样
   ```
   (T, 512) → reshape + mean → (T/down_time, 512)
   ```

4. **切片提取**: 每 512 行切成一个 patch
   ```
   (T', 512) → reshape → (N_slices, 512, 512)
   ```

5. **归一化**: 频率通道归一化 + 5%-95% clip + min-max 到 [0,1]
   ```python
   data /= mean(data, axis=0)      # 按频率通道归一化
   data = clip(data, p5, p95)      # 去除极端值
   data = (data - min) / (max - min)  # 归一化到 [0,1]
   ```

**输出**: HDF5 文件
```
images:    (N, 512, 512) float32, 归一化到 [0,1]
names:     (N, 1) 原始 FITS 文件路径
down_time: (N, 1) 时间下采样倍率
slice_idx: (N, 1) 该倍率下的 slice 序号
```

### 阶段 2: 模型训练

**目标**: 让模型学会重建"正常"的时频谱图

**模型架构**:

```
Input (1, 512, 512)
    │
    ▼
┌─────────────────────────────────────────────┐
│  ConvNeXt-Tiny Encoder (ImageNet pretrained) │
│  ├── Stem (stride 4)   → (96, 128, 128)     │
│  ├── Stage 0           → (96, 128, 128)     │
│  ├── Stage 1 (down)    → (192, 64, 64)      │
│  ├── Stage 2 (down)    → (384, 32, 32)      │
│  └── Stage 3 (down)    → (768, 16, 16)  ◀── Bottleneck
└─────────────────────────────────────────────┘
    │
    │ [VAE: → μ, logσ² → reparameterize → z]
    │
    ▼
┌─────────────────────────────────────────────┐
│  Symmetric Decoder (Gated Skip Connections)  │
│  ├── Up 1 (×2)         → (384, 32, 32)      │
│  ├── Up 2 (×2)         → (192, 64, 64)      │
│  ├── Up 3 (×2)         → (96, 128, 128)     │
│  └── Final (×4)        → (1, 512, 512)      │
└─────────────────────────────────────────────┘
    │
    ▼
Output (1, 512, 512) with Sigmoid → [0, 1]
```

**损失函数**:
- **AE**: `Loss = SmoothL1(Recon, Input)`
- **VAE**: `Loss = SmoothL1(Recon, Input) + β × KL(q(z|x) || p(z))`

使用 **Huber/SmoothL1** 而非 MSE，对 RFI 等异常值更鲁棒。

**关键技术**:
- Gated Skip Connections: 初始化接近 0，强制信息通过 bottleneck
- KL Annealing: β 从 0 线性升到目标值，防止 KL 坍塌
- Differential LR: Encoder 使用更小的学习率（0.3×）

### 阶段 3: 异常检测推理

**核心思想**: 模型只学过"正常"数据 → 无法准确重建"异常"数据 → 重建误差大

**推理流程**:

```python
# 1. 前向传播
recon = model(input)  # 或 recon, mu, logvar = vae(input)

# 2. 计算误差图
error_map = SmoothL1(recon, input, reduction='none')  # (B, 1, H, W)

# 3. 计算异常分数（兼顾全局和局部）
error_flat = error_map.flatten(1)                     # (B, H×W)
score_mean = error_flat.mean(dim=1)                   # 全局平均误差
score_p99  = quantile(error_flat, 0.99, dim=1)        # 局部最大误差
score = max(score_mean, score_p99)                    # 取较大者

# 4. 阈值判断
threshold = mean(scores) + 3 × std(scores)            # 3σ 法则
is_anomaly = score > threshold
```

**为什么用 max(mean, p99)?**

| 异常类型 | mean 特点 | p99 特点 | 最终 |
|---------|----------|---------|------|
| 全局异常（整张图都异常） | 高 | 高 | 检出 ✅ |
| 稀疏异常（局部小区域） | 低 | 高 | 检出 ✅ |
| 正常样本 | 低 | 低 | 正常 ✅ |

### 阈值选择方法

| 方法 | 公式 | 适用场景 |
|------|------|---------|
| **3σ** | `μ + 3σ` | 正态分布假设 |
| **IQR** | `Q3 + 1.5×IQR` | 对异常值鲁棒 |
| **百分位** | `95th percentile` | 固定比例 |

---

## 🏗️ 模型架构

### RadioAutoencoder (AE)

```
Params: ~28M
Input:  (B, 1, 512, 512)
Output: (B, 1, 512, 512)
```

### VariationalRadioAutoencoder (VAE)

```
Params: ~28M + latent layers
Input:  (B, 1, 512, 512)
Output: (B, 1, 512, 512), μ, logσ²
Latent: (B, latent_ch, 16, 16)  # 默认 latent_ch=128
```

VAE 额外优势：
- 可检测"潜在空间异常"（KL 项）
- 可生成新样本（采样 z）

---

## ⚙️ 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--h5_path` | (必填) | HDF5 数据路径 |
| `--model` | `ae` | `ae` 或 `vae` |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 32 | 每卡 batch size |
| `--lr` | 1e-4 | 学习率 |
| `--encoder_lr_mult` | 0.3 | Encoder 学习率倍率 |
| `--kl_weight` | 0.1 | VAE KL 权重 |
| `--kl_warmup_epochs` | 10 | KL 预热 epoch 数 |
| `--augment` | False | 启用数据增强 |
| `--no_pretrained` | False | 不使用 ImageNet 预训练 |

---

## 📊 输出文件

训练输出 (`./outputs/YYYYMMDD_HHMMSS/`):
```
├── epoch001.pth      # 每 epoch 权重
├── epoch002.pth
├── ...
├── best.pth          # 最佳模型（验证集 loss 最低）
├── checkpoint.pth    # 断点续训用
└── logs.npy          # 训练日志 [train_loss, val_loss]
```

推理输出 (`./inference_results/YYYYMMDD_HHMMSS/`):
```
├── results.npz              # 异常分数 + 阈值
├── score_distribution.png   # 分数分布图
├── anomalies.txt            # 异常样本列表
└── anomaly_images/          # Top-N 异常可视化
    ├── anomaly_01_idx123.png
    └── ...
```

---

## 📖 参考

- [ConvNeXt](https://arxiv.org/abs/2201.03545) - Encoder backbone
- [β-VAE](https://arxiv.org/abs/1804.03599) - Disentangled VAE
- [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) - 鲁棒损失函数

---

## 📝 License

MIT License
