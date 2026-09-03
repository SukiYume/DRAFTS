# Radio Telescope Anomaly Detection

这个目录实现基于 AE/VAE 重建误差的射电时频图异常检测，也支持用前一个切片预测下一个切片的时间一致性模式。模型使用 ConvNeXt-Tiny 或 ResNet18 编码器，以及带门控跳连的对称解码器。目录包含完整的数据准备、模型、训练、推理、依赖清单和本地权重，可以单独复制运行。

## 保留内容

- `prepare_data.py`：递归读取 PSRFITS，生成 HDF5 切片。
- `dataset_h5.py`：单切片数据集、严格连续的相邻切片数据集和数据增强。
- `models.py`：AE/VAE、训练损失和统一异常分数。
- `train_ddp.py`：单进程或 `torchrun` DDP 训练。
- `run_inference.py`：HDF5/FITS 推理、阈值判断和可视化。
- `extract_latents.py`：VAE 潜变量导出及 UMAP/t-SNE 图。
- `train.sh`：集中配置的 Linux 训练入口。
- `best_loss_model.pth`：本地保留的 ConvNeXt、128 潜通道 VAE 权重。

历史运行目录、临时 HDF5 和缓存不属于源码。默认输出目录均已忽略；大规模输入和结果应放在仓库外。

## 环境

需要 Python 3.10 或更新版本，以及 PyTorch 2.3 或更新版本。

```bash
pip install -r requirements.txt
```

`fitsio` 用于快速读取；无法使用时，预处理会回退到 Astropy。ConvNeXt/ResNet 的 ImageNet 初始化可能需要首次下载权重，离线环境请传 `--no_pretrained`。

## 数据准备

命令行可以接收一个或多个 FITS 文件、目录或通配符。目录会递归扫描 `.fit`/`.fits` 文件，不再依赖文件名包含特定波束编号。

```bash
python prepare_data.py /data/observation_a /data/observation_b \
    --output /data/anomaly/train.h5 \
    --down_times 8 16 32 64 128 256 \
    --num_workers 8
```

也可在 Python 中调用：

```python
from prepare_data import build_h5_from_fits_paths

h5_path, factors, workers = build_h5_from_fits_paths(
    ["/data/observation_a", "/data/observation_b"],
    out_h5_path="/data/anomaly/train.h5",
    down_time_list=(8, 16, 32, 64, 128, 256),
    num_workers=8,
)
```

每个 PSRFITS 的处理顺序是：

1. 将 `DATA` 重排为 `(time, polarization, frequency)`，平均前两个偏振；
2. 将相邻频率通道平均到 512 个通道；
3. 分别按请求倍率做时间平均；
4. 切成 `(512 time, 512 frequency)` 的非重叠块；
5. 每个块独立用有限值的 1%–99% 分位裁剪，并线性缩放到 `[0, 1]`。

某个倍率所需的时间样本不足时只跳过该倍率，不影响同一文件的较小倍率。常数切片会得到全零；非有限值会先用该切片的有限中位数替换。输出采用临时文件写入并在成功后原子替换目标，失败不会留下一个伪装成完整结果的 HDF5。

HDF5 契约：

| 数据集 | 形状 | 含义 |
|---|---:|---|
| `images` | `(N, 512, 512)` | `float32`，轴顺序为 `(time, frequency)` |
| `names` | `(N, 1)` | 原始 FITS 完整路径 |
| `down_time` | `(N, 1)` | 时间下采样倍率 |
| `slice_idx` | `(N, 1)` | 该 FITS、该倍率内的切片序号 |

## 训练

单卡 AE：

```bash
python train_ddp.py \
    --h5_path /data/anomaly/train.h5 \
    --model ae \
    --encoder convnext \
    --epochs 50 \
    --batch_size 32 \
    --augment
```

八卡 VAE：

```bash
torchrun --standalone --nproc_per_node=8 train_ddp.py \
    --h5_path /data/anomaly/train.h5 \
    --model vae \
    --encoder convnext \
    --latent_dim 128 \
    --kl_weight 0.1 \
    --kl_warmup_epochs 10 \
    --epochs 50 \
    --batch_size 32 \
    --augment
```

相邻切片预测模式在上述任一模型上增加 `--pred`。一对样本必须满足：完整 `name` 相同、`down_time` 相同且 `slice_idx` 严格相差 1；代码不会跨 FITS 文件或跨缺口配对。

训练/验证划分也按完整 FITS 路径分组，至少有两个源文件时，同一文件的切片或相邻对不会同时落入两侧；只有一个源文件时才退回样本级划分。

```bash
python train_ddp.py \
    --h5_path /data/anomaly/train.h5 \
    --model vae \
    --pred \
    --down_times 32
```

`train.sh` 提供同一组参数的集中配置。它从 `CUDA_VISIBLE_DEVICES` 自动计算进程数，单卡和多卡走同一份参数数组，失败时不会误报“训练完成”。

每次训练写入 `OUTPUT_ROOT/YYYYMMDD_HHMMSS/`：

```text
Epoch001_Tloss..._Vloss....pth  # 每轮纯模型权重
best_loss_model.pth             # 最低验证损失的纯模型权重
last_checkpoint.pth             # 模型、优化器、调度器、scaler 和配置
logs.npy                        # 本次进程的 [train_loss, val_loss]
```

`--resume` 接收 `last_checkpoint.pth`。最佳权重和每轮权重可以直接用于推理；完整 checkpoint 也可直接用于推理，程序会读取其中的模型权重并核对配置。

## 异常分数

对每个样本先计算逐像素 SmoothL1 误差 `e`：

```text
reconstruction_score = max(mean(e), percentile_99(e))
AE_score  = reconstruction_score
VAE_score = reconstruction_score + beta * mean(KL(mu, logvar))
```

VAE 只在训练状态从后验采样；`eval()` 时固定解码 `mu`，因此同一权重和输入会得到可复现分数。预测模式将前一切片作为输入、当前切片作为目标，输出索引和元数据都对应当前目标切片。

## 推理

使用本地保留权重从 FITS 运行：

```bash
python run_inference.py \
    --fits_dir /data/observation \
    --model_path ./best_loss_model.pth \
    --model vae \
    --encoder convnext \
    --latent_dim 128 \
    --output_dir /data/anomaly/inference \
    --plot_anomalies
```

从已有 HDF5 运行：

```bash
python run_inference.py \
    --h5_path /data/anomaly/test.h5 \
    --model_path /models/best_loss_model.pth \
    --model vae \
    --kl_weight 0.1 \
    --batch_size 32
```

对已有 HDF5，省略 `--down_times` 表示使用全部倍率；提供该参数时只推理指定倍率。直接从 FITS 推理且省略该参数时，默认生成 `8 16 32 64 128 256`。

阈值可选 `3sigma`、`iqr` 或 `percentile`。这些方法从当前被评估数据估计阈值，适合探索，但会受其中异常样本污染。生产或跨观测比较应从独立正常校准集得到阈值，并用 `--threshold VALUE` 固定。

每次推理输出：

```text
results.npz             # 总分、重建分、KL 分、判定和完整 HDF5 元数据
score_distribution.png # 分布与排序图
anomalies.txt           # 可追溯到 FITS、倍率、slice_idx 的异常表
anomaly_images/         # 可选的输入/目标/重建/差值图
```

预测模式增加 `--pred`，且必须使用以同一模式训练的权重。

## 潜变量

```bash
python extract_latents.py \
    --h5_path /data/anomaly/test.h5 \
    --model_path ./best_loss_model.pth \
    --encoder convnext \
    --latent_dim 128 \
    --kl_weight 0.1 \
    --method umap \
    --output_dir /data/anomaly/latents
```

潜变量工具在一次前向传播中同时取得 `mu`、重建和统一异常分数，避免重复编码同一批数据。UMAP/t-SNE 至少需要三个样本。
