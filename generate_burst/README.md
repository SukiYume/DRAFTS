# DRAFTS CenterNet 训练 H5 生成

本目录把真实 FAST 背景数据和模拟 FRB 注入结合起来，生成 `object_detection/` 可直接读取的 CenterNet 训练 H5。当前主线是 multifitting 风格模拟信号、0-4096 DM 画布、512 x 512 time-DM 图像和最多 3 个目标框。

## 文件职责

| 路径 | 用途 |
| --- | --- |
| `generate_dataset.py` | 主生成器。读取背景 FITS，注入模拟 FRB，做消色散和裁切，写 H5、配置、metadata 和逐图 bounding-box 标注。 |
| `run_generation.sh` | 批量入口。默认生成 50,000 个 unique signals，每个信号 4 个 crop，合并成约 200,000 张训练图。 |
| `merge_shards.py` | 合并 shard H5，并同步生成合并后的 annotation/config/metadata/inspect 文件。 |
| `inspect_dataset.py` | 快速检查 H5 内容，输出 JSON 摘要和 contact sheet。 |
| `simulation_utils.py` | FRB 动态谱基本函数和 FAST 背景处理工具。物理参数采样在 `generate_dataset.py` 中完成。 |
| `d-center-binary-core.py` | 生成器复用的 GPU 消色散核心；文件名有意与 `runcode` 保持一致。 |
| `rawdata/` | 背景 FITS 位置。大数据不随仓库保存，运行时用目录、软链接或 `RAW_DIR` 指定。 |
| `shards_50000/` | 批量生成产物目录。属于可重算产物，不作为源码维护。 |

Git 只跟踪上述代码和 README；背景 FITS、shard、H5、缓存、日志和检查图均由
`.gitignore` 排除，不属于可提交的生成器源码。

## 当前生成配置

`run_generation.sh` 是当前推荐入口。它按 40 个 shard 生成数据，默认每轮并发 4 个 shard，并把任务分配到 GPU 0-7。

关键默认值：

| 参数 | 当前值 |
| --- | --- |
| unique signals | `50000` |
| crops per signal | `4` |
| signals per scene | `3` |
| max objects per image | `3` |
| background file range | `11..275` |
| DM canvas | `0..4096` |
| DM scale | `1.0` |
| scene output width | shard 间轮换 `4096`、`8192`、`16384` |
| min split effective S/N | `1.5` |
| physical fallback min S/N | `3.0` |

## 信号模型

训练集不直接在 time-DM 图上画亮斑。生成器先把模拟 FRB 注入真实 raw8 Stokes-I 背景，再用 DRAFTS 消色散核生成 0-4096 DM canvas，最后裁切和 area-resample 成 512 x 512 detector 图。

单个信号的动态谱模型：

| 项 | 构建方式 |
| --- | --- |
| 参考时间 | `toa_raw` 是最高频通道的到达采样。 |
| 色散延迟 | `t(f) = toa_raw + K_DM * DM * (f^-2 - f_ref^-2) / time_reso`，其中 `f_ref=max(freq)`，`K_DM=4148.808 s MHz^2 pc^-1 cm^3`。 |
| 频率包络 | 高斯包络 `exp(-0.5*((f-center_freq)/sigma_freq)^2)`，`sigma_freq = bandwidth_fwhm / 2.355`。 |
| 时间轮廓 | 本征高斯宽度 `sigma_time = width_fwhm / 2.355`。有散射时使用高斯卷积单边指数尾的 ex-Gaussian，并按通道峰值归一到 1。 |
| 散射频率标度 | 先采 `tau_1ghz`，再算 `tau(f)=tau_1ghz*(f/1000 MHz)^-4`。代码内部等价地存 `scattering_ms_at_highest_freq`，再从最高频缩放到各通道。 |
| 单通道幅度 | `per_channel_peak_snr = snr / sqrt(effective_channels_fwhm)`；`effective_channels_fwhm` 是频率包络大于 0.5 的通道数。 |
| 注入值 | `delta = per_channel_peak_snr * frequency_envelope * pulse_profile * channel_std`，并入背景后 clip 到 `[0,255]`。 |

## 参数范围和分布

这些分布用于检测器训练，范围刻意比注入评估更宽，覆盖低 DM、弱信号、宽时宽和强散射情况。

| 参数 | 范围 | 分布 |
| --- | --- | --- |
| `dm_pc_cm3` | `[5, 4000]` | 均匀分布。 |
| `snr` | `[1.5, 120]` | 对数均匀分布。 |
| `width_ms_fwhm` | `[0.5, 50] ms` | 对数均匀分布。 |
| `bandwidth_mhz_fwhm` | `[50, 500] MHz` | 对数均匀分布。 |
| `center_freq_mhz` | 观测频带 `[freq_min, freq_max]` | 均匀分布；超出观测带的频率尾自然截断。 |
| `scattering_ms_at_1ghz` | `0` 或 `[0.03, 80] ms` | 15% 精确为 `0`；其余对数均匀分布。 |
| `toa_down` | scene 内合法 guard 区间 | 均匀分布；guard 保证完整色散响应和散射尾落在 scene 输出窗口内。 |

每个物理信号会派生多个 detector crop。crop 视角属于训练增强：

| 参数 | 取值 | 分布 |
| --- | --- | --- |
| `crop_dm_span` | `512, 768, 1024, 1536, 2048, 3072, 4096` | 离散均匀抽样。 |
| `crop_time_width` | `1024, 2048, 4096, 8192, 16384` 中不超过当前 `scene_output_down` 的值 | 离散均匀抽样。 |
| `split_index` | `full/lower/upper` | 先以 55% 选 `full`；其余 45% 按中心频率偏向对应半带，75% 选匹配半带、25% 回到 `full`。 |
| x 方向落点 | 512 像素图中的横坐标 | 42% 贴低/高边，58% 落在 `[96,416]` 中央区。 |
| y 方向落点 | 512 像素图中的纵坐标 | 48% 贴低/高边，52% 落在 `[96,416]` 中央区。 |

每次运行会把这些范围写入输出 H5 旁边的 `.config.json`，每张图的实际参数写入 `.metadata.jsonl`。

## 生成完整训练集

```bash
cd DRAFTS/generate_burst
RAW_DIR=/path/to/rawdata \
PY=/path/to/miniconda3/bin/python \
SHARDS_PER_WAVE=4 \
./run_generation.sh
```

脚本会先写每个 shard，再调用 `merge_shards.py` 合并，最后调用 `inspect_dataset.py` 做抽样检查。

主要输出：

| 文件 | 说明 |
| --- | --- |
| `centernet_dataset_sim50000_max3.h5` | 合并后的训练 H5。 |
| `centernet_dataset_sim50000_max3_annotations.json` | 从 H5 导出的逐图 bounding-box JSON，便于独立检查。 |
| `centernet_dataset_sim50000_max3.config.json` | 生成和合并参数。 |
| `centernet_dataset_sim50000_max3.metadata.jsonl` | 每张图的来源、注入和裁切信息。 |
| `centernet_dataset_sim50000_max3.inspect.json` | 抽样检查摘要。 |
| `centernet_dataset_sim50000_max3_visual_inspect.json` | contact-sheet 抽样所用帧及其统计。 |
| `centernet_dataset_sim50000_max3_contact.png` | 可视化 contact sheet。 |
| `centernet_dataset_sim50000_max3.merge.log` | shard 合并日志。 |
| `shards_50000/shard_*.h5` / `shard_*.log` | 合并前的 shard 数据和各 shard 运行日志。 |

生成完成后，把 H5 复制到 `../object_detection/Data/`，再进入
`../object_detection/` 训练检测器。训练代码直接读取 H5 内的 `annotations`；
外部 `_annotations.json` 只用于独立检查，不是训练必需文件。

## 小规模检查

先用 dry-run 确认参数、输入文件和输出路径：

```bash
cd DRAFTS/generate_burst
python generate_dataset.py \
  --rawdata-dir ./rawdata \
  --output ./test_multifit.h5 \
  --unique-signals 100 \
  --crops-per-signal 4 \
  --dry-run
```

实际写一个小 H5：

```bash
python generate_dataset.py \
  --rawdata-dir ./rawdata \
  --output ./test_multifit.h5 \
  --unique-signals 100 \
  --crops-per-signal 4 \
  --signals-per-batch 200 \
  --max-objects-per-image 3
```

检查输出：

```bash
python inspect_dataset.py \
  ./test_multifit.h5 \
  --json ./test_multifit.inspect.json \
  --contact-sheet ./test_multifit_contact.png
```

## H5 内容

主 H5 中保留训练需要的图像和标注字段：

- `images`：512 x 512 time-DM 图像。
- `annotations`：形状为 `(M, 5)`；每行依次是
  `image_index, left, top, width, height`。
- `original_filename`、`original_path`、`original_slice`：背景文件和切片来源。

类别、DM/time 位置、信号物理参数和背景来源等扩展信息保存在配套
JSON/JSONL 中；复查训练数据时优先结合 metadata 和 contact sheet。

## 运行注意事项

- 重新跑同一输出目录前先确认旧 shard 是否要保留，避免新旧 shard 混合。
- 如果只改训练策略，优先在 `../object_detection/` 改；本目录只负责生成检测器训练数据。
