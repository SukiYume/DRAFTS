# DRAFTS 真实数据搜索入口

本目录是一套可以拷到服务器独立运行的 FRB 单脉冲搜索 runtime。运行时只需要 Python/CUDA 环境、FAST FITS 数据、模型权重和输出目录。

当前默认搜索链路是 **CenterNet ConvNeXt-Tiny detector v10 + ConvNeXt-Small binary classifier**。
所有搜索入口、模型定义、调度脚本、fixed-DM 工具和依赖清单均位于本目录内。模型权重放入本目录的 `models/`。

## 入口选择

| 场景 | 推荐入口 | 说明 |
|---|---|---|
| 命令行盲搜测试 | `t-blind-section.py` | 参数完整，适合单 section / 多 section 调试。 |
| 原始分辨率 GPU 生产盲搜 | `d-center-binary-gate.py` | 脚本底部配置物理 DM、路径和 GPU section；细通道消色散后再下采样。 |
| 已知 DM / 候选 DM follow-up | `d-dm-time.py` | 原始采样 GPU 消色散后下采样，多倍率 binary 分类，支持末尾短文件。 |
| 压缩观测 fixed-DM 批处理 | `search_xzdata/run_fixed_dm_xz.sh` | 校验并解压 `.fits.xz`、按 manifest 分 section 提交搜索。 |
| gate / d-dm-time PBS 提交 | `s-pbsspt.py` | 自动读取 section 数，支持配置每块 GPU 的搜索进程并发上限。 |
| fixed-DM binary 对比 | `t-binary-bench.sh` | 比较不同 binary 分类器的真实输出。 |

推荐先用 `t-blind-section.py --dry-run` 确认数据分片和输出路径，再跑实际搜索。生产运行时
修改 `d-center-binary-gate.py` 底部配置，并用 `s-pbsspt.py` 提交 PBS。

## 文件职责

| 文件或目录 | 作用 |
|---|---|
| `d-center-binary-gate.py` | 两阶段盲搜调度入口：组织 FITS、section 切分、加载模型、断点日志，并使用物理 DM 与动态 RFI 配置调用 core。 |
| `d-center-binary-core.py` | 原始时间/频率分辨率 GPU 两阶段盲搜核心：细通道消色散、下采样、DM-time 图、CenterNet 检测、candidate cutout、binary 分类和保存。 |
| `d-dm-time.py` | GPU fixed-DM：原始整数上传、细通道消色散后平均、多尺度零拷贝分类、短文件窗口补齐。 |
| `search_xzdata/` | 可复用的 `.fits.xz` fixed-DM 控制器；目标专用 preset 和 manifest 仅保存在本地。 |
| `t-blind-section.py` | 命令行式盲搜 section wrapper。 |
| `t-blind-batch.sh` | 多 GPU 盲搜批量测试 wrapper。 |
| `t-binary-bench.sh` | fixed-DM binary 模型输出对比。 |
| `s-pbsspt.py` | gate / d-dm-time PBS 提交器；均分 section，限制每块 GPU 的并发搜索进程数，并支持 dry-run 与状态查询。 |
| `c-data-check.py` | 拼接若干 FITS 后画频率时间图，做数据检查。 |
| `c-manifest-build.py` | 预扫描 CRAFTS 数据根目录，生成 gate 可直接加载的任务 manifest。 |
| `c-manifest-summary.py` | 汇总各 beam 输出目录下的 `candidate_manifest.jsonl` 为 CSV/JSON。 |
| `extract_xz.sh` | 批量解压 `.xz` FITS。 |
| `binary_model.py` | binary 分类器模型定义。 |
| `centernet_model.py` / `centernet_eval.py` | CenterNet 构建和 heatmap 解码。 |
| `models/` | 本目录默认读取的部署权重及下载说明。 |
| `requirements.txt` | 搜索 runtime 的 Python 依赖清单。 |

## 环境

```bash
python -m pip install -r requirements.txt
```

CuPy 不在通用依赖清单中：请单独安装与目标节点 CUDA toolkit 匹配的构建，例如
`cupy-cuda12x`。PyTorch 与 torchvision 同样应选择和目标 CUDA 环境匹配的 wheel。

正式搜索前检查 GPU 环境：

```bash
python - <<'PY'
import torch, cupy
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print(cupy.cuda.runtime.getDeviceCount())
PY
```

## 模型权重

当前默认运行只需要两份权重：

| 模型 | 文件名 | SHA-256 |
|---|---|---|
| CenterNet + ConvNeXt-Tiny detector v10 | `object_best_model_centernet_conv_tiny_ema_v10.pth` | `bcad4e710f5f1ccd3c8609d35a8d3fbfc36abd1d85bfefd035e945a573fb0629` |
| ConvNeXt-Small binary classifier | `binary_best_model_conv_small_ema.pth` | `2055745aab76ddc16074516aa7b9aafdfaedf16df37ce8924389573eab27ffd8` |

默认权重公开发布在
[DRAFTS Hugging Face 模型仓库](https://huggingface.co/TorchLight/DRAFTS)，
可直接下载到 `models/`：

```bash
mkdir -p models
curl -fL \
  -o models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  https://huggingface.co/TorchLight/DRAFTS/resolve/main/object_best_model_centernet_conv_tiny_ema_v10.pth
curl -fL \
  -o models/binary_best_model_conv_small_ema.pth \
  https://huggingface.co/TorchLight/DRAFTS/resolve/main/binary_best_model_conv_small_ema.pth
```

默认入口将这两个文件分别作为 `centernet_conv_tiny` detector 和
`convnext_small` classifier 加载。更换权重时需要同时保持模型类型、backbone 和
checkpoint 架构一致。

支持的 detector：

```text
centernet_conv_tiny
centernet_conv_small
```

支持的 binary backbone：

```text
convnext_tiny
convnext_small
```

## FITS 数据布局

入口支持两类路径：

```text
source/date/*.fits
source/date1/*.fits
source/date2/*.fits
```

如果 `data_path` 直接包含 FITS，脚本按文件名中的 `-Mxx_` 聚合 beam。如果 `data_path`
指向 source 根目录，脚本递归扫描日期子目录。`beam_filter='M01'` 或 `--beam M01` 只跑
M01，`all` 跑全部 beam。

脚本会跳过文件名中包含 `_N_`、`_W_`、`_F_` 的 FITS。

### 望远镜、频段与固定通道选择

当前 runtime 面向 FAST L 波段 PSRFITS。频率轴本身由 `OBSFREQ`、`OBSBW` 和
`NCHAN` 读取，检测输入固定下采样为 512 个通道；但消色散检测还沿用了针对
4096-channel FAST 数据设定的固定有效通道段 `[10, 650)` 与 `[820, 4050)`，并按
`NCHAN/4096` 缩放索引。对标称 1000–1500 MHz/4096-channel 数据，这约对应
1001.2–1079.3 MHz 与 1100.1–1494.3 MHz。二分类 cutout 仍使用 full-frequency
数据。

用于其他望远镜、接收机频段、频率轴方向或坏频段配置时，先按目标观测修改
`d-center-binary-core.py::_build_dedispersion_cache()` 的 `index_array`，再用已知
脉冲和注入基准验证召回、DM/TOA 误差；`NCHAN` 还必须能整除到 512 通道。

### 损坏 FITS 的连续搜索策略

`d-center-binary-gate.py` 按文件容错：某个 FITS 读取失败时，把异常写到与 completed
log 同名前缀的 `*_bad_fits.log`，以相同 shape 的随机噪声替换该文件，并继续搜索后续
FITS。`process_fits_list()` 遍历完成后，gate 将整个 section 写入 completed log。

随机替换覆盖的窗口不含真实观测内容。归档结果时一并检查 `*_bad_fits.log`；修复原
FITS 后，可清理该 section 的完成记录及对应输出并重跑。

## 生产盲搜数据流

每个 `d-center-binary-gate.py` 进程按以下顺序处理自己的任务：

1. gate 从 task manifest 或 `data_paths` 得到按 source/date/beam 排序的 FITS 列表，并按
   `section_num` 分配任务。单个连续观测跨 section 时会依据物理 DM 网格自动补齐前、后
   色散读取余量；重叠文件只提供上下文。
2. CPU 读取当前滑动窗口。常规多偏振 FITS 在解码时合并前两路偏振，2-bit FITS 按 packed
   格式解码。缓存以 `uint16` 保存原始时间和频率采样。
3. `fits_prefetch_workers` 个读取线程同时准备当前窗口和后续文件。GPU 计算与下一批 FITS
   读取可以重叠，滑动窗口缓存让相邻窗口复用已解码文件。
4. 每个原始分辨率窗口上传 GPU 一次。动态 RFI、细通道消色散、时间下采样、`NCHAN → 512`
   频率平均、DM-time 归一化和 CenterNet 检测都在 GPU 上完成。
5. detector 候选按 `classifier_batch_size` 生成消色散时频 cutout，并交给 binary 分类器。
   精确共享同一起始采样和细通道时移的候选复用 cutout。
6. 通过两级阈值的候选写入 JPG、NPY 和 `candidate_manifest.jsonl`。候选坐标、模型分数和
   最终保存产物会回到 CPU。

CuPy 与 PyTorch 通过 DLPack 共享 GPU 缓冲。DM-time 图、binary cutout 与模型输入保持在
显存中，批量大小限制单个时间片的峰值显存占用。

前缀和缓冲按 `prefix_channel_chunk` 个细通道分块复用，默认值为 `512`；原始输入和
三幅 DM-time 输出保持完整精度。转置 kernel 的时间维使用 `grid.x`，通道维使用
`grid.y`，以支持超过 65535 个时间 tile 的长输入。

末尾短 FITS 的输出长度向上补齐至 `block_size` 的整数倍，原始输入复用现有随机噪声
补齐逻辑。例如 8960 点补齐为 12288 点后，按起点 0、4096、8192 的三个常规窗口处理；
切片、时间坐标、NPY 保存和 manifest 字段沿用统一流程。

`fits_prefetch_workers=1` 使用单读取线程；更大的值会增加并行读取数和后续文件预取深度。
每个搜索进程拥有独立的预取池和主机缓存。多 GPU 或多 worker 运行时，应结合存储吞吐、
主机内存和 GPU 处理速度选择该值，默认值为 `2`。

RFI 统计时间分辨率固定为 393.216 us（49.152 us 数据每 8 点、98.304 us 数据每 4 点），
频率轴固定汇总为 512 个 coarse channels。该统计分辨率独立于 detector 使用的
`time_factor`。`NCHAN` 可以是 512 的任意正整数倍；GPU 消色散完成后统一平均为 512
通道。`OBSBW < 0` 的 FITS 会在解码时翻转为升频序。

## 未知 DM 盲搜

### 命令行 section 测试

```bash
CUDA_VISIBLE_DEVICES=0 python t-blind-section.py \
  --section 0 \
  --gpu-num 1 \
  --data-path /path/to/fast_observation/source/date \
  --output-root /path/to/search_outputs/blind \
  --beam M01 \
  --detector-type centernet_conv_tiny \
  --detector-ckpt models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  --classifier-ckpt models/binary_best_model_conv_small_ema.pth \
  --dm-min 0 \
  --dm-range 4096 \
  --dm-step 1 \
  --dm-threshold 10 \
  --block-size 4096 \
  --det-prob 0.45
```

常用参数：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--gpu-num` | 8 | section 总数。单卡调试设为 1。 |
| `--dm-min` | 0.0 | 搜索起点，单位为 pc cm^-3。 |
| `--dm-range` | 4096.0 | 搜索区间的物理 DM 宽度，单位为 pc cm^-3。 |
| `--dm-step` | 1.0 | 相邻 trial 的物理 DM 间隔，单位为 pc cm^-3。 |
| `--dm-threshold` | 50.0 | 低于该 DM 的候选丢弃；仓库内 gate 模板显式设为 10。 |
| `--block-size` | 8192 | 每个时间块的降采样后样本数；仓库内 gate 模板显式设为 4096。 |
| `--det-prob` | 0.45 | CenterNet 候选阈值。 |
| `--time-factor` | 8.0 | FITS 时间降采样率。 |

多 GPU 可以分别提交 `--section 0..gpu_num-1`，或使用 `t-blind-batch.sh`。`--gpu-num` 在这里表示总 section 数，不一定等于物理 GPU 数。

开发节点上直接跑多 section 示例：

```bash
ROOT=/path/to/search_pipeline \
OUTPUT_ROOT=/path/to/search_outputs/blind \
BEAM=all GPU_NUM=8 \
DM_MIN=0 DM_RANGE=4096 DM_STEP=1 DM_THRESHOLD=10 BLOCK_SIZE=4096 DET_PROB=0.45 \
  bash t-blind-batch.sh /path/to/fast_observation/
```

由 PBS 管理的节点使用 `s-pbsspt.py` 生成并提交作业。

### 固定配置 gate

`d-center-binary-gate.py` 底部提供一个使用通用占位路径的 CRAFTS 配置模板：

```python
process_config = ProcessConfig(
    dm_min=0,
    dm_range=4096,
    dm_step=1,
    dm_threshold=10,
    block_size=4096,
    det_prob=0.45,
    section_num=32,
    time_factor=8,
    fits_prefetch_workers=2,
    rfi_filter=True,
    rfi_level_ratio=0.05,
    rfi_cv_ratio=3.0,
    rfi_zero_dm_filter=True,
    rfi_zero_dm_sigma=8.0,
    rfi_zero_dm_occupancy=0.03,
)

data_path = None
data_paths = [
    "/path/to/CRAFTS/",
]
task_manifest = "/path/to/observations/CRAFTS/task_manifest.json"
save_base = "/path/to/observations/CRAFTS/"
beam_filter = "all"
```

如果 `task_manifest` 指向一个存在的文件，gate 会直接加载预构建任务；否则才扫描
`data_paths`。大规模目录建议先构建 manifest，减少每个 section 重复扫描文件系统：

```bash
python c-manifest-build.py \
  --data-path /path/to/CRAFTS/ \
  --beam-filter all \
  --output /path/to/observations/CRAFTS/task_manifest.json
```

### 物理 DM 参数

`d-center-binary-gate.py` 加载 `d-center-binary-core.py`，使用实际 DM 值定义搜索区间：

```python
process_config = ProcessConfig(
    dm_min=0,          # 搜索起点
    dm_range=4096,     # 实际物理 DM 宽度
    dm_step=1,         # 相邻 trial 的物理 DM 间隔
    block_size=4096,
    time_factor=8,
    fits_prefetch_workers=2,
)
```

`dm_range` 是 pc cm^-3 单位的物理区间宽度。搜索区间为半开区间
`[dm_min, dm_min + dm_range)`；上例覆盖 `[0, 4096)`，内部派生出 4096 个 trial。
每张检测图固定包含 1024 个 trial、平均成 512 行，所以 gate 默认生成 4 张
分别覆盖 0--1024、1024--2048、2048--3072 和 3072--4096 的检测图。

`dm_range / dm_step` 必须是正整数且是 1024 的整数倍。配置不能完整切块时
程序会对无法完整切块的配置报错，并给出最近的合法 `dm_range`，保持检测图的有效范围。
常规 4096-DM 盲搜写作 `dm_min=0, dm_range=4096, dm_step=1`。
当搜索区间包含负 DM 时，gate 会根据 FITS 头中的频率范围、`TBIN` 和单文件长度
自动计算前向、后向 overlap；前置 FITS 只提供消色散余量，不会被重复搜索。观测
起点确实缺少前置数据时使用噪声补齐，不会复制首个 FITS 形成周期性伪结构。

`c-manifest-build.py` 接受重复的 `--data-path`，可把多个数据根写入同一份清单。生成后
把 gate 中的 `task_manifest` 指向该 JSON；更换数据集时应重新生成任务清单。

直接运行某个 section：

```bash
CUDA_VISIBLE_DEVICES=0 python d-center-binary-gate.py 0
```

PBS 批量运行时使用 `s-pbsspt.py`。提交器读取 gate 的 `section_num`，
把全部 section 编号均匀分配给可用 GPU。`--workers-per-gpu` 控制每个 GPU 作业的
并发搜索进程上限，默认值 1 会逐个完成自己的 section。`section_num` 可以独立于 GPU 数设置。

`d-center-binary-gate.py` 默认使用 `processing_log.txt`，每行记录一个已完成
identifier。重跑全集时删除对应日志：

```bash
rm -f processing_log.txt
```

未在完成日志中的 beam 级任务重跑前会自动删除该 beam 的旧输出目录；单次观测的 section 分段不执行自动清理。

## 盲搜输出

盲搜输出布局：

```text
<output-root>/<source>/CentData/<date>/<beam>/
```

典型文件：

```text
FRB20220912A_tracking-M01_0352-TS00-FS1-BX0-DM228.0.jpg
FRB20220912A_tracking-M01_0352-TS00-FS1.npy
candidate_manifest.jsonl
```

字段含义：

| 字段 | 含义 |
|---|---|
| `0352` | 起始 FITS 文件号。 |
| `TS00` | 时间 slice 编号。 |
| `FS1` | 频带拆分通道。 |
| `BX0` | 检测图中的候选编号。 |
| `DM228.0` | 检测器中心换算出的 DM。 |

`candidate_manifest.jsonl` 每行记录一个通过 detector + binary 阈值的候选。关键字段：

| 字段 | 含义 |
|---|---|
| `root` / `source` / `date` / `beam` | 候选所属数据根、源目录、日期和 beam。 |
| `block_start_mjd` | 搜索 block 起始 FITS 的 MJD。 |
| `toa_sec` | 候选相对 `block_start_mjd` 的秒数。 |
| `signal_mjd` | 候选绝对 MJD，等于 `block_start_mjd + toa_sec / 86400`。 |
| `dm` | 检测图中心换算出的 DM，可用于初步切出信号。 |
| `jpg_path` / `npy_path` | 对应 review 图和 DM-time numpy 文件。 |

review 图的两幅时间轴分别判断总时长：小于 1000 ms 时使用毫秒，达到 1000 ms
时自动改用秒；这只改变刻度显示，不改变 detector 或 binary 的输入时间范围。

合并所有 beam 的 manifest：

```bash
python c-manifest-summary.py \
  --root /path/to/observations/CRAFTS \
  --csv c-candidates-zd202x.csv \
  --json c-candidates-zd202x.json
```

后续使用 [AFTER](https://github.com/SukiYume/AFTER) 裁切原始数据时，先用
`signal_mjd` 减去整次观测起始 MJD：

```python
cut_toa_sec = (signal_mjd - obs_start_mjd) * 86400.0
```

正式运行使用仓库之外的独立运行目录保存搜索输出。

## Fixed-DM Follow-up

`d-dm-time.py` 适合：

- 已知重复源 DM，回看一批观测。
- 盲搜发现候选后，在候选 DM 附近复查。
- 比较不同 binary 分类器和时间下采样倍率。

脚本底部需要修改：

```python
config = ProcessConfig(
    DM=273.5,
    prob=0.6,
    block_size=512,
    section_num=32,
    down_sampling_rate_list=np.array([8]),
)

classifier_ckpt = "./models/binary_best_model_conv_small_ema.pth"
data_path = "/path/to/fast_observation/source/date/"
save_base = "/path/to/drafts_search_output/"
beam_filter = "M01"
```

`prob=0.6`、`section_num=32` 和 `[8]` 是通用入口默认值。多尺度复查可把
`down_sampling_rate_list` 设置为多个正整数。实现先按细通道原始采样时移消色散，
公共时间平均倍率为各倍率的最大公约数。相邻窗口复用 GPU 整数缓存，后台线程有界预读 FITS。

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python d-dm-time.py 0
```

Fixed-DM 输出布局：

```text
<save_base>/<source>/CalData/<date>/<beam>/<down_sampling_rate>/
```

`save_base` 可以指向公共输出根，也可以直接指向 source 目录。末级目录名已经等于
`source` 时，入口直接在其下创建 `CalData/`，路径中只出现一层 source。

典型文件：

```text
FRB20240114A_tracking-M01_0086-0014-550.453248.jpg
FRB20240114A_tracking-M01_0086-0014.npy
```

批量处理压缩观测时使用 [`search_xzdata/`](search_xzdata/README.md)。通用控制器从环境变量和
TSV manifest 读取目标、DM、模型、PBS 节点与 section 数；含真实观测路径和集群配置的
目标专用 preset、manifest 应仅保存在本地，不纳入版本控制。

## 数据准备

解压 `.xz`：

```bash
./extract_xz.sh /path/to/xz_dir/ /path/to/fits/
./extract_xz.sh /path/to/xz_dir/ /path/to/fits/ all
./extract_xz.sh ./files.txt /path/to/fits/
CONCURRENT_TASKS=20 ./extract_xz.sh /path/to/xz_dir/ /path/to/fits/
```

快速检查 FITS：

```bash
python c-data-check.py 0
```

`0` 表示 M01，`1` 表示 M02。使用前先改脚本底部的数据路径和输出目录。

## PBS 批量提交

### GPU fixed-DM 入口

`d-dm-time.py` 通过底部配置指定 DM、观测路径、输出路径与模型权重，通过位置参数指定
section 编号，候选输出使用 `CalData/<date>/<beam>/<rate>/` 布局。
配置类的 `classifier_batch_size` 默认 16，通用入口显式设为 32，可按单卡显存降低。
`fits_prefetch_workers=2` 控制后台预读，设为 0 使用同步读取。入口依赖 CUDA、CuPy、PyTorch 和
现有 FITS/绘图依赖。原始输入及重叠文件缓存保存在 GPU，显存需求随文件长度、通道数、
DM 读取余量和并发进程数增加。

CPU 负责读取原始整数 FITS。8-bit 偏振合并、2-bit 解包均在 GPU 完成；输入缓存使用
uint16。融合 CUDA kernel 按细频率通道的原始采样时移取值，再完成时间和频率平均。
多倍率搜索对每个输入块只执行一次消色散，公共时间平均倍率取所有倍率的最大公约数。
各倍率从公共 512 通道数据继续做整数倍时间平均，并在独立缓冲中预处理和分类。
CuPy 输出通过 DLPack 零拷贝送入 PyTorch，GPU 按图完成基线归一化、5/95% 分位截断和
分类。分类分数、命中编号及需要保存的候选图像复制回 CPU。

末尾短文件按真实长度向上补齐完整 512 点检测窗口；色散余量和补齐噪声在 GPU 生成。
保存候选时检查峰值位于真实数据区域。单观测切分时保留原始首文件的标称长度，保持
末尾短文件的 TOA 换算正确。任意文件组处理失败，入口最终返回非零退出码。

```bash
CUDA_VISIBLE_DEVICES=0 python d-dm-time.py 0
python s-pbsspt.py -s d-dm-time.py -n 16:8 -j fixed-dm -w 1
```

### gate / d-dm-time

`s-pbsspt.py` 要求显式提供 `--script`、`--nodes` 和 `--job-name`。它读取入口脚本中的
`section_num`，把全部逻辑 section 均匀分配给 `--nodes` 指定的 GPU 作业。
`--workers-per-gpu` 控制同一 GPU 作业同时运行的搜索进程上限，默认值为 1。当前 gate
的 `section_num=32` 和 `--nodes 1:8` 会生成 8 个 PBS 作业，每个作业分到 4 个 section。
`--workers-per-gpu 1` 会依次运行这 4 个 section；`--workers-per-gpu 2` 会分两批运行；
`--workers-per-gpu 4` 会同时运行这 4 个 section。

每个搜索进程都使用 `--threads` 指定的 CPU 线程上限。提高 `--workers-per-gpu` 会同时增加
GPU 显存占用、主机内存占用和 CPU 线程需求。生产运行前可以用 `--dry-run` 检查生成的 PBS。

`--conda-env` 默认使用 `pytorch`，同一提交批次的所有作业使用指定环境。需要其他环境的
节点应单独提交，例如 `--conda-env normal`；环境选择由提交参数控制。

每个 PBS 作业的所有 section 输出统一写入 `<job-name>-jobxx-yy.log`，每行搜索输出带有
`[section N]` 前缀。日志末尾列出成功和失败的 section，以及作业退出状态。提交器检查
搜索进程退出码和 `Error processing` / Python traceback；入口捕获处理异常并返回 0 时，
该 section 仍计为失败，PBS 作业返回非零退出码。

| 短名 | 完整参数 | 含义 |
|---|---|---|
| `-s` | `--script` | 搜索入口脚本，必填 |
| `-n` | `--nodes` | 节点和 GPU 作业数，必填 |
| `-j` | `--job-name` | PBS 作业名前缀，必填 |
| `-r` | `--root` | 搜索代码运行目录 |
| `-e` | `--conda-env` | conda 环境 |
| `-t` | `--threads` | 每个搜索进程的 CPU 线程数 |
| `-w` | `--workers-per-gpu` | 每块 GPU 的并发搜索进程上限 |
| `-p` | `--submit-delay` | 两次 qsub 之间的秒数 |
| `-d` | `--dry-run` | 只生成 PBS 文件 |
| `-S` | `--status` | 查看 PBS 作业状态 |

```bash
# 查看 gate 的提交计划并生成 PBS 文件
python s-pbsspt.py \
  --script d-center-binary-gate.py \
  --nodes 1:8 \
  --job-name gate \
  --dry-run

# 提交 gate
python s-pbsspt.py \
  --script d-center-binary-gate.py \
  --nodes 1:8 \
  --job-name gate

# 每块 GPU 最多同时运行 4 个 section
python s-pbsspt.py \
  --script d-center-binary-gate.py \
  --nodes 1:8 \
  --job-name gate \
  --workers-per-gpu 4

# 等价的短参数写法
python s-pbsspt.py -s d-center-binary-gate.py -n 1:8 -j gate -w 4

# 同一个提交器运行任意 d-dm-time 入口
python s-pbsspt.py \
  --script d-dm-time.py \
  --nodes 1:8 \
  --job-name d-dm \
  --dry-run

# 两个节点分别提供 8 块 GPU
python s-pbsspt.py \
  --script d-center-binary-gate.py \
  --nodes 1:8,2:8 \
  --job-name gate
```

入口脚本需要接受位置参数 `<section>`，并在主配置中提供唯一的字面量
`section_num = N,`。`d-center-binary-gate.py` 和 `d-dm-time.py` 均使用该契约。

目标专用入口可以从 `d-dm-time.py` 复制并修改底部配置，命名为 `d-dm-time-<target>.py`。
这些文件保存观测路径和目标参数，保留在各自的运行目录；仓库发布通用入口。

## Benchmark 工具

Fixed-DM binary 对比：

```bash
bash t-binary-bench.sh
bash t-binary-bench.sh --summarize-only
```

`t-binary-bench.sh` 是开发用模板。运行前配置脚本底部的观测路径和 DM，并准备以下
两份与 backbone 匹配的权重：

```text
models/binary_best_model_conv_tiny_ema.pth
models/binary_best_model_conv_small_ema.pth
```

默认部署提供 ConvNeXt-Small 权重；ConvNeXt-Tiny benchmark 使用自行训练或另行取得的
兼容 checkpoint。仅比较 Small 时，可删除或注释脚本中的 Tiny 任务。
benchmark 结果依赖存储、进程并发、FITS 数据布局和 GPU，报告时应同时记录这些条件。

## 排错

| 现象 | 检查项 |
|---|---|
| 提交器无法读取 section 数量 | 确认入口脚本主配置中只包含一个字面量 `section_num = N,`。 |
| 并发运行时 GPU OOM | 降低 `--workers-per-gpu`；每个并发进程都会独立加载模型和工作数据。 |
| `Unknown detector_type` | 检查是否为 `centernet_conv_tiny` 或 `centernet_conv_small`。 |
| detector 权重加载失败 | 检查 detector 类型、backbone 和 checkpoint 是否对应。 |
| classifier 权重加载失败 | 检查 `convnext_tiny` / `convnext_small` 是否与权重一致。 |
| 找不到 FITS | 检查 `data_path`、文件名中的 `-Mxx_`、`beam_filter` / `--beam`。 |
| 所有任务都跳过 | 检查对应 `processing_log*.txt`；需要重跑时清理对应记录。 |
| CUDA OOM | 减小 `dm_range` 或 `block_size`，或降低同一 GPU 上并发进程数。 |
| 输出目录混有旧结果 | 换新的输出目录，或清理目标目录后重跑。 |
| `import cupy` 失败 | 安装与节点 CUDA toolkit 匹配的 CuPy 构建。 |
| 新环境的 `models/` 为空 | 按“模型权重”一节下载两份默认权重并校验文件名。 |
