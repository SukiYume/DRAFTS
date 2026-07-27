# DRAFTS 真实数据搜索入口

本目录是一套可以拷到服务器独立运行的 FRB 单脉冲搜索 runtime。运行时只需要 Python/CUDA 环境、FAST FITS 数据、模型权重和输出目录。

当前默认搜索链路是 **CenterNet ConvNeXt-Tiny detector v10 + ConvNeXt-Small binary classifier**。

## 入口选择

| 场景 | 推荐入口 | 说明 |
|---|---|---|
| 不想改源码，先测试一段盲搜 | `t-blind-section.py` | 命令行参数完整，适合单 section / 多 section 调试。 |
| 固定配置生产盲搜 | `d-center-binary-gate.py` | 脚本底部写路径、模型和 GPU section；适合 PBS 渲染。 |
| 已知 DM / 候选 DM follow-up | `d-dm-time-predown.py` | 固定 DM，多时间下采样倍率，binary 分类。 |
| PBS 批量提交 | `s-pbsspt.py` | 生成并提交多 section 作业，也可渲染目标专用 gate 脚本。 |
| detector 后端 benchmark | `t-object-bench.py` / `t-object-matrix.sh` | 只跑消色散、检测图和 detector，不跑 binary。 |
| fixed-DM binary 对比 | `t-binary-bench.sh` | 比较不同 binary 分类器的真实输出。 |

推荐先用 `t-blind-section.py --dry-run` 确认数据分片和输出路径，再跑实际搜索；生产批量再改 `d-center-binary-gate.py`，并用 `s-pbsspt.py` 提交 PBS。

## 文件职责

| 文件或目录 | 作用 |
|---|---|
| `d-center-binary-gate.py` | 两阶段盲搜调度入口：组织 FITS、section 切分、加载模型、断点日志、调用 core。 |
| `d-center-binary-core.py` | 两阶段盲搜核心：CuPy 消色散、DM-time 图、CenterNet 检测、candidate cutout、binary 分类和保存。 |
| `d-dm-time-predown.py` | fixed-DM follow-up：单 DM 消色散、多尺度图片生成、binary 分类。 |
| `t-blind-section.py` | 命令行式盲搜 section wrapper。 |
| `t-blind-batch.sh` | 多 GPU 盲搜批量测试 wrapper。 |
| `t-object-bench.py` / `t-object-matrix.sh` | detector-only benchmark。 |
| `t-binary-bench.sh` | fixed-DM binary 模型输出对比。 |
| `s-pbsspt.py` | PBS 脚本生成、提交和状态查询。 |
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
`cupy-cuda12x`。`t-object-bench.py --backend numba` 是可选 benchmark，只有使用该后端时
才需要额外安装 `numba`。PyTorch 与 torchvision 同样应选择和目标 CUDA 环境匹配的
wheel。

2026-07-27 在 `pg13` 完成了以下生产环境实测：

| Python | PyTorch / CUDA build | torchvision | CuPy / runtime | GPU / 驱动 | GPU smoke test |
|---|---|---|---|---|---|
| 3.11.15 | 2.5.1+cu121 / 12.1 | 0.20.1+cu121 | 14.0.1 / 12.9 | NVIDIA L40 / 535.129.03 | PyTorch、CuPy 均识别 8 卡，基本 CUDA 运算通过 |

这是已验证基线，不是最低版本约束；同一环境的 NumPy/SciPy/Astropy/h5py 分别为
2.4.4/1.16.3/7.2.0/3.16.0。

检查环境：

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

从 [DRAFTS model repository](https://huggingface.co/TorchLight/DRAFTS) 下载到
`models/`：

```bash
mkdir -p models
curl -L \
  -o models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  https://huggingface.co/TorchLight/DRAFTS/resolve/main/object_best_model_centernet_conv_tiny_ema_v10.pth
curl -L \
  -o models/binary_best_model_conv_small_ema.pth \
  https://huggingface.co/TorchLight/DRAFTS/resolve/main/binary_best_model_conv_small_ema.pth
```

当前默认：

```python
DETECTOR_TYPE = "centernet_conv_tiny"
DETECTOR_CKPT = "./models/object_best_model_centernet_conv_tiny_ema_v10.pth"
CLASSIFIER_MODEL_NAME = "convnext_small"
CLASSIFIER_CKPT = "./models/binary_best_model_conv_small_ema.pth"
```

更换权重时需要同时保持 detector type、classifier backbone 和 checkpoint 架构一致。

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

如果 `data_path` 直接包含 FITS，脚本按文件名中的 `-Mxx_` 聚合 beam。如果 `data_path` 指向 source 根目录，脚本递归扫描日期子目录。`beam_filter='M01'` 或 `--beam M01` 只跑 M01，`all` 跑全部 beam。

脚本会跳过文件名中包含 `_N_`、`_W_`、`_F_` 的 FITS。

### 望远镜、频段与固定通道选择

当前 runtime 面向 FAST L 波段 PSRFITS。频率轴本身由 `OBSFREQ`、`OBSBW` 和
`NCHAN` 读取，检测输入固定下采样为 512 个通道；但消色散检测还沿用了针对
4096-channel FAST 数据设定的固定有效通道段 `[10, 650)` 与 `[820, 4050)`，并按
`NCHAN/4096` 缩放索引。对标称 1000–1500 MHz/4096-channel 数据，这约对应
1001.2–1079.3 MHz 与 1100.1–1494.3 MHz。二分类 cutout 仍使用 full-frequency
数据。

因此，其他望远镜、其他接收机频段、倒置频率轴，或物理坏频段不同的数据都不能只改
FITS 路径后直接视为已验证。应先按目标观测修改
`d-center-binary-core.py::_build_dedispersion_cache()` 的 `index_array`，再用已知
脉冲和注入实验验证召回、DM/TOA 误差；`NCHAN` 还必须能整除到 512 通道。

### 损坏 FITS 的连续搜索策略

`d-center-binary-gate.py` 的盲搜策略是按文件容错：某个 FITS 读取失败时，把异常写到
与 completed log 同名前缀的 `*_bad_fits.log`，只用相同 shape 的随机噪声替换该文件，
然后继续读取和搜索后续 FITS，不会因替换动作提前结束任务。只有
`process_fits_list()` 遍历完成后，gate 才把整个 section 写入 completed log。

随机替换会使涉及该文件的窗口没有真实观测内容，但可以避免一次坏文件阻断后续文件。
正式结果必须同时归档并检查 `*_bad_fits.log`；修复原 FITS 后若需要补回这段观测，应
清理该 section 的完成记录及对应输出后重跑。

## 未知 DM 盲搜

### 命令行 section 测试

```bash
CUDA_VISIBLE_DEVICES=0 python t-blind-section.py \
  --section 0 \
  --gpu-num 1 \
  --data-path /path/to/fast_observation/source/date \
  --output-root /path/to/drafts_runs/blind \
  --beam M01 \
  --detector-type centernet_conv_tiny \
  --detector-ckpt models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  --classifier-ckpt models/binary_best_model_conv_small_ema.pth \
  --dm-range 4096 \
  --dm-scale 1 \
  --dm-offset 0 \
  --dm-threshold 10 \
  --block-size 4096 \
  --dm-span 1024 \
  --det-prob 0.40
```

常用参数：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--gpu-num` | 8 | section 总数。单卡调试设为 1。 |
| `--dm-range` | 4096 | DM index 数量。 |
| `--dm-scale` | 1.0 | 每个 DM index 对应的 pc cm^-3 步长。 |
| `--dm-offset` | 0.0 | DM 起点。 |
| `--dm-threshold` | 50.0 | 低于该 DM 的候选丢弃；仓库内 gate 模板显式设为 10。 |
| `--block-size` | 8192 | 每个时间块的降采样后样本数；仓库内 gate 模板显式设为 4096。 |
| `--dm-span` | 1024 | 每张检测图覆盖的 DM 点数。 |
| `--det-prob` | 0.45 | CenterNet 候选阈值；仓库内 gate 模板显式设为 0.40。 |
| `--time-factor` | 8.0 | FITS 时间降采样率。 |

多 GPU 可以分别提交 `--section 0..gpu_num-1`，或使用 `t-blind-batch.sh`。`--gpu-num` 在这里表示总 section 数，不一定等于物理 GPU 数。

开发节点上直接跑多 section 示例：

```bash
ROOT=/path/to/drafts_runs/data_searching \
OUTPUT_ROOT=/path/to/drafts_runs/blind \
BEAM=all GPU_NUM=8 \
DM_THRESHOLD=10 BLOCK_SIZE=4096 DM_SPAN=1024 DET_PROB=0.40 \
  bash t-blind-batch.sh /path/to/fast_observation/
```

在由 PBS 管理的 gate 节点上，不要直接启动 `t-blind-batch.sh`；应使用
`s-pbsspt.py` 生成并提交作业。

### 固定配置 gate

`d-center-binary-gate.py` 底部提供一个使用通用占位路径的 CRAFTS 配置模板：

```python
process_config = ProcessConfig(
    dm_range=4096,
    dm_scale=1,
    dm_offset=0,
    dm_threshold=10,
    block_size=4096,
    dm_span=1024,
    det_prob=0.40,
    section_num=32,
    time_factor=8,
)

data_path = None
data_paths = [
    "/path/to/CRAFTS/",
]
task_manifest = "/path/to/observations/CRAFTS/task_manifest_zd202x_1_1_2bit.json"
save_base = "/path/to/observations/CRAFTS/"
beam_filter = "all"
```

如果 `task_manifest` 指向一个存在的文件，gate 会直接加载预构建任务；否则才扫描
`data_paths`。大规模目录建议先构建 manifest，减少每个 section 重复扫描文件系统：

```bash
python c-manifest-build.py \
  --data-path /path/to/CRAFTS/ \
  --beam-filter all \
  --output /path/to/observations/CRAFTS/task_manifest_zd202x_1_1_2bit.json
```

多个数据根可以重复传入 `--data-path`。生成后把 gate 中的 `task_manifest` 指向该
JSON；更换数据集时应重新生成，避免继续使用旧任务清单。

直接运行某个 section：

```bash
CUDA_VISIBLE_DEVICES=0 python d-center-binary-gate.py 0
```

`processing_log_zd202x_1_1_2bit.txt` 记录已完成 identifier。重跑全集前删除对应日志：

```bash
rm -f processing_log_zd202x_1_1_2bit.txt
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

合并所有 beam 的 manifest：

```bash
python c-manifest-summary.py \
  --root /path/to/observations/CRAFTS \
  --csv c-candidates-zd202x.csv \
  --json c-candidates-zd202x.json
```

后续用 `data_processing` 切原始数据时，先用 `signal_mjd` 减去整次观测起始 MJD：

```python
cut_toa_sec = (signal_mjd - obs_start_mjd) * 86400.0
```

正式运行应写到仓库之外的独立运行目录，不要把搜索输出混入代码目录。

## Fixed-DM Follow-up

`d-dm-time-predown.py` 适合：

- 已知重复源 DM，回看一批观测。
- 盲搜发现候选后，在候选 DM 附近复查。
- 比较不同 binary 分类器和时间下采样倍率。

脚本底部需要修改：

```python
config = ProcessConfig(
    DM=273.5,
    prob=0.5,
    block_size=512,
    section_num=15,
    down_sampling_rate_list=np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
)

classifier_ckpt = "./models/binary_best_model_conv_small_ema.pth"
data_path = "/path/to/fast_observation/source/date/"
save_base = "/path/to/drafts_search_output/"
beam_filter = "M01"
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python d-dm-time-predown.py 0
```

Fixed-DM 输出布局：

```text
<save_base>/<source>/CalData/<date>/<beam>/<down_sampling_rate>/
```

典型文件：

```text
FRB20240114A_tracking-M01_0086-0014-550.453248.jpg
FRB20240114A_tracking-M01_0086-0014.npy
```

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

`s-pbsspt.py` 将 `node_config` 展开成连续 section，每块 GPU（每次 qsub）生成一个 PBS 脚本并提交。关键配置：

```python
root_path = "/path/to/runtime/"
script_name = "d-center-binary-gate.py"
node_config = {1: 8}        # {节点号: GPU数}
job_name = "drafts"
workers_per_gpu = 4         # 每块 GPU 上并发跑几个 section（1~4）
```

`workers_per_gpu > 1` 时，同一次 qsub 只申请 1 块 GPU，脚本内部用后台进程并发跑多个 section
（各自输出到独立日志文件），共享这一块 GPU。`sum(node_config.values()) * workers_per_gpu`
必须等于入口脚本中的 `section_num`。仓库内示例配置是 `8 * 4 = 32`，对应
`d-center-binary-gate.py` 的 `section_num=32`。

常用命令：

```bash
python s-pbsspt.py --dry-run
python s-pbsspt.py
python s-pbsspt.py --status
qstat -u $USER
```

## Benchmark 工具

Detector-only：

```bash
CUDA_VISIBLE_DEVICES=0 python t-object-bench.py \
  --backend cupy \
  --detector-type centernet_conv_tiny \
  --detector-ckpt models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  --data-path /path/to/fast_observation/source/date \
  --output-root /path/to/drafts_runs/object_backend \
  --dm-range 4096 \
  --dm-scale 1 \
  --dm-offset 0 \
  --dm-threshold 10 \
  --block-size 4096 \
  --detect-dm-span 1024 \
  --det-prob 0.40
```

多组合矩阵：

```bash
ROOT=/path/to/drafts_runs/data_searching \
DATA_PATH=/path/to/fast_observation/source/date \
OUT_ROOT=/path/to/drafts_runs/object_backend \
GPU_NUM=4 BACKENDS="cupy" MODELS="centernet_conv_tiny" SAVE_PLOT=0 \
  bash t-object-matrix.sh
```

Fixed-DM binary 对比：

```bash
bash t-binary-bench.sh
bash t-binary-bench.sh --summarize-only
```

`t-binary-bench.sh` 是开发用模板，不是开箱即用的生产入口。实际运行前需要编辑脚本
底部的观测路径和 DM，并准备以下两份与 backbone 匹配的权重：

```text
models/binary_best_model_conv_tiny_ema.pth
models/binary_best_model_conv_small_ema.pth
```

公开的默认部署权重只包含 ConvNeXt-Small；ConvNeXt-Tiny benchmark 需要使用自行训练
或另行取得的兼容 checkpoint。如果只比较 Small，应删除或注释脚本中的 Tiny 任务。
benchmark 结果依赖存储、进程并发、FITS 数据布局和 GPU，报告时应同时记录这些条件。

## 排错

| 现象 | 检查项 |
|---|---|
| `node_config` 数量不匹配 | 调整 `node_config` / `workers_per_gpu`，或入口脚本的 `section_num`。 |
| `Unknown detector_type` | 检查是否为 `centernet_conv_tiny` 或 `centernet_conv_small`。 |
| detector 权重加载失败 | 检查 detector 类型、backbone 和 checkpoint 是否对应。 |
| classifier 权重加载失败 | 检查 `convnext_tiny` / `convnext_small` 是否与权重一致。 |
| 找不到 FITS | 检查 `data_path`、文件名中的 `-Mxx_`、`beam_filter` / `--beam`。 |
| 所有任务都跳过 | 删除对应 `processing_log*.txt`。 |
| CUDA OOM | 减小 `dm_range` 或 `block_size`，或降低同一 GPU 上并发进程数。 |
| 输出目录混有旧结果 | 换新的输出目录，或清理目标目录后重跑。 |
| `import cupy` 失败 | 安装与节点 CUDA toolkit 匹配的 CuPy 构建。 |
| 新服务器 `models/` 为空 | 按“模型权重”一节下载两份默认权重并校验文件名。 |
