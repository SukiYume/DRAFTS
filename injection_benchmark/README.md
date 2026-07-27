# DRAFTS 注入基准

本目录用于评估 DRAFTS 搜索链路：把模拟 FRB 注入真实 FAST 背景，生成 raw8/packed2 FITS，调用搜索 runtime，匹配 truth manifest，并汇总召回、误报和量化影响。

## 流程

```text
generate_injections.py
  -> simdata/<run_label>_bXX_{raw8,packed2} + truth_archive/<run_label>_bXX_{raw8,packed2}
  -> run_campaign.py / launch_search.py 调用 search_runtime
  -> evaluate_results.py 匹配 truth 和 candidates
  -> aggregate_results.py 汇总 batch 指标
```

`run_campaign.py` 是主入口。它支持完整流程、只生成注入数据、只复用已有注入数据重新搜索三种模式。

## 文件职责

| 路径 | 用途 |
| --- | --- |
| `run_campaign.py` | 注入评估主控脚本，负责分 batch 生成、搜索、分析和汇总。 |
| `generate_injections.py` | 生成注入后的 raw8/packed2 FITS 和 `truth_manifest.jsonl`。 |
| `injection_model.py` | 注入用动态谱基本函数：高斯频率包络、散射高斯时间 profile 和色散延迟常数。参数采样在 `generate_injections.py` 中完成。 |
| `launch_search.py` | 单 batch 的 DRAFTS 多 section 搜索封装，调用 `search_runtime/`。 |
| `evaluate_results.py` | 读取 truth 与候选表，做 source/event 匹配和误报统计。 |
| `aggregate_results.py` | 汇总多个 batch 的分析结果。 |
| `matching.py` | DRAFTS 与 PRESTO 共用的最大基数、最小代价一对一匹配。 |
| `search_runtime/` | 注入基准专用搜索运行时。运行前需要把检测器和 binary 分类器权重放入 `search_runtime/models/`。 |
| `presto_runtime/` | PRESTO blind-search 基线、快速汇总和阈值重画脚本。 |
| `simdata/` | 运行时生成的大体积 raw8/packed2 注入 FITS，供 DL 和 PRESTO 共用。 |
| `truth_archive/` | 每个 batch/量化版本的 `truth_manifest.jsonl` 和 `run_config.json`，供 DL 和 PRESTO 共用。 |
| `runs/` | DL 搜索、分析、汇总和日志目录；大体积注入 FITS 与其分开保存。 |

Git 只跟踪源码、README 和权重清单。`simdata/`、`truth_archive/`、`runs/`、结果目录，
以及本地论文/笔记目录均由 `.gitignore` 排除，不属于可提交的实验代码。

`search_runtime/` 与 `search_pipeline/` 共享搜索模型实现，但不是整目录逐字镜像：
`binary_model.py`、`centernet_model.py` 和 `centernet_eval.py` 应保持逐文件一致；
`d-center-binary-core.py`、`d-center-binary-gate.py` 与 `t-blind-section.py` 则保留
注入评估需要的候选/proposal manifest、重叠 DM window、classifier time
downsample 和物理 TOA-DM 去重接口。同步生产搜索改动时应逐项移植并重新执行
campaign dry-run，不能直接用 `search_pipeline/` 覆盖整个目录。

## 注入信号模型

注入基准使用与训练 H5 生成相同的物理动态谱形式，但参数范围更窄，目标是可控评估搜索召回、误报和 raw8/packed2 量化影响。

单个注入源的构建方式：

| 项 | 构建方式 |
| --- | --- |
| 参考时间 | `highest_freq_toa_global_raw_sample` 是最高频通道到达时刻，作为 truth manifest 的统一时间轴。 |
| 色散延迟 | `t(f) = t_high + K_DM * DM * (f^-2 - f_ref^-2) / time_reso`，其中 `f_ref=max(freq)`，`K_DM=4148.808 s MHz^2 pc^-1 cm^3`。 |
| 频率包络 | 高斯包络，`sigma_freq_mhz = bandwidth_mhz_fwhm / 2.355`。中心频率在观测带均匀抽样，带外尾部由有限观测带截断。 |
| 时间轮廓 | `sigma_time_ms = width_ms_fwhm / 2.355`。无散射或散射很弱时为纯高斯；其余情况使用峰值归一的 ex-Gaussian 散射 profile。 |
| 散射频率标度 | `tau(f) = tau_1ghz * (f/1000 MHz)^-4`。truth 中同时保存 `scattering_ms_at_1ghz` 和最高频处的 `scattering_ms_at_highest_freq`。 |
| 单通道幅度 | `per_channel_peak_snr = snr / sqrt(effective_channels_fwhm)`，其中 `effective_channels_fwhm` 是频率包络大于 0.5 的通道数。 |
| raw8 注入 | `per_channel_peak_snr * frequency_envelope * pulse_profile * channel_std` 加到 Stokes-I 背景上，四舍五入并 clip 到 `[0,255]`。 |
| packed2 注入 | 先在 raw8/Stokes-I 空间累加同一模型，再按频率通道量化成 2-bit packed 输出。 |

## 参数范围和分布

`generate_injections.py` 对 5 个核心维度使用 Latin-hypercube stratified unit cube，再映射到物理参数；这样每个 campaign 内参数覆盖更均匀，适合做 completeness 和分箱图。

| 参数 | 范围 | 分布 |
| --- | --- | --- |
| `dm_pc_cm3` | `[100, 2000]` | Latin-hypercube 分层均匀分布。 |
| `snr` | `[5, 100]` | Latin-hypercube 分层对数均匀分布。 |
| `width_ms_fwhm` | `[1, 20] ms` | Latin-hypercube 分层对数均匀分布。 |
| `bandwidth_mhz_fwhm` | `[50, 500] MHz` | Latin-hypercube 分层对数均匀分布。 |
| `scattering_ms_at_1ghz` | `0` 或 `[0.03, 10] ms` | 15% 精确为 `0`；其余 Latin-hypercube 分层对数均匀分布。 |
| `center_freq_mhz` | 观测频带 `[freq_min, freq_max]` | 在生成 truth 行时均匀分布。 |
| `highest_freq_toa_global_raw_sample` | `inject_file_first..inject_file_last` 覆盖的样本窗口 | 合法窗口内均匀抽样；每个注入保留至少 `0.24 s` 间隔，并使用 `0.18 s` 或模型支持宽度作为边界 guard。 |

TOA 放置最多尝试 1024 次。若给定窗口、注入数量、边界 guard 和最小间距无法同时满足，
`generate_injections.py` 会抛出 `ValueError` 并终止该批生成；不会返回违反间距约束的“尽力而为”
样本。此时应扩大注入文件范围、减少底层 `--count`（campaign 入口为
`--count-per-batch`），或显式调整间距参数。

每个 raw8/packed2 输出目录都会写 `run_config.json` 和 `truth_manifest.jsonl`。`run_config.json` 记录参数范围和分布，`truth_manifest.jsonl` 记录每个注入源的实际 DM、S/N、宽度、带宽、中心频率、散射、TOA 和 per-channel 幅度。

## 权重和 runtime

搜索脚本不会自动读取训练日志目录。运行前把需要评估的 `.pth` 放到 `search_runtime/models/`，并在命令行显式指定：

- `--detector-type centernet_conv_tiny`
- `--detector-ckpt models/<detector-weight>.pth`
- `--classifier-ckpt models/binary_best_model_conv_small_ema.pth`
- `--classifier-model-name convnext_small`

本文给出的 v10/1024-ds2 参考基线使用
`object_best_model_centernet_conv_tiny_ema_v10.pth` 和
`binary_best_model_conv_small_ema.pth`。

两份默认权重可从
[DRAFTS model repository](https://huggingface.co/TorchLight/DRAFTS) 下载：
文件名和 SHA-256 同时记录在
[`search_runtime/models/PUT_WEIGHTS_HERE.txt`](search_runtime/models/PUT_WEIGHTS_HERE.txt)。

```bash
mkdir -p search_runtime/models
curl -L \
  -o search_runtime/models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  https://huggingface.co/TorchLight/DRAFTS/resolve/main/object_best_model_centernet_conv_tiny_ema_v10.pth
curl -L \
  -o search_runtime/models/binary_best_model_conv_small_ema.pth \
  https://huggingface.co/TorchLight/DRAFTS/resolve/main/binary_best_model_conv_small_ema.pth
```

## 只生成注入数据

复用同一批注入数据比较不同模型时，先生成一次注入数据：

```bash
cd DRAFTS/injection_benchmark
python run_campaign.py \
  --background-dir /path/to/rawdata \
  --work-root /path/to/injection_benchmark/runs \
  --sim-root /path/to/injection_benchmark/simdata \
  --truth-root /path/to/injection_benchmark/truth_archive \
  --run-label my_injection_10000 \
  --batches 20 \
  --count-per-batch 500 \
  --gpu-num 8 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --generate-only \
  --keep-injected-fits
```

如果只需要单独检查注入 FITS 生成，可直接用底层脚本：

```bash
python generate_injections.py \
  --background-dir /path/to/rawdata \
  --output-root ./simdata \
  --run-label demo001 \
  --count 64 \
  --dry-run
```

去掉 `--dry-run` 后会写 `simdata/demo001_raw8`、`simdata/demo001_packed2`、`run_config.json` 和 `truth_manifest.jsonl`。

## 复用注入数据重新搜索

下面的命令复现 v10/1024-ds2 参考搜索配置。`run-label` 只是本次 campaign 的标识，
可以替换成不会与已有结果冲突的名称。

```bash
cd DRAFTS/injection_benchmark
conda activate pytorch
python run_campaign.py \
  --work-root /path/to/injection_runs/runs \
  --sim-root /path/to/injection_runs/simdata \
  --truth-root /path/to/injection_runs/truth_archive \
  --run-label v10_det03_injection_10000 \
  --batches 20 \
  --count-per-batch 500 \
  --search-only \
  --overwrite-search \
  --runtime-dir ./search_runtime \
  --gpu-num 8 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --detector-type centernet_conv_tiny \
  --detector-ckpt models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  --classifier-ckpt models/binary_best_model_conv_small_ema.pth \
  --classifier-model-name convnext_small \
  --class-block-size 1024 \
  --class-time-downsample 2 \
  --classifier-batch-size 64 \
  --source-dm-tolerance 60 \
  --source-time-tolerance-ms 30 \
  --poll-seconds 30
```

## 分析和汇总

`run_campaign.py` 正常完成后会自动分析和汇总。需要手动复查时，可分别运行：

```bash
python evaluate_results.py \
  --truth /path/to/injection_benchmark/truth_archive/<run_label>_bXX_raw8/truth_manifest.jsonl \
  --candidates /path/to/candidates.csv \
  --output-dir /path/to/analysis \
  --source-dm-tolerance 60 \
  --source-time-tolerance-ms 30

python aggregate_results.py \
  --analysis-root /path/to/runs/<run_label>/analysis \
  --output-dir /path/to/runs/<run_label>/aggregate
```

DL 与 PRESTO 的候选先按时间/DM 容差形成事件，但事件内任意两成员都必须满足整体直径
限制，避免单链式 `A≈B≈C` 把相距过远的 A/C 桥接成一个事件。truth-event 分配随后在
所有合法边上做“最大匹配数优先、总归一化距离最小”的一对一全局匹配，不使用依赖输入
顺序的逐 truth 贪心。这样不会因为某个 truth 抢先占用共享近邻而丢掉本来可达到的召回。

## 运行注意事项

- 默认根目录环境变量为 `INJECTION_BENCHMARK_ROOT`；为兼容已有远端任务，
  `INJECTION_EXPERIMENT_ROOT` 仍作为后备名称。`INJECTION_WORK_ROOT`、
  `INJECTION_SIM_ROOT` 和 `INJECTION_TRUTH_ROOT` 的优先级更高。
- 注入 FITS 体积很大。只有需要复用同一批注入数据做模型对比时才使用 `--keep-injected-fits`。
- `--search-only` 依赖 `simdata/` 和 `truth_archive/` 中已有的注入数据；换 `--run-label` 前先确认对应 campaign 已生成。
- `--overwrite-search` 只用于替换搜索、分析和汇总产物；它不负责重新生成注入 truth。
