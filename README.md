# bssearch - DRAFTS FRB 搜索与注入实验工作区

[English documentation](README.en.md)

本目录维护当前 DRAFTS 单脉冲/FRB 搜索链路。当前主线是 **CenterNet 目标检测 + ConvNeXt binary 分类器**，覆盖训练数据生成、检测器训练、分类器训练、真实 FAST 数据搜索，以及 raw8/packed2 注入评估。

远端训练和大规模搜索通常在 `gpu13` 的 `pytorch` 环境中运行；本地主要做代码编辑、文档维护和轻量检查。

## 目录关系

```text
真实 FAST 背景 FITS
  -> generate_burst/             生成 CenterNet 训练 H5
  -> object_detection/           训练 CenterNet 检测器
  -> binary_classification/      训练二分类过滤器
  -> runcode/                    真实观测搜索入口
  -> injection_experiment/       注入信号后评估召回、误报和量化影响
  -> output/                     搜索样例和 binary 对比产物
```

## 子目录

| 目录 | 当前内容 |
|---|---|
| `generate_burst/` | 从真实 FAST 背景 raw data 注入 multifitting 风格模拟 FRB，消色散成 512 x 512 time-DM 图，写 CenterNet 训练 H5。当前批量入口是 `launch_shards_50000.sh`。 |
| `object_detection/` | 当前检测器训练目录，仅保留 CenterNet 线。脚本包括 `centernet_train.py`、`centernet_data.py`、`centernet_model.py`、`centernet_eval.py`、`centernet_infer.py`。当前本地保留 `logs_v10/` 和 `Data/`。 |
| `binary_classification/` | ConvNeXt/SPPConvNeXt 二分类过滤器训练和推理。真实搜索部署默认偏向 `convnext_small` 以保 recall。 |
| `runcode/` | 真实数据 DRAFTS 搜索入口，可独立拷到服务器运行。包含未知 DM 盲搜、fixed-DM follow-up、PBS 提交、后端 benchmark 和运行时模型定义。 |
| `injection_experiment/` | 注入评估主线：生成 raw8/packed2 FITS、调用 `search_runtime/` 搜索、匹配 truth、汇总指标并生成论文图。当前保留 v10/1024-ds2 结果。 |
| `output/` | 真实搜索样例输出和 binary 模型输出对比，不放代码或权重。 |

`bslocate/` 在上一级，保存旧版 YOLO 定位实验；当前 DRAFTS 主线使用 CenterNet 检测器。

## 训练数据生成

入口：`generate_burst/launch_shards_50000.sh`

默认配置：

- 50,000 个唯一注入信号。
- 每个信号 4 个 crop，总计 200,000 张 512 x 512 图。
- 40 个 shard，默认每 wave 并发 4 个 shard，避免 FITS I/O 过载。
- 输出 `centernet_dataset_sim50000_max3.h5`、`centernet_dataset_sim50000_max3_annotations.json`、配置 JSON、metadata JSONL 和 contact sheet。

运行前需要在 `generate_burst/rawdata/` 或 `RAW_DIR` 指定目录中准备背景 FITS。生成后的 H5/JSON 拷到 `object_detection/Data/` 供训练使用。

详细参数、分片合并、输出文件和维护规则见 `generate_burst/README.md`。

## CenterNet 检测器训练

入口：`object_detection/`

```bash
cd object_detection
DATA_PATH=./Data/v10_sim50000_only BATCH_SIZE=24 EPOCHS=50 ./train.sh "0,1,2,3,4,5,6,7" centernet-conv-tiny
```

当前保留的 v10 训练产物在 `object_detection/logs_v10/`：

```text
best_model_ema.pth
best_model.pth
last_checkpoint.pth
swa_model_ema.pth
logs_centernet.json
args.json
```

部署到真实搜索时，把选定的 `best_model_ema.pth` 复制到 `runcode/models/`，并按入口脚本需要命名，例如 `object_best_model_centernet_conv_tiny_ema.pth` 或带版本的 `object_best_model_centernet_conv_tiny_ema_v10.pth`。

## Binary 分类器训练

入口：`binary_classification/`

```bash
cd binary_classification
MODEL_NAME=convnext_small EPOCHS=50 ./train.sh "0,1,2,3"
```

训练目录的 README 记录了 Tiny/Small 的验证指标。当前真实搜索入口使用：

```text
runcode/models/binary_best_model_conv_small_ema.pth
```

Tiny 仍保留为轻量备选：

```text
runcode/models/binary_best_model_conv_tiny_ema.pth
```

## 真实数据搜索

入口：`runcode/`

| 场景 | 脚本 |
|---|---|
| 未知 DM 盲搜 | `d-center-binary-gate.py` 或命令行式 `t-blind-section.py` |
| 已知 DM / 候选 DM follow-up | `d-dm-time-predown.py` |
| PBS 批量提交 | `s-pbsspt.py` |
| object detector 后端 benchmark | `t-object-bench.py` / `t-object-matrix.sh` |
| fixed-DM binary 模型对比 | `t-binary-bench.sh` |

详细参数、输出文件名和排错表见 `runcode/README.md`。`runcode/models/` 是生产入口直接读取的权重目录；训练目录里的 checkpoint 不会自动生效。

## 注入实验

入口：`injection_experiment/`

主要脚本：

| 脚本 | 作用 |
|---|---|
| `inject_fits.py` | 把模拟 FRB 注入真实 FAST 背景，输出 raw8；可同时生成 packed2。 |
| `run_injection_campaign.py` | 批量生成、搜索、分析和聚合的总调度。支持 generate-only、search-only 和 raw8/packed2 并行搜索。 |
| `launch_v8_injection_campaign.sh` | 旧的默认启动脚本，仍可作为环境变量和参数写法参考。 |
| `analyze_search_results.py` | 把 truth 与 candidate manifest 匹配，输出召回、误报和参数分箱结果。 |
| `aggregate_campaign_results.py` | 汇总多个 batch 的 analysis 结果。 |
| `plot_publication_performance.py` | 从结果目录生成 publication figures。 |
| `search_runtime/` | 注入实验专用搜索 runtime，包含精简版 gate/core、模型定义和权重占位说明。 |

当前本地保留的结果目录：

```text
injection_experiment/results/pg13_v10_1024ds2_20260629_1351/
```

该结果使用 `object_best_model_centernet_conv_tiny_ema_v10.pth`、`binary_best_model_conv_small_ema.pth`、`class_block_size=1024`、`class_time_downsample=2`，并已生成 analysis、aggregate 和 publication figures。`publication_figures/run_summary.json` 记录了 10,000 个 truth source、20,000 个 raw8/packed2 matches、351 个 false positives、64 张图。

详细运行模式、权重放置、search-only 复跑和分析/画图命令见 `injection_experiment/README.md`。

复用旧 campaign 的搜索结果时参考：

```bash
python run_injection_campaign.py \
  --work-root /path/to/drafts_runs/injection_experiment/runs \
  --run-label v9_injection_10000_eventdedup_20260628_0200 \
  --batches 20 \
  --count-per-batch 500 \
  --search-only \
  --overwrite-search \
  --runtime-dir /path/to/drafts_runs/injection_experiment/search_runtime \
  --gpu-num 8 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --detector-type centernet_conv_tiny \
  --detector-ckpt models/object_best_model_centernet_conv_tiny_ema_v10.pth \
  --classifier-ckpt models/binary_best_model_conv_small_ema.pth \
  --classifier-model-name convnext_small \
  --class-block-size 1024 \
  --class-time-downsample 2 \
  --source-dm-tolerance 60 \
  --source-time-tolerance-ms 30
```

## 快速选择

| 要做什么 | 进入 |
|---|---|
| 生成新的检测器训练 H5 | `generate_burst/` |
| 训练或复查当前 CenterNet 检测器 | `object_detection/` |
| 训练 binary 过滤器 | `binary_classification/` |
| 在真实 FAST 数据上搜索 | `runcode/` |
| 复查搜索样例输出或 binary 对比图 | `output/` |
| 跑 raw8/packed2 注入评估 | `injection_experiment/` |

## 输出和权重约定

- `object_detection/Data/` 和 `binary_classification/Data/` 是训练数据位置。
- `object_detection/logs_v10/`、`binary_classification/logs/` 是训练产物位置。
- `runcode/models/` 是真实搜索部署权重位置。
- `injection_experiment/search_runtime/models/` 不随仓库带权重，运行前按 `PUT_WEIGHTS_HERE.txt` 放入对应 `.pth`。
- `injection_experiment/results/` 保存可分享的评估结果和论文图。
- `output/` 只放真实搜索样例和对比产物，不放训练日志或 checkpoint。

## Git 追踪边界

仓库追踪代码、脚本、README、论文源文件、轻量配置和小型评估摘要。以下内容保留在本地或服务器，不进入 Git：

- FITS 原始观测、H5/NumPy 训练数据；
- `.pth`、`.pt`、`.ckpt` 等模型权重；
- 训练日志、批处理状态、搜索输出和可重新生成的结果表；
- 本地抓取的文献全文与 LaTeX 编译产物。

公开训练数据和模型分别存放在
[Hugging Face 数据集](https://huggingface.co/datasets/TorchLight/DRAFTS) 与
[Hugging Face 模型库](https://huggingface.co/TorchLight/DRAFTS)。各工作流 README
说明了本地数据和权重应放置的位置；`.gitignore` 只阻止 Git 追踪，不会删除这些本地文件。

项目采用 MIT 许可证，见 [LICENSE](LICENSE)。
