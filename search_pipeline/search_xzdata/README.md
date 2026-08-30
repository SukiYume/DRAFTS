# 压缩 FITS fixed-DM 搜索

本目录负责“校验并解压一组 `.fits.xz` → 按 section 提交 fixed-DM 搜索 → 等待人工确认
TOA”的外层调度。搜索算法仍由上一级的 `d-dm-time-predown.py` 提供，这里不保留目标专用
算法副本。

## 文件

- `run_fixed_dm_xz.sh`：通用控制器，只读取 `XZ_SEARCH_*` 配置和 manifest。
- `fixed_dm_runner.py`：单个 PBS section 的 Python 入口。
- 目标专用 preset 和 manifest：保存真实路径、DM、模型与集群配置，只在本地保留，不纳入
  版本控制。

## Manifest

为每个目标准备一个制表符分隔的 manifest：

```text
unit_id	obs_id	beam	source_dir	prefix	expected_files	sections
example_M01	20260101	M01	/data/example/20260101	Example_tracking-M01_	280	8
```

- `source_dir` 中应直接包含 `${prefix}*.fits.xz`。
- `expected_files` 用于在解压前后做严格数量校验。
- `sections` 是该 unit 实际提交的 PBS section 数，不再由文件数隐式推断。
- 时间分辨率、频率通道数和每个 FITS 的采样数均从 FITS header 读取。

## 配置与运行

可以直接导出配置，也可以把它们写入一个不纳入版本控制的目标专用 preset：

```bash
export XZ_SEARCH_ROOT=/home/user/runs/example
export XZ_SEARCH_MANIFEST=/path/to/units_example.tsv
export XZ_SEARCH_PIPELINE_DIR=/path/to/bssearch/search_pipeline
export XZ_SEARCH_OUTPUT_BASE=/path/to/search_outputs
export XZ_SEARCH_OUTPUT_GROUP=FRB_EXAMPLE
export XZ_SEARCH_STAGING_GROUP=FRB_EXAMPLE
export XZ_SEARCH_DM=500.0
export XZ_SEARCH_PROBABILITY=0.6
export XZ_SEARCH_DOWNSAMPLE_RATES=8,16
export XZ_SEARCH_MODEL_TYPE=ConvNeXtNet
export XZ_SEARCH_MODEL_NAME=convnext_small
export XZ_SEARCH_CLASSIFIER=/path/to/classifier.pth
export XZ_SEARCH_PYTHON=/path/to/python
export XZ_SEARCH_PBS_NODE=gpuXX

bash /path/to/search_xzdata/run_fixed_dm_xz.sh list
bash /path/to/search_xzdata/run_fixed_dm_xz.sh start example_M01
bash /path/to/search_xzdata/run_fixed_dm_xz.sh status example_M01
```

输出布局为：

```text
<XZ_SEARCH_OUTPUT_BASE>/<XZ_SEARCH_OUTPUT_GROUP>/CalData/<obs_id>/<beam>/<downsample_rate>/
```

状态检查和空结果检查会递归覆盖配置中的所有下采样倍率，不假设固定的倍率目录。

直接在 PBS 提交节点运行时，`XZ_SEARCH_PBS_SUBMIT_HOST` 留空；需要经登录节点提交时，
将它设为主机名。`XZ_SEARCH_QSUB`、`XZ_SEARCH_QSTAT`、`XZ_SEARCH_SSH`、
`XZ_SEARCH_PBS_QUEUE` 和 `XZ_SEARCH_PBS_GRES` 都可以按集群覆盖。

需要在 PBS 作业中激活环境时，可设置 `XZ_SEARCH_JOB_BASHRC`、`XZ_SEARCH_CONDA_ENV`
和 `XZ_SEARCH_RUN_NOMPS`。

## 状态与清理

状态依次为 `PENDING → EXTRACTING → EXTRACTED → SEARCH_SUBMITTED → WAIT_TOA`。
任一 section 失败会标记为 `SEARCH_FAILED`；部分提交不会自动重提已经提交的 section，避免
重复搜索。

确认 `WAIT_TOA` 且所有 section 成功后，如果递归检查不到任何 JPG/NPY 候选，可执行：

```bash
bash run_fixed_dm_xz.sh discard-empty <unit_id>
```

该命令只永久删除 `XZ_SEARCH_ROOT/staging/<unit_id>` 中解压出的临时 FITS，原始 `.xz`
始终保留。存在候选、搜索进程仍在引用数据、路径越出 staging 根目录或 section 未全部成功时
都会拒绝删除。
