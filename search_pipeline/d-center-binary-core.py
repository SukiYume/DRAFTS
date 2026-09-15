# fmt: off
"""两阶段搜索核心（原始时间/频率分辨率 GPU 消色散）。

数据流（全 GPU 流水线）：
    CPU                          ：FITS 读盘并合并前两个偏振；不做时间/频率下采样
    CPU → GPU                    ：uint16 原始时间 × 原始频率数据
    GPU                          ：细频率通道消色散，再融合完成时间与 NCHAN→512 频率平均
    GPU 常驻                    ：下采样后的 dm_time，零拷贝桥接为 torch tensor
    GPU 完成                    ：reshape / normalize / 中心检测 / binary cutout / 分类
    CPU 仅用于                  ：FITS 读盘、候选元数据与可视化保存（仅检测命中时）

检测模型与中心点解码函数均由 caller（gate.py）传入。``decode_fn`` 接受
``(raw_pred, conf_thr)``，并为每张图返回 ``(scores_np, centers_xy_np)``。
CenterNet 的亚像素中心坐标会一直保留到 DM 和 TOA 换算，避免过早取整损失精度。

binary 分类器使用 :mod:`binary_model.build_binary_model`（与训练侧 ``binary_classification/`` 同步）。

性能设计：
  1. FITS 缓存保存 uint16 偏振和，主机内存与 H2D 流量仅为 float32 的一半
  2. tiled transpose kernel 在 GPU 上直接生成 (freq, time) 布局
  3. 相邻 32 个 DM 同块计算，利用其相近时移复用只读缓存数据
  4. 原位时间前缀和让每个输出只需两次读取即可完成时间平均
  5. shift_base 与原始数据缓冲常驻显存，候选切片直接读取显存
  6. binary cutout 用单个 kernel 融合细通道消色散、时间平均和频率平均
  7. CuPy ↔ PyTorch 零拷贝桥接；归一化使用 ``torch.kthvalue``
  8. 用已上传 GPU 的当前搜索块生成轻量 512-channel RFI 掩码；每个坏 coarse
     channel 对应的一组细通道不参与 DM-time 求和
  9. 前缀和按 prefix_channel_chunk 个细通道分块，块间复用缓冲并延续 float32 累加值

默认前缀和缓冲覆盖 512 个细通道。原始数据和三幅 DM-time 输出保留完整精度；
显存需求由输入窗口、DM trial 数、前缀和分块大小与模型推理 batch 共同决定。
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from astropy.io import fits

cv2.setNumThreads(4)
plt.style.use("default")
_BURST_CMAP = sns.color_palette("mako", as_cmap=True)

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reverse_freq = False


# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------


@dataclass
class ProcessConfig:
    """搜索配置。

    DM 参数全部使用物理单位 pc cm^-3，搜索区间为
    ``[dm_min, dm_min + dm_range)``。
    每张 512x512 检测图固定包含 1024 个 DM trial（每个像素平均 2 个
    trial），因此物理 DM 宽度由 ``1024 * dm_step`` 自动派生。
    """

    dm_min: float              = 0.0
    dm_range: float            = 4096.0
    dm_step: float             = 1.0
    dm_threshold: float        = 50.0
    block_size: int            = 8192
    det_prob: float            = 0.45
    section_num: int           = 5
    time_factor: float         = 8.0
    class_block_size: int      = 512
    class_threshold: float     = 0.5
    classifier_batch_size: int = 64
    # 每轮前缀和覆盖的细通道数；缓冲在频率块间复用。
    prefix_channel_chunk: int  = 512

    # 同时控制 FITS 读取线程数和后续文件的预读深度；设为 1 时串行读取。
    fits_prefetch_workers: int   = 2

    rfi_filter: bool             = True
    rfi_level_ratio: float       = 0.05
    rfi_cv_ratio: float          = 3.0
    rfi_zero_dm_filter: bool     = True
    rfi_zero_dm_sigma: float     = 8.0
    rfi_zero_dm_occupancy: float = 0.03

    save_npy: bool               = True
    save_plot: bool              = True
    save_manifest: bool          = True
    manifest_name: str           = "candidate_manifest.jsonl"
    verbose: bool                = False


# Detector 输入的 DM 轴为 512 像素，每个像素平均 2 个 DM trial。
DM_TRIALS_PER_CHUNK = 1024
# 搜索时间分辨率为 49.152 us * time_factor；RFI 统计固定为 393.216 us。
SEARCH_TIME_UNIT_US    = 49.152
RFI_TIME_RESOLUTION_US = 393.216


@dataclass
class DedispersionCache:
    """预计算 + 常驻显存的消色散缓存（在多个 block_file 间复用）。

    输入 data 比输出 dm_time 宽：dm_time_width 是时间后降采样后的 block 输出长度，
    data_width 是原始时间采样长度，包含色散延迟所需的读取提前量。
    """

    split_pos: int                       # 低频和高频输出的细通道分界
    shift_base_gpu: cp.ndarray           # (dm_trials, freq_reso) int32
    shift_unit_raw_cpu: np.ndarray       # (freq_reso,) float32，精确 cutout 复用键
    shift_unit_raw_gpu: cp.ndarray       # (freq_reso,) float32，每 DM 的原始采样时移
    dm_time_gpu: cp.ndarray              # (3, dm_trials, dm_time_width) float32
    raw_data_gpu: cp.ndarray             # (data_width, freq_reso) uint16
    data_cumsum_gpu: cp.ndarray          # (min(prefix_channel_chunk, freq_reso), data_width) float32
    dm_time_width: int                   # block 输出的降采样时间长度
    data_width: int                      # 含色散读取余量的原始时间长度


# ---------------------------------------------------------------------------
# 2. CuPy RawKernel：转置/归一化、消色散、binary cutout
# ---------------------------------------------------------------------------
#
# kernel 布局（参数调整必须重新基准）：
#   1. data 转置成 (freq, time) → 同 warp 内 shift 连续 → 完美合并访问
#   2. block 维度 (BLOCK_TIME, BLOCK_DM) = (32, 32)，一个 warp 对应一个 DM
#   3. 同块相邻 DM 的 shift 只差少量样本，前缀数据可由 L1/L2 大量复用
#   4. shift_base 保存在全局显存
#   5. split_pos 切两段循环，消除 hot loop 内分支
# ---------------------------------------------------------------------------

# Kernel sources are kept ASCII-only because some NVRTC builds reject non-ASCII comments.
_PREPARE_KERNEL = cp.RawKernel(
    r"""
#define TILE 32
extern "C" __global__
void prepare_data(const unsigned short* raw_data,
                  const float* coarse_mean,
                  const unsigned char* rfi_bad_coarse,
                  const unsigned char* rfi_bad_time,
                  float* data_out,
                  int time_size, int freq_size, int freq_start, int freq_count,
                  int rfi_down_time_rate, int down_freq_rate) {
    __shared__ float tile[TILE][TILE];

    int f_in = freq_start + blockIdx.y * TILE + threadIdx.x;
    int time_base = blockIdx.x * TILE;

    for (int j = threadIdx.y; j < TILE; j += blockDim.y) {
        int t = time_base + j;
        if (f_in < freq_start + freq_count && t < time_size) {
            float denom = coarse_mean[f_in / down_freq_rate] + 1.0e-8f;
            tile[j][threadIdx.x] = rfi_bad_coarse[f_in / down_freq_rate]
                ? 0.0f : (rfi_bad_time[t / rfi_down_time_rate]
                    ? 1.0f : (float)raw_data[
                        (size_t)t * (size_t)freq_size + (size_t)f_in
                    ] / denom);
        } else {
            tile[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    int t_out = time_base + threadIdx.x;
    for (int j = threadIdx.y; j < TILE; j += blockDim.y) {
        int f_out = blockIdx.y * TILE + j;
        if (f_out < freq_count && t_out < time_size) {
            data_out[
                (size_t)f_out * (size_t)time_size + (size_t)t_out
            ] = tile[threadIdx.x][j];
        }
    }
}
""",
    "prepare_data",
)


_DE_DISP_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void de_disp(float* dm_time,
             const float* data_cumsum,
             const int*   shift_base,
             int freq_size, int split_pos, int dm_size,
             int freq_start, int freq_count,
             int dm_time_size,
             int data_time_size,
             int down_time_rate,
             int down_freq_rate) {

    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dm_size || y >= dm_time_size) return;

    int base = x * dm_time_size + y;
    int freq_end = freq_start + freq_count;
    int low_end = min(split_pos + 1, freq_end);
    // Continue the original channel-by-channel accumulation across launches.
    float td_i = freq_start == 0 ? 0.0f : dm_time[base];
    float td_mid = freq_start <= split_pos ? 0.0f : dm_time[dm_size * dm_time_size + base];
    int   data_stride = data_time_size;
    int   y_raw = y * down_time_rate;
    float inv_rate = 1.0f / (float)(down_time_rate * down_freq_rate);
    const int* shift_row = shift_base + (size_t)x * (size_t)freq_size;

    for (int idx = freq_start; idx < low_end; idx++) {
        int t0 = shift_row[idx] + y_raw;
        int t1 = t0 + down_time_rate - 1;
        if (t0 >= 0 && t1 < data_time_size) {
            const float* row = data_cumsum
                + (size_t)(idx - freq_start) * (size_t)data_stride;
            float value = row[t1];
            if (t0 > 0) value -= row[t0 - 1];
            td_i += value * inv_rate;
        }
    }
    if (freq_start <= split_pos) td_mid = td_i;

    for (int idx = max(split_pos + 1, freq_start); idx < freq_end; idx++) {
        int t0 = shift_row[idx] + y_raw;
        int t1 = t0 + down_time_rate - 1;
        if (t0 >= 0 && t1 < data_time_size) {
            const float* row = data_cumsum
                + (size_t)(idx - freq_start) * (size_t)data_stride;
            float value = row[t1];
            if (t0 > 0) value -= row[t0 - 1];
            td_i += value * inv_rate;
        }
    }

    // Commit each half-band output once, when its contributing channels finish.
    if (freq_start <= split_pos && freq_end > split_pos)
        dm_time[1 * dm_size * dm_time_size + base] = td_mid;
    if (freq_end == freq_size)
        dm_time[2 * dm_size * dm_time_size + base] = td_i - td_mid;
    dm_time[0 * dm_size * dm_time_size + base] = td_i;
}
""",
    "de_disp",
)


_CUTOUT_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void make_cutout(float* output,
                 const unsigned short* raw_data,
                 const int* shifts,
                 int data_time_size, int freq_size, int start_raw,
                 int output_time_size, int down_time_rate,
                 int down_freq_rate, float raw_scale) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int output_freq_size = freq_size / down_freq_rate;
    int total = output_time_size * output_freq_size;
    if (pos >= total) return;

    int y = pos / output_freq_size;
    int coarse_f = pos - y * output_freq_size;
    int fine_f0 = coarse_f * down_freq_rate;
    float value = 0.0f;

    for (int df = 0; df < down_freq_rate; df++) {
        int f = fine_f0 + df;
        int t0 = start_raw + y * down_time_rate + shifts[f];
        for (int dt = 0; dt < down_time_rate; dt++) {
            int t = t0 + dt;
            if ((unsigned)t < (unsigned)data_time_size) {
                value += (float)raw_data[
                    (size_t)t * (size_t)freq_size + (size_t)f
                ];
            }
        }
    }
    output[pos] = value * raw_scale / (float)(down_time_rate * down_freq_rate);
}
""",
    "make_cutout",
)


_COARSE_TIME_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void make_coarse_time(const unsigned short* raw_data,
                      float* coarse_time,
                      int output_time_size, int freq_size,
                      int rfi_down_time_rate, int down_freq_rate) {
    int coarse_freq_size = freq_size / down_freq_rate;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total = output_time_size * coarse_freq_size;
    if (pos >= total) return;

    int y = pos / coarse_freq_size;
    int coarse_f = pos - y * coarse_freq_size;
    int t0 = y * rfi_down_time_rate;
    int f0 = coarse_f * down_freq_rate;
    float value = 0.0f;
    for (int dt = 0; dt < rfi_down_time_rate; dt++) {
        const unsigned short* row = raw_data
            + (size_t)(t0 + dt) * (size_t)freq_size + (size_t)f0;
        for (int df = 0; df < down_freq_rate; df++) {
            value += (float)row[df];
        }
    }
    coarse_time[pos] = value / (float)(rfi_down_time_rate * down_freq_rate);
}
""",
    "make_coarse_time",
)


# ---------------------------------------------------------------------------
# 3. FITS 读取 + 观测参数
# ---------------------------------------------------------------------------


def load_fits_file(file_name):
    """读取常规 FITS；只合并前两个偏振，保留原始时间和频率采样。"""
    with fits.open(file_name, memmap=True, lazy_load_hdus=True) as f:
        h   = f[1].header
        arr = f[1].data["DATA"].reshape(h["NAXIS2"] * h["NSBLK"], h["NPOL"], h["NCHAN"])
        if reverse_freq:
            arr = arr[:, :, ::-1]
        data = arr[:, 0, :].astype(np.uint16)
        np.add(data, arr[:, 1, :], out=data, casting="unsafe")
    return np.ascontiguousarray(data)


def load_2bit_fits_file(file_name):
    """读取 2-bit FITS；保留原始时间和频率采样。"""
    with fits.open(file_name) as f:
        h    = f[1].header
        data = f[1].data["DATA"]
    data = np.unpackbits(data.reshape(h["NAXIS2"], -1), axis=1).reshape(
        h["NAXIS2"], -1, 2
    )
    data = data[..., 0] << 1 | data[..., 1]
    data = data.reshape(h["NAXIS2"] * h["NSBLK"], h["NCHAN"])
    if reverse_freq:
        data = data[:, ::-1]
    return np.ascontiguousarray(data, dtype=np.uint16)


def get_obparams(file_name, time_factor=8.0):
    """读 fits 头并写入模块级全局变量（freq, freq_reso, time_reso, file_leng,
    down_freq_rate, down_time_rate, rfi_down_time_rate, nbits）。"""
    global freq, freq_reso, time_reso, file_leng, down_freq_rate, down_time_rate
    global rfi_down_time_rate, nbits, reverse_freq
    with fits.open(file_name) as f:
        time_reso = f[1].header["TBIN"]
        freq_reso = f[1].header["NCHAN"]
        file_leng = f[1].header["NAXIS2"] * f[1].header["NSBLK"]
        obs_freq  = f[0].header["OBSFREQ"]
        obs_bw    = f[0].header["OBSBW"]
        nbits     = f[1].header["NBITS"]
    if freq_reso < 512 or freq_reso % 512 != 0:
        raise ValueError(f"NCHAN={freq_reso} 必须是 512 的正整数倍")
    if not np.isfinite(time_factor) or time_factor <= 0:
        raise ValueError(f"time_factor={time_factor} 必须是有限正数")

    obs_bw_abs = abs(float(obs_bw))
    if not np.isfinite(obs_bw_abs) or obs_bw_abs == 0.0:
        raise ValueError(f"OBSBW={obs_bw!r} 必须是有限非零数值")
    reverse_freq = obs_bw < 0
    freq = obs_freq - obs_bw_abs / 2 + (
        np.arange(freq_reso, dtype=np.float32) * obs_bw_abs / freq_reso
    )
    # 每个输出 coarse channel 聚合的原始细通道数；FITS 读取保持原始频率分辨率。
    down_freq_rate = freq_reso // 512
    time_ratio     = (SEARCH_TIME_UNIT_US * time_factor / 1e6) / time_reso
    down_time_rate = int(round(time_ratio))
    if down_time_rate < 1 or not np.isclose(time_ratio, down_time_rate, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"目标时间分辨率 {SEARCH_TIME_UNIT_US:g} us * time_factor={time_factor:g} 必须是 "
            f"TBIN={time_reso:g} s 的正整数倍；当前比值为 {time_ratio:g}"
        )
    rfi_time_ratio     = (RFI_TIME_RESOLUTION_US / 1e6) / time_reso
    rfi_down_time_rate = int(round(rfi_time_ratio))
    if rfi_down_time_rate < 1 or not np.isclose(
        rfi_time_ratio, rfi_down_time_rate, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"RFI 固定时间分辨率 {RFI_TIME_RESOLUTION_US:g} us 必须是 "
            f"TBIN={time_reso:g} s 的正整数倍；当前比值为 {rfi_time_ratio:g}"
        )


def get_fits_start_mjd(file_name):
    """Return FITS start MJD from PSRFITS primary-header timing fields."""
    with fits.open(file_name, memmap=True, lazy_load_hdus=True) as f:
        h = f[0].header
        return (
            float(h["STT_IMJD"])
            + (float(h["STT_SMJD"]) + float(h["STT_OFFS"])) / 86400.0
        )


def _dm_grid_geometry(process_config):
    """校验物理 DM 网格，返回 ``(trial 数, 首 trial, 末 trial)``。"""
    if (
        not np.isfinite(process_config.dm_min)
        or not np.isfinite(process_config.dm_range)
        or not np.isfinite(process_config.dm_step)
    ):
        raise ValueError("dm_min、dm_range 和 dm_step 必须是有限数值")
    if process_config.dm_range <= 0:
        raise ValueError(f"dm_range {process_config.dm_range} 必须大于 0")
    if process_config.dm_step <= 0:
        raise ValueError(f"dm_step {process_config.dm_step} 必须大于 0")

    dm_trials_float = process_config.dm_range / process_config.dm_step
    dm_trials       = int(round(dm_trials_float))
    if not np.isclose(dm_trials_float, dm_trials, rtol=0.0, atol=1e-6):
        raise ValueError(f"dm_range / dm_step = {dm_trials_float}，必须是整数")
    return (
        dm_trials,
        float(process_config.dm_min),
        float(process_config.dm_min + (dm_trials - 1) * process_config.dm_step),
    )


def _dispersion_file_margins(dm_first_trial, dm_last_trial):
    """返回负、正 DM 分别需要向前/向后补读的 FITS 文件数。

    最低频相对最高频的最大正/负延迟（时间降采样前的样本数）换算成文件数。
    依赖 :func:`get_obparams` 已设置的全局量 ``freq`` / ``time_reso`` / ``file_leng``。
    """
    samples_per_dm = 4.15 * (freq.min() ** -2 - freq.max() ** -2) * 1e3 / time_reso
    pre_samples    = int(max(0.0, -dm_first_trial) * samples_per_dm)
    post_samples   = int(max(0.0, dm_last_trial) * samples_per_dm)
    return int(np.ceil(pre_samples / file_leng)), int(np.ceil(post_samples / file_leng))


def compute_dispersion_margins(fits_file, process_config):
    """读取观测参数后返回 ``(前向文件数, 后向文件数)``。"""
    get_obparams(fits_file, process_config.time_factor)
    _, dm_first_trial, dm_last_trial = _dm_grid_geometry(process_config)
    return _dispersion_file_margins(dm_first_trial, dm_last_trial)


def _validate_config(process_config, block_size, down_file_leng):
    """集中校验所有整除 / 配置约束，未通过则一次性列出全部错误并 raise。

    依赖 :func:`get_obparams` 已设置的全局量 ``file_leng`` / ``down_time_rate`` /
    ``rfi_down_time_rate`` / ``freq_reso`` / ``down_freq_rate``。
    """
    errors         = []
    dm_trials      = 0
    dm_first_trial = dm_last_trial = 0.0
    try:
        dm_trials, dm_first_trial, dm_last_trial = _dm_grid_geometry(process_config)
    except ValueError as exc:
        errors.append(str(exc))

    if dm_trials > 0 and dm_trials % DM_TRIALS_PER_CHUNK != 0:
        lower_chunks    = dm_trials // DM_TRIALS_PER_CHUNK
        upper_chunks    = lower_chunks + 1
        suggested_range = []
        if lower_chunks > 0:
            suggested_range.append(
                lower_chunks * DM_TRIALS_PER_CHUNK * process_config.dm_step
            )
        suggested_range.append(
            upper_chunks * DM_TRIALS_PER_CHUNK * process_config.dm_step
        )
        suggestions = " 或 ".join(f"{value:g}" for value in suggested_range)
        errors.append(
            f"DM trial 总数 {dm_trials} 必须是 {DM_TRIALS_PER_CHUNK} 的整数倍；"
            f"保持 dm_step 不变时，可将 dm_range 改为 {suggestions}"
        )
    if block_size <= 0 or block_size % 512 != 0:
        errors.append(f"block_size {block_size} 必须是 512 的整数倍")
    if file_leng % down_time_rate != 0:
        errors.append(
            f"file_leng {file_leng} 必须能被 down_time_rate {down_time_rate} 整除"
        )
    if process_config.rfi_filter and file_leng % rfi_down_time_rate != 0:
        errors.append(
            f"file_leng {file_leng} 必须能被固定 RFI 下采样率 "
            f"rfi_down_time_rate {rfi_down_time_rate} 整除"
        )
    if freq_reso // down_freq_rate != 512:
        errors.append(
            f"freq_reso/down_freq_rate 必须得到 512，当前为 "
            f"{freq_reso}/{down_freq_rate}"
        )
    if process_config.classifier_batch_size <= 0:
        errors.append(
            f"classifier_batch_size {process_config.classifier_batch_size} 必须大于 0"
        )
    if (
        isinstance(process_config.prefix_channel_chunk, bool)
        or not isinstance(process_config.prefix_channel_chunk, (int, np.integer))
        or process_config.prefix_channel_chunk <= 0
    ):
        errors.append("prefix_channel_chunk 必须是正整数")
    if process_config.class_block_size <= 0:
        errors.append(f"class_block_size {process_config.class_block_size} 必须大于 0")
    if (
        isinstance(process_config.fits_prefetch_workers, bool)
        or not isinstance(process_config.fits_prefetch_workers, (int, np.integer))
        or process_config.fits_prefetch_workers <= 0
    ):
        errors.append(
            f"fits_prefetch_workers {process_config.fits_prefetch_workers!r} 必须是正整数"
        )
    if not 0.0 <= process_config.det_prob <= 1.0:
        errors.append(f"det_prob {process_config.det_prob} 必须在 [0, 1] 内")
    if not 0.0 <= process_config.class_threshold <= 1.0:
        errors.append(
            f"class_threshold {process_config.class_threshold} 必须在 [0, 1] 内"
        )
    if process_config.rfi_filter:
        if not 0.0 < process_config.rfi_level_ratio < 1.0:
            errors.append(
                f"rfi_level_ratio {process_config.rfi_level_ratio} 必须在 (0, 1) 内"
            )
        if process_config.rfi_cv_ratio <= 1.0:
            errors.append(f"rfi_cv_ratio {process_config.rfi_cv_ratio} 必须大于 1")
        if process_config.rfi_zero_dm_filter:
            if process_config.rfi_zero_dm_sigma <= 0.0:
                errors.append(
                    f"rfi_zero_dm_sigma {process_config.rfi_zero_dm_sigma} 必须大于 0"
                )
            if not 0.0 < process_config.rfi_zero_dm_occupancy < 1.0:
                errors.append(
                    "rfi_zero_dm_occupancy "
                    f"{process_config.rfi_zero_dm_occupancy} 必须在 (0, 1) 内"
                )
    if down_file_leng % block_size != 0:
        errors.append(
            f"down_file_leng {down_file_leng} 必须能被 block_size {block_size} 整除"
            f"（否则每个 block 尾部 {down_file_leng % block_size} 个样本不会被检索）"
        )
    if errors:
        raise ValueError(
            "process_fits_list 参数自检失败:\n  - " + "\n  - ".join(errors)
        )
    return dm_trials, dm_trials // DM_TRIALS_PER_CHUNK, dm_first_trial, dm_last_trial


# ---------------------------------------------------------------------------
# 4. dedispersion cache 构建
# ---------------------------------------------------------------------------


def _build_dedispersion_cache(
    height, dm_time_width, data_width, process_config, input_origin_raw=0
):
    """预计算细频率通道时移矩阵并分配常驻 GPU 缓冲。

    Args:
        height        : DM trial 数（dm_range / dm_step）。
        dm_time_width : 后降采样输出时间维，已补齐至 block_size 的整数倍。
        data_width    : 原始输入时间维，包含负/正 DM 两侧的读取余量。
        input_origin_raw: 当前搜索 block 起点在输入数组中的原始采样位置。
    """
    freq_float32  = freq.astype(np.float32, copy=False)
    freq_inv2     = np.float32(1.0) / (freq_float32 * freq_float32)
    freq_max_inv2 = float(freq_inv2[-1])

    # cache 覆盖全部细通道；动态 RFI mask 在每个输入窗口上单独计算。
    # FS1 / FS2 严格按原始频带中点二分；4096-channel、1000--1500 MHz 数据即
    # FS1=[1000, 1250) MHz，FS2=[1250, 1500) MHz。
    split_pos = freq_reso // 2 - 1

    dm_values = np.arange(height, dtype=np.float32) * process_config.dm_step + process_config.dm_min
    delta_inv2 = (freq_inv2 - freq_max_inv2).astype(np.float32)
    # shift 必须保留原始采样精度；时间平均在 GPU kernel 内、完成频率对齐之后执行。
    factor         = np.float32(4.15e3 / time_reso)
    shift_unit_raw = factor * delta_inv2
    shift_base     = np.trunc(dm_values[:, None] * shift_unit_raw[None, :]).astype(np.int32)
    shift_base += np.int32(input_origin_raw)

    return DedispersionCache(
        split_pos          = split_pos,
        shift_base_gpu     = cp.asarray(shift_base),
        shift_unit_raw_cpu = shift_unit_raw,
        shift_unit_raw_gpu = cp.asarray(shift_unit_raw),
        dm_time_gpu        = cp.empty((3, height, dm_time_width), dtype=cp.float32),
        raw_data_gpu       = cp.empty((data_width, freq_reso), dtype=cp.uint16),
        data_cumsum_gpu    = cp.empty((min(process_config.prefix_channel_chunk, freq_reso), data_width), dtype=cp.float32),
        dm_time_width      = dm_time_width,
        data_width         = data_width,
    )


def _prepare_rfi_gpu(raw_data_gpu, process_config):
    """生成带通归一化均值、坏 coarse-channel mask 和零 DM 时间 mask。

    数据已经在 GPU；融合 kernel 直接生成固定 393.216 us 时间分辨率、512-channel
    频率分辨率的轻量统计矩阵。RFI 时间分辨率不依赖搜索 ``time_factor``。
    每个 coarse channel 用两项统计量判定：

    * 平均电平低于有效频段中位数的 ``rfi_level_ratio``：带边/坏通道；
    * 时间变异系数高于有效通道中位数的 ``rfi_cv_ratio``：脉冲型 RFI。

    频率 mask 长度固定为 512，被判坏的一项对应 ``down_freq_rate`` 个原始细通道。时间 mask 在
    ``time/rfi_down_time_rate`` 分辨率上，用频率归一化后的全频平均显著性与通道
    占用率联合判定，并向两侧各扩一个 bin。坏频率通道在 DM-time 输入中置零，
    坏时间 bin 以归一化基线 1 替换；没有额外固定频段或固定时间掩码。
    """
    input_time           = int(raw_data_gpu.shape[0])
    coarse_channel_count = freq_reso // down_freq_rate
    rfi_time_bins        = (input_time + rfi_down_time_rate - 1) // rfi_down_time_rate

    if not process_config.rfi_filter:
        coarse_mean_gpu = cp.mean(
            raw_data_gpu.reshape(input_time, coarse_channel_count, down_freq_rate),
            axis  = (0, 2),
            dtype = cp.float32,
        )
        return (
            coarse_mean_gpu,
            cp.zeros(coarse_channel_count, dtype=cp.bool_),
            cp.zeros(rfi_time_bins, dtype=cp.bool_),
        )

    coarse_time_gpu = cp.empty((rfi_time_bins, coarse_channel_count), dtype=cp.float32)
    total = rfi_time_bins * coarse_channel_count
    _COARSE_TIME_KERNEL(
        ((total + 255) // 256,),
        (256,),
        (
            raw_data_gpu,
            coarse_time_gpu,
            rfi_time_bins,
            freq_reso,
            rfi_down_time_rate,
            down_freq_rate,
        ),
    )
    coarse_mean_gpu = coarse_time_gpu.mean(axis=0, dtype=cp.float32)
    coarse_cv_gpu   = coarse_time_gpu.std(axis=0, dtype=cp.float32) / (
        coarse_mean_gpu + np.float32(1.0e-6)
    )
    median_level_gpu = cp.maximum(cp.median(coarse_mean_gpu), np.float32(1.0e-6))
    cv_limit_gpu      = cp.maximum(
        cp.median(coarse_cv_gpu) * np.float32(process_config.rfi_cv_ratio),
        np.float32(1.0e-3),
    )
    bad_channel_mask_gpu = (
        coarse_mean_gpu < median_level_gpu * np.float32(process_config.rfi_level_ratio)
    ) | (coarse_cv_gpu > cv_limit_gpu)
    bad_time_mask_gpu = cp.zeros(rfi_time_bins, dtype=cp.bool_)

    if process_config.rfi_zero_dm_filter:
        good_channel_mask_gpu   = ~bad_channel_mask_gpu
        good_channel_count_gpu  = cp.count_nonzero(good_channel_mask_gpu).astype(cp.float32)
        normalization_count_gpu = cp.maximum(good_channel_count_gpu, np.float32(1.0))
        coarse_time_gpu /= coarse_mean_gpu[None, :] + np.float32(1.0e-6)
        coarse_time_gpu[:, bad_channel_mask_gpu] = np.float32(0.0)
        zero_dm_gpu = coarse_time_gpu.sum(axis=1, dtype=cp.float32) / normalization_count_gpu
        reference_cv_gpu = cp.nanmedian(cp.where(good_channel_mask_gpu, coarse_cv_gpu, cp.nan))
        reference_cv_gpu = cp.nan_to_num(reference_cv_gpu, nan=0.0, copy=False)
        occupancy_gpu = cp.count_nonzero(
            coarse_time_gpu > np.float32(1.0) + np.float32(3.0) * reference_cv_gpu,
            axis=1,
        ) / normalization_count_gpu
        zero_dm_center = cp.median(zero_dm_gpu)
        zero_dm_sigma  = cp.maximum(
            np.float32(1.4826) * cp.median(cp.abs(zero_dm_gpu - zero_dm_center)),
            np.float32(1.0e-4),
        )
        raw_bad_time_gpu = (
            (
                zero_dm_gpu
                > zero_dm_center
                + np.float32(process_config.rfi_zero_dm_sigma) * zero_dm_sigma
            )
            & (occupancy_gpu >= np.float32(process_config.rfi_zero_dm_occupancy))
            & (good_channel_count_gpu > 0)
        )
        bad_time_mask_gpu = raw_bad_time_gpu.copy()
        bad_time_mask_gpu[1:] |= raw_bad_time_gpu[:-1]
        bad_time_mask_gpu[:-1] |= raw_bad_time_gpu[1:]

    if process_config.verbose:
        bad_channel_count = int(cp.count_nonzero(bad_channel_mask_gpu).get())
        bad_time_count    = int(cp.count_nonzero(bad_time_mask_gpu).get())
        rfi_bin_us        = time_reso * rfi_down_time_rate * 1e6
        print(
            f"[RFI] masked {bad_channel_count}/{coarse_channel_count} coarse channels, "
            f"{bad_time_count}/{bad_time_mask_gpu.size} zero-DM bins "
            f"({bad_time_count * rfi_bin_us / 1e3:.3f} ms; {rfi_bin_us:.3f} us/bin)"
        )

    return coarse_mean_gpu, bad_channel_mask_gpu, bad_time_mask_gpu


def d_dm_time_g(data, dd_cache, process_config):
    """在原始时间/频率分辨率消色散后融合下采样，返回常驻显存的结果。

    Args:
        data          : 单个 ``(time, freq_reso)`` uint16 numpy，或此类数组的序列；
                        均未做时间/频率下采样。
    """
    data_parts    = (data,) if isinstance(data, np.ndarray) else tuple(data)
    input_time    = sum(part.shape[0] for part in data_parts)
    height        = int(dd_cache.dm_time_gpu.shape[1])
    dm_time_width = dd_cache.dm_time_width
    if input_time != dd_cache.data_width:
        raise ValueError(
            f"data input_time {input_time} must equal cache.data_width {dd_cache.data_width}"
        )

    if any(part.ndim != 2 or part.shape[1] != freq_reso for part in data_parts):
        raise ValueError(f"each data part must have shape (time, {freq_reso})")

    # 直接逐文件写入 GPU 对应切片，避免 CPU 端每轮重新拼接约 3 GiB 数据。
    offset = 0
    for part in data_parts:
        stop = offset + part.shape[0]
        dd_cache.raw_data_gpu[offset:stop].set(np.ascontiguousarray(part, dtype=np.uint16))
        offset = stop

    # RFI 统计覆盖完整输入窗口，包括负、正 DM 色散所需的读取余量。
    coarse_mean_gpu, bad_channel_mask_gpu, bad_time_mask_gpu = _prepare_rfi_gpu(
        dd_cache.raw_data_gpu, process_config
    )

    BLOCK_TIME, BLOCK_DM = 32, 32
    nthreads = (BLOCK_TIME, BLOCK_DM)
    nblocks  = ((dm_time_width + BLOCK_TIME - 1) // BLOCK_TIME, (height + BLOCK_DM - 1) // BLOCK_DM)
    channel_chunk = dd_cache.data_cumsum_gpu.shape[0]
    # 全窗口 RFI 统计供所有块共享；每块延续原始细通道顺序和 float32 累计值。
    for freq_start in range(0, freq_reso, channel_chunk):
        freq_count = min(channel_chunk, freq_reso - freq_start)
        prefix = dd_cache.data_cumsum_gpu[:freq_count]
        # 时间维使用 grid.x，避免长输入超过 grid.y 的 65535 块限制。
        _PREPARE_KERNEL(
            ((input_time + 31) // 32, (freq_count + 31) // 32),
            (32, 8),
            (dd_cache.raw_data_gpu, coarse_mean_gpu, bad_channel_mask_gpu,
             bad_time_mask_gpu, prefix, input_time, freq_reso, freq_start,
             freq_count, rfi_down_time_rate, down_freq_rate),
        )
        cp.cumsum(prefix, axis=1, out=prefix)
        _DE_DISP_KERNEL(
            nblocks, nthreads,
            (dd_cache.dm_time_gpu, prefix, dd_cache.shift_base_gpu,
             freq_reso, dd_cache.split_pos, height, freq_start, freq_count,
             dm_time_width, input_time, down_time_rate, down_freq_rate),
        )
    del coarse_mean_gpu, bad_channel_mask_gpu, bad_time_mask_gpu
    return dd_cache.dm_time_gpu


# ---------------------------------------------------------------------------
# 5. GPU normalize：percentile → kthvalue 加速
# ---------------------------------------------------------------------------


def _normalize_batch_gpu(img_flat):
    """(N, H, W) GPU tensor → 按图独立做 1% / 99.5% 分位截断 + min-max 归一化到 [0, 1]。

    用 torch.kthvalue 取分位（O(n) selection），比 torch.quantile 基于 sort 的实现快 2-3×。
    """
    flat  = img_flat.reshape(img_flat.shape[0], -1)
    total = flat.shape[1]
    k1    = max(1, int(round(total * 0.010)))
    k2    = max(1, int(round(total * 0.995)))

    vmin = flat.kthvalue(k1, dim=1).values[:, None, None]
    vmax = flat.kthvalue(k2, dim=1).values[:, None, None]
    img_flat.clamp_(vmin, vmax)
    imin = img_flat.amin(dim=(1, 2), keepdim=True)
    imax = img_flat.amax(dim=(1, 2), keepdim=True)
    return img_flat.sub_(imin).div_(imax - imin + 1e-8)


def preprocess_data_gpu(data: torch.Tensor, exp_cut: float = 5.0) -> torch.Tensor:
    """单张 (T, F) GPU tensor 的归一化，与 binary_data.BurstDataset.preprocess 等价。

    quantile → kthvalue 加速；FP 用法上无差异。
    """
    data.add_(1.0)
    data.div_(data.mean(dim=0, keepdim=True) + 1e-8)
    flat  = data.reshape(-1)
    total = flat.shape[0]
    k_lo  = max(1, int(round(total * exp_cut / 100.0)))
    k_hi  = max(1, int(round(total * (1.0 - exp_cut / 100.0))))
    lo    = flat.kthvalue(k_lo).values
    hi    = flat.kthvalue(k_hi).values
    data.clamp_(lo, hi)
    data_min = data.min()
    data_max = data.max()
    return data.sub_(data_min).div_(data_max - data_min + 1e-8)


def _dedisperse_cutout_postdown_gpu(dd_cache, dm_value, toa_samp, output, input_origin_raw=0):
    """把消色散和下采样结果直接写入预分配的 CuPy binary batch 行。"""
    block_size     = int(output.shape[0])
    expected_shape = (block_size, freq_reso // down_freq_rate)
    if (
        not isinstance(output, cp.ndarray)
        or output.shape != expected_shape
        or output.dtype != cp.float32
        or not output.flags.c_contiguous
    ):
        raise ValueError(
            f"binary output 必须是连续 CuPy float32 array {expected_shape}，"
            f"实际为 shape={tuple(output.shape)}, dtype={output.dtype}"
        )
    start_down = max(0, int(toa_samp) - block_size // 2)
    start_raw  = input_origin_raw + start_down * down_time_rate
    shifts_gpu = cp.trunc(float(dm_value) * dd_cache.shift_unit_raw_gpu).astype(cp.int32)
    total = output.size
    _CUTOUT_KERNEL(
        ((total + 127) // 128,),
        (128,),
        (
            output,
            dd_cache.raw_data_gpu,
            shifts_gpu,
            dd_cache.raw_data_gpu.shape[0],
            freq_reso,
            start_raw,
            block_size,
            down_time_rate,
            down_freq_rate,
            np.float32(1.0 if nbits == 2 else 0.5),
        ),
    )


# ---------------------------------------------------------------------------
# 6. 检测解码接口
# ---------------------------------------------------------------------------
# 期望 decode_fn 签名 :
#     decode_fn(raw_pred, conf_thr=...) -> list[(scores_np, centers_xy_np)]


# ---------------------------------------------------------------------------
# 7. 绘图工具
# ---------------------------------------------------------------------------


def _draw_detection(vis, cx, cy, color=(0, 220, 0)):
    """在 CenterNet 亚像素中心坐标处画空心十字。"""
    x, y = int(round(cx)), int(round(cy))
    cv2.line(vis, (x - 9, y), (x - 3, y), color, 1)
    cv2.line(vis, (x + 3, y), (x + 9, y), color, 1)
    cv2.line(vis, (x, y - 9), (x, y - 3), color, 1)
    cv2.line(vis, (x, y + 3), (x, y + 9), color, 1)
    cv2.circle(vis, (x, y), 3, color, 1)


def _save_plot(img_display, new_burst, out_path, d, block_size, process_config):
    """左：DM-time 检测标记，右：dedispersed burst 频谱（time-freq）。"""
    plt.figure(figsize=(7.5, 3))
    plt.subplot(121)
    plt.imshow(img_display, origin="lower")
    detector_time_ms      = block_size * time_reso * down_time_rate * 1e3
    detector_time_seconds = detector_time_ms >= 1000.0
    detector_tick_values  = np.linspace(0, detector_time_ms, 5)
    if detector_time_seconds:
        detector_tick_values /= 1000.0
    plt.xticks(
        np.linspace(0, 512, 5),
        np.round(detector_tick_values, 3 if detector_time_seconds else 1),
    )
    dm_span        = DM_TRIALS_PER_CHUNK * process_config.dm_step
    dm_chunk_start = process_config.dm_min + dm_span * d
    dm_chunk_stop  = dm_chunk_start + dm_span
    dm_ticks       = np.linspace(dm_chunk_start, dm_chunk_stop, 5)
    plt.yticks(np.linspace(0, 512, 5), [f"{value:g}" for value in dm_ticks])
    plt.xlabel(f"Time ({'s' if detector_time_seconds else 'ms'})")
    plt.ylabel("DM (pc cm-3)")
    num_dm_chunks = int(round(process_config.dm_range / dm_span))
    plt.title(
        f"DM chunk {d + 1}/{num_dm_chunks}: {dm_chunk_start:g}-{dm_chunk_stop:g}",
        fontsize=10,
    )

    plt.subplot(122)
    plt.imshow(new_burst.T, origin="lower", cmap=_BURST_CMAP)
    plt.yticks(
        np.linspace(0, 512, 6),
        np.round(np.linspace(freq.min(), freq.max(), 6)).astype(np.int64),
    )
    burst_time_bins    = int(new_burst.shape[0])
    burst_time_ms      = burst_time_bins * time_reso * down_time_rate * 1e3
    burst_time_seconds = burst_time_ms >= 1000.0
    burst_tick_values  = np.linspace(0, burst_time_ms, 5)
    if burst_time_seconds:
        burst_tick_values /= 1000.0
    plt.xticks(
        np.linspace(0, burst_time_bins, 5),
        np.round(burst_tick_values, 3 if burst_time_seconds else 2),
    )
    plt.xlabel(f"Time ({'s' if burst_time_seconds else 'ms'})")
    plt.ylabel("Frequency (MHz)")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 8. 主处理流程：消色散 → 检测候选 → burst 提取 → 分类 → 保存
# ---------------------------------------------------------------------------


def process_fits_list(
    fits_list,
    save_path,
    model,
    class_model,
    process_config,
    decode_fn,
    n_search_files=None,
    task_identifier=None,
    task_info=None,
    bad_fits_log=None,
):
    """两阶段搜索的核心循环（model-agnostic：模型与解码均由 caller 提供）。

    Parameters
    ----------
    fits_list :
        待处理的 fits 文件绝对路径列表（已排序）。
    save_path :
        结果保存目录。
    model :
        中心检测模型（已 ``.eval()``、已 ``.to(device)``）。
    class_model :
        二分类模型（已 ``.eval()``、已 ``.to(device)``）。
    process_config :
        :class:`ProcessConfig` 实例。
    decode_fn :
        ``decode_fn(raw_pred, conf_thr) -> list[(scores_np, centers_xy_np)]``。
    n_search_files :
        仅对前 ``n_search_files`` 个文件启动检索 block，其余文件只作消色散读取提前量
        （gate.py 按 GPU 切分单个连续观测时用，避免段间重复检索）。``None`` 表示全部检索。
    """
    if not fits_list:
        return
    if n_search_files is not None and n_search_files < 0:
        raise ValueError(f"n_search_files={n_search_files} 不能为负数")

    fits_list  = list(fits_list)
    block_size = process_config.block_size
    os.makedirs(save_path, exist_ok=True)
    manifest_path = os.path.join(save_path, process_config.manifest_name)
    get_obparams(fits_list[0], process_config.time_factor)

    read_prefix_files = int((task_info or {}).get("read_prefix_files", 0))
    if read_prefix_files < 0 or read_prefix_files >= len(fits_list):
        raise ValueError(f"read_prefix_files={read_prefix_files} 与输入文件数不兼容")

    input_start_mjd = get_fits_start_mjd(fits_list[read_prefix_files])

    block_file = int(np.ceil(down_time_rate * block_size / file_leng))
    down_file_leng = block_file * file_leng // down_time_rate
    # 短文件向上补齐检测窗口，缺少的原始采样复用后面的随机噪声补齐逻辑。
    down_file_leng = ((down_file_leng + block_size - 1) // block_size) * block_size

    # 校验配置并把物理 DM 区间换算成 trial 几何和读取余量。
    dm_trials, num_dm_chunks, dm_first_trial, dm_last_trial = _validate_config(
        process_config, block_size, down_file_leng
    )
    required_prefix_files, dds_file = _dispersion_file_margins(dm_first_trial, dm_last_trial)
    comb_file        = required_prefix_files + block_file + dds_file  # 前余量 + block + 后余量
    input_origin_raw = required_prefix_files * file_leng
    if process_config.verbose:
        print(block_file, comb_file)

    comb_file_leng = (
        (required_prefix_files + dds_file) * file_leng + down_file_leng * down_time_rate
    )  # 补齐后的原始输入长度，包含前后色散余量

    dd_cache = _build_dedispersion_cache(
        dm_trials,
        down_file_leng,
        comb_file_leng,
        process_config,
        input_origin_raw=input_origin_raw,
    )

    # 检测图几何：[dm_min, dm_min + dm_range) 每 1024 个 trial 切一块，再平均到 512 行。
    dm_span             = DM_TRIALS_PER_CHUNK * process_config.dm_step  # 每张图的物理 DM 宽度
    dm_per_pixel_trials = DM_TRIALS_PER_CHUNK // 512                    # 固定为每像素 2 个 trial
    dm_per_pixel        = dm_per_pixel_trials * process_config.dm_step
    time_pool = block_size // 512  # 时间轴 pool 的倍数（block_size → 512）
    dm_max = process_config.dm_min + process_config.dm_range

    # 相邻 block 重叠 comb_file - block_file 个 fits；sliding-window 缓存让每个文件只读一次盘
    # 每文件只在 load_func 内把前两个偏振相加；时间和频率轴都保持 FITS 原始采样。
    file_cache       = {}
    load_func        = load_2bit_fits_file if nbits == 2 else load_fits_file
    prefetch_futures = {}
    random_generator = np.random.default_rng()

    # n_search_files 之外的文件仅提供色散读取余量，不启动搜索 block。
    searchable_file_count = len(fits_list) - read_prefix_files
    search_file_count     = (
        searchable_file_count
        if n_search_files is None
        else min(n_search_files, searchable_file_count)
    )
    search_start     = read_prefix_files
    search_stop      = search_start + search_file_count
    last_block_start = search_start + ((max(search_file_count, 1) - 1) // block_file) * block_file
    final_read_stop = min(
        len(fits_list),
        last_block_start - required_prefix_files + comb_file,
    )

    manifest_context = (
        open(manifest_path, "a", encoding="utf-8", buffering=1)
        if process_config.save_manifest
        else nullcontext(None)
    )
    with (
        manifest_context as manifest_stream,
        ThreadPoolExecutor(max_workers=int(process_config.fits_prefetch_workers)) as prefetch_executor,
    ):
        for i in range(search_start, search_stop, block_file):
            filename = os.path.splitext(os.path.basename(fits_list[i]))[0]
            if i == search_start:
                block_start_mjd = input_start_mjd
            else:
                try:
                    block_start_mjd = get_fits_start_mjd(fits_list[i])
                except (OSError, ValueError, KeyError, IndexError, TypeError):
                    searched_index  = i - read_prefix_files
                    block_start_mjd = (
                        input_start_mjd
                        + searched_index * file_leng * time_reso / 86400.0
                    )
            block_search_fits_count = min(block_file, search_stop - i)
            if process_config.verbose:
                print(filename)

            # ── 文件读取与拼接（带 sliding-window 缓存）─────────────────────────
            input_start          = i - required_prefix_files
            prefix_padding_files = max(0, -input_start)
            read_start           = max(0, input_start)
            read_stop            = min(len(fits_list), input_start + comb_file)
            needed = [
                fits_list[file_index]
                for file_index in range(read_start, read_stop)
            ]
            for fp in list(file_cache):
                if fp not in needed:
                    del file_cache[fp]

            # 当前窗口和后续 FITS 按文件顺序进入预读队列，队列深度由 fits_prefetch_workers 控制。
            prefetch_stop = min(
                final_read_stop,
                read_stop + int(process_config.fits_prefetch_workers),
            )
            for file_index in range(read_start, prefetch_stop):
                fp = fits_list[file_index]
                if fp not in file_cache and fp not in prefetch_futures:
                    prefetch_futures[fp] = prefetch_executor.submit(load_func, fp)

            for fp in needed:
                if fp not in file_cache:
                    try:
                        file_cache[fp] = prefetch_futures.pop(fp).result()
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        IndexError,
                        TypeError,
                    ) as exc:
                        if bad_fits_log:
                            with open(bad_fits_log, "a", encoding="utf-8") as f:
                                f.write(
                                    f"{task_identifier}\t{fp}\t{type(exc).__name__}: {exc}\n"
                                )
                        print(f"[Bad FITS] {fp}: {exc}; using random replacement")
                        file_cache[fp] = random_generator.integers(
                            0,
                            4 if nbits == 2 else 512,
                            size  = (file_leng, freq_reso),
                            dtype = np.uint16,
                        )
            raw_data = []
            if prefix_padding_files:
                # 观测起点没有更早数据时用噪声补齐，避免重复首个 FITS 产生周期性伪结构。
                raw_data.append(
                    random_generator.integers(
                        0,
                        4 if nbits == 2 else 512,
                        size  = (prefix_padding_files * file_leng, freq_reso),
                        dtype = np.uint16,
                    )
                )
            raw_data.extend(file_cache[fp] for fp in needed)

            loaded_samples = sum(part.shape[0] for part in raw_data)
            if loaded_samples < comb_file_leng:
                padding = random_generator.integers(
                    0,
                    4 if nbits == 2 else 512,
                    size  = (comb_file_leng - loaded_samples, freq_reso),
                    dtype = np.uint16,
                )
                raw_data.append(padding)

            # ── 原始时间/频率精度上传 → 消色散 → 融合时间/频率平均 ───────────────
            new_data = d_dm_time_g(raw_data, dd_cache, process_config)

            # ── 全 GPU 后处理 ───────────────────────────────────────────────────
            new_data_t = torch.from_dlpack(new_data)  # (3, dm_trials, down_file_leng)，零拷贝视图

            num_slices = down_file_leng // block_size
            if process_config.verbose:
                print(f"Processing {num_slices} slices, {num_dm_chunks} DM chunks")

            for j in range(num_slices):
                # ── 切片 → 降采样 → 归一化（全在显存）──────────────────────────
                slice_data = new_data_t[:, :, j * block_size : (j + 1) * block_size]
                # 一次 reshape+mean 完成每个 chunk 内的 DM/时间等宽平均：
                #   DM trial 轴 → (num_dm_chunks, 512, dm_per_pixel_trials)，时间轴 → (512, time_pool)
                # 等宽均值池化满足结合律，故可一次性 pool 成 num_dm_chunks 张 512x512 检测图
                img_batch = slice_data.reshape(
                    3, num_dm_chunks, 512, dm_per_pixel_trials, 512, time_pool
                ).mean(dim=(3, 5))  # (3, num_dm_chunks, 512, 512)
                img_batch_flat = img_batch.flatten(0, 1)               # (N, 512, 512)
                img_batch_flat = _normalize_batch_gpu(img_batch_flat)  # kthvalue 加速版

                # ── 检测推理（数据已在显存，无 H2D；单通道输入）────────────────
                det_input = img_batch_flat.unsqueeze(1)  # (N, 1, 512, 512)
                with torch.inference_mode():
                    raw_pred = model(det_input)
                det_results = decode_fn(raw_pred, conf_thr=process_config.det_prob)
                del raw_pred, det_input
                if len(det_results) != img_batch_flat.shape[0]:
                    raise ValueError(
                        f"decode_fn 返回 {len(det_results)} 张图，"
                        f"但 detector batch 为 {img_batch_flat.shape[0]}"
                    )

                # 只保存轻量元数据；binary cutout 在下面按分类 batch 现算、现用、现释放。
                candidates = []
                for idx, (pred_scores, pred_centers) in enumerate(det_results):
                    if pred_centers is None:
                        continue
                    pred_centers = np.asarray(pred_centers)
                    if pred_centers.ndim != 2 or pred_centers.shape[1] != 2:
                        raise ValueError(
                            f"decode_fn 第 {idx} 张图必须返回 (N, 2) centers，"
                            f"实际为 {pred_centers.shape}"
                        )
                    if pred_scores is not None and len(pred_scores) != len(pred_centers):
                        raise ValueError(
                            f"decode_fn 第 {idx} 张图的 scores/centers 数量不一致："
                            f"{len(pred_scores)} != {len(pred_centers)}"
                        )
                    d = idx % num_dm_chunks
                    for center_idx, center in enumerate(pred_centers):
                        cx, cy = float(center[0]), float(center[1])
                        if not np.isfinite(cx) or not np.isfinite(cy):
                            continue
                        score_val = float(pred_scores[center_idx]) if pred_scores is not None else 0.0
                        dm_value = process_config.dm_min + d * dm_span + cy * dm_per_pixel
                        if dm_value <= process_config.dm_threshold:
                            continue
                        toa_samp = int(round(cx * time_pool + j * block_size))
                        candidates.append(
                            {
                                "box_idx": center_idx,
                                "idx": idx,
                                "cx": cx,
                                "cy": cy,
                                "dm_value": float(dm_value),
                                "score": score_val,
                                "toa_samp": toa_samp,
                            }
                        )

                if not candidates:
                    del img_batch_flat, img_batch, slice_data
                    continue

                # 仅共享起始采样和逐细通道整数时移完全相同的 cutout。
                cutout_groups = {}
                for cand in candidates:
                    start_down = max(0, cand["toa_samp"] - process_config.class_block_size // 2)
                    shift_signature = (
                        np.trunc(
                            np.float32(cand["dm_value"]) * dd_cache.shift_unit_raw_cpu,
                        )
                        .astype(np.int32)
                        .tobytes()
                    )
                    cutout_groups.setdefault((start_down, shift_signature), []).append(cand)
                candidate_groups = list(cutout_groups.values())
                if process_config.verbose and len(candidate_groups) < len(candidates):
                    print(
                        f"[Binary] reused {len(candidates) - len(candidate_groups)}/"
                        f"{len(candidates)} exactly equivalent cutouts"
                    )

                # kernel 按 classifier_batch_size 直接生成分类 batch，控制 cutout 显存占用。
                npy_paths     = {}
                display_cache = {}
                for batch_start in range(
                    0, len(candidate_groups), process_config.classifier_batch_size
                ):
                    batch_candidate_groups = candidate_groups[
                        batch_start : batch_start + process_config.classifier_batch_size
                    ]
                    burst_batch_gpu = cp.empty(
                        (
                            len(batch_candidate_groups),
                            process_config.class_block_size,
                            512,
                        ),
                        dtype=cp.float32,
                    )
                    for burst_index, candidate_group in enumerate(batch_candidate_groups):
                        cand = candidate_group[0]
                        _dedisperse_cutout_postdown_gpu(
                            dd_cache,
                            cand["dm_value"],
                            cand["toa_samp"],
                            burst_batch_gpu[burst_index],
                            input_origin_raw=input_origin_raw,
                        )

                    # 整个 batch 只做一次 DLPack 零拷贝交接，再原位归一化。
                    burst_batch = torch.from_dlpack(burst_batch_gpu)
                    del burst_batch_gpu
                    for new_burst in burst_batch:
                        preprocess_data_gpu(new_burst)

                    with torch.inference_mode():
                        class_scores = (
                            torch.softmax(
                                class_model(burst_batch.unsqueeze(1)),
                                dim=1,
                            )[:, 1]
                            .cpu()
                            .tolist()
                        )

                    # 保存当前 batch 的正样本后立即释放 burst_batch。
                    for candidate_group, new_burst, cls_score in zip(
                        batch_candidate_groups, burst_batch, class_scores
                    ):
                        if cls_score <= process_config.class_threshold:
                            continue
                        burst_for_plot = new_burst.cpu().numpy() if process_config.save_plot else None
                        for cand in candidate_group:
                            k, d = divmod(cand["idx"], num_dm_chunks)

                            if process_config.verbose:
                                print(
                                    f"Score: {cand['score']:.3f}, DM: {cand['dm_value']:.2f}, "
                                    f"CLS: {cls_score:.3f}"
                                )

                            npy_path = None
                            if process_config.save_npy:
                                npy_path = npy_paths.get(k)
                                if npy_path is None:
                                    # 同一时间片/频带只回传和保存一次，避免候选间重复搬运。
                                    data_slice = cp.asnumpy(
                                        new_data[
                                            k, :, j * block_size : (j + 1) * block_size
                                        ]
                                    )
                                    npy_path = os.path.join(
                                        save_path,
                                        f"{filename}-TS{j:0>2d}-FS{k}.npy",
                                    )
                                    np.save(npy_path, data_slice)
                                    npy_paths[k] = npy_path

                            jpg_path = None
                            if process_config.save_plot:
                                base_display = display_cache.get(cand["idx"])
                                if base_display is None:
                                    base_display = (
                                        img_batch_flat[cand["idx"]].cpu().numpy() * 255
                                    ).astype(np.uint8)
                                    base_display = cv2.cvtColor(base_display, cv2.COLOR_GRAY2BGR)
                                    display_cache[cand["idx"]] = base_display
                                img_display = base_display.copy()
                                _draw_detection(img_display, cand["cx"], cand["cy"])
                                jpg_path = os.path.join(
                                    save_path,
                                    f"{filename}-TS{j:0>2d}-FS{k}-BX{cand['box_idx']}"
                                    f"-DM{cand['dm_value']:.1f}.jpg",
                                )
                                _save_plot(
                                    img_display,
                                    burst_for_plot,
                                    jpg_path,
                                    d,
                                    block_size,
                                    process_config,
                                )

                            if manifest_stream is not None:
                                toa_sec    = cand["toa_samp"] * time_reso * down_time_rate
                                signal_mjd = block_start_mjd + toa_sec / 86400.0
                                task_meta  = task_info or {}
                                record     = {
                                    "task_identifier": task_identifier,
                                    "root": task_meta.get("root"),
                                    "source": task_meta.get("source"),
                                    "date": task_meta.get("date"),
                                    "beam": task_meta.get("beam"),
                                    "input_start_mjd": input_start_mjd,
                                    "block_start_mjd": block_start_mjd,
                                    "block_start_fits_path": fits_list[i],
                                    "block_start_fits_index": int(
                                        i - read_prefix_files
                                    ),
                                    "block_fits_count": int(block_search_fits_count),
                                    "toa_sec": toa_sec,
                                    "signal_mjd": signal_mjd,
                                    "toa_sample_from_block_start": int(
                                        cand["toa_samp"] * down_time_rate
                                    ),
                                    "toa_sample_from_input_start": int(
                                        (i - read_prefix_files) * file_leng
                                        + cand["toa_samp"] * down_time_rate
                                    ),
                                    "time_reso": float(time_reso),
                                    "down_time_rate": int(down_time_rate),
                                    "rfi_down_time_rate": int(rfi_down_time_rate),
                                    "rfi_time_reso": float(time_reso * rfi_down_time_rate),
                                    "file_leng": int(file_leng),
                                    "slice_index": int(j),
                                    "freq_slice": int(k),
                                    "dm_chunk": int(d),
                                    "box_idx": int(cand["box_idx"]),
                                    "cx": float(cand["cx"]),
                                    "cy": float(cand["cy"]),
                                    "dm": float(cand["dm_value"]),
                                    "det_score": float(cand["score"]),
                                    "class_score": float(cls_score),
                                    "det_prob": float(process_config.det_prob),
                                    "class_threshold": float(process_config.class_threshold),
                                    "dm_threshold": float(process_config.dm_threshold),
                                    "dm_min": float(process_config.dm_min),
                                    "dm_range": float(process_config.dm_range),
                                    "dm_max": float(dm_max),
                                    "dm_step": float(process_config.dm_step),
                                    "block_size": int(process_config.block_size),
                                    "dm_span": float(dm_span),
                                    "dm_trials": int(dm_trials),
                                    "dm_chunk_trials": int(DM_TRIALS_PER_CHUNK),
                                    "dm_last_trial": float(dm_last_trial),
                                    "dm_chunk_start": float(
                                        process_config.dm_min + d * dm_span
                                    ),
                                    "dm_chunk_stop": float(
                                        process_config.dm_min + (d + 1) * dm_span
                                    ),
                                    "jpg_path": jpg_path,
                                    "npy_path": npy_path,
                                }
                                manifest_stream.write(
                                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                                )
                    del new_burst, burst_batch

                del img_batch_flat, img_batch, slice_data

            del new_data_t, new_data, raw_data
# fmt: on
