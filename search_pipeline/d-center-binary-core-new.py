"""两阶段搜索核心（原始时间/频率分辨率 GPU 消色散版）。

数据流（全 GPU 流水线）：
    CPU                          ：FITS 读盘并合并前两个偏振；不做时间/频率下采样
    CPU → GPU                    ：uint16 原始时间 × 原始频率数据
    GPU                          ：细频率通道消色散，再融合完成时间与 4096→512 频率平均
    GPU 常驻                    ：下采样后的 dm_time，零拷贝桥接为 torch tensor
    GPU 完成                    ：reshape / normalize / 中心检测 / binary cutout / 分类
    CPU 仅用于                  ：FITS 读盘、候选元数据与可视化保存（仅检测命中时）

**本文件 model-agnostic**：检测模型与解码函数全部由 caller（gate.py）传入。
``decode_fn`` 必须接受 ``(raw_pred, conf_thr, iou_thr)`` 并返回每图
``(scores_np, boxes_cxcywh_np)`` 列表（CenterNet 由 gate.py 用
``centers → 占位 box`` 的小适配器对齐到该接口）。

binary 分类器使用 :mod:`binary_model.build_binary_model`（与训练侧 ``binary_classification/`` 同步）。

性能优化点：
  1. FITS 缓存保存 uint16 偏振和，主机内存与 H2D 流量仅为 float32 的一半
  2. tiled transpose kernel 在 GPU 上直接生成 (freq, time) 布局
  3. 相邻 32 个 DM 同块计算，利用其相近时移复用只读缓存数据
  4. 原位时间前缀和让每个输出只需两次读取即可完成时间平均
  5. shift_base 与原始数据缓冲常驻显存，候选切片不再重复 CPU→GPU
  6. binary cutout 用单个 kernel 融合细通道消色散、时间平均和频率平均
  7. CuPy ↔ PyTorch 零拷贝桥接；归一化使用 ``torch.kthvalue``
  8. 用已上传 GPU 的当前搜索块生成轻量 512-channel RFI 掩码；坏 coarse channel
     对应的 8 个细通道不参与 DM-time 求和

已验证的实时部署是单张 NVIDIA L40 运行单个 worker；完整 4096-DM 配置连同两套模型
约占 11.6 GiB，不能直接沿用旧 PBS 示例的一卡四 worker 设置。
"""

import os, cv2, json
cv2.setNumThreads(4)

import numpy as np
import cupy as cp
import torch
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_color_codes()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    dm_min:                float = 0.0
    dm_range:              float = 4096.0
    dm_step:               float = 1.0
    dm_threshold:          float = 50.0
    block_size:            int   = 8192
    det_prob:              float = 0.45
    section_num:           int   = 5
    time_factor:           float = 8.0
    det_iou:               float = 0.5
    class_block_size:      int   = 512
    class_threshold:       float = 0.5
    classifier_batch_size: int   = 64
    rfi_filter:            bool  = True
    rfi_level_ratio:       float = 0.05
    rfi_cv_ratio:          float = 3.0
    rfi_zero_dm_filter:    bool  = True
    rfi_zero_dm_sigma:     float = 8.0
    rfi_zero_dm_occupancy: float = 0.03
    save_npy:              bool  = True
    save_plot:             bool  = True
    save_manifest:         bool  = True
    manifest_name:         str   = 'candidate_manifest.jsonl'
    verbose:               bool  = False


# Detector 输入的 DM 轴固定为 512 像素；保持现有模型使用的每像素 2 个 trial。
DM_TRIALS_PER_CHUNK = 1024


@dataclass
class DedispersionCache:
    """预计算 + 常驻显存的消色散缓存（在多个 block_file 间复用）。

    输入 data 比输出 dm_time 宽：dm_time_width 是时间后降采样后的 block 输出长度，
    data_width 是原始时间采样长度，包含色散延迟所需的读取提前量。
    """
    split_pos:      int          # 保持旧 512-channel core 的低/高频分割位置
    shift_base_gpu: cp.ndarray   # (dm_trials, freq_reso)        int32
    dm_time_gpu:    cp.ndarray   # (3, dm_trials, dm_time_width) float32
    raw_data_gpu:   cp.ndarray   # (data_width, freq_reso)      uint16（前两个偏振之和）
    data_cumsum_gpu: cp.ndarray  # (freq_reso, data_width)      float32（原位前缀和）
    dm_time_width:  int          # 输出时间维（block_file 长度）
    data_width:     int          # 输入时间维（comb_file 长度，含色散提前量）


# ---------------------------------------------------------------------------
# 2. CuPy RawKernel：转置/归一化、消色散、binary cutout
# ---------------------------------------------------------------------------
#
# 当前 kernel 布局（参数调整必须重新基准）：
#   1. data 转置成 (freq, time) → 同 warp 内 shift 连续 → 完美合并访问
#   2. block 维度 (BLOCK_TIME, BLOCK_DM) = (32, 32)，一个 warp 对应一个 DM
#   3. 同块相邻 DM 的 shift 只差少量样本，前缀数据可由 L1/L2 大量复用
#   4. shift_base 走只读缓存；省去 shared-memory 搬运、同步和 DM 数限制
#   5. split_pos 切两段循环，消除 hot loop 内分支
#   6. __ldg / __restrict__ 编译器提示
# ---------------------------------------------------------------------------

# Kernel sources are kept ASCII-only because some NVRTC builds reject non-ASCII comments.
_PREPARE_KERNEL = cp.RawKernel(r'''
#define TILE 32
extern "C" __global__
void prepare_data(const unsigned short* __restrict__ raw_data,
                  const float* __restrict__ coarse_mean,
                  const unsigned char* __restrict__ rfi_bad_coarse,
                  const unsigned char* __restrict__ rfi_bad_time,
                  float* __restrict__ data_out,
                  int time_size, int freq_size,
                  int down_time_rate, int down_freq_rate) {
    __shared__ float tile[TILE][TILE + 1];

    int f_in = blockIdx.x * TILE + threadIdx.x;
    int time_base = blockIdx.y * TILE;

    #pragma unroll
    for (int j = threadIdx.y; j < TILE; j += blockDim.y) {
        int t = time_base + j;
        if (f_in < freq_size && t < time_size) {
            float denom = coarse_mean[f_in / down_freq_rate] + 1.0e-8f;
            tile[j][threadIdx.x] = rfi_bad_coarse[f_in / down_freq_rate]
                ? 0.0f : (rfi_bad_time[t / down_time_rate]
                    ? 1.0f : (float)raw_data[t * freq_size + f_in] / denom);
        } else {
            tile[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    int t_out = time_base + threadIdx.x;
    #pragma unroll
    for (int j = threadIdx.y; j < TILE; j += blockDim.y) {
        int f_out = blockIdx.x * TILE + j;
        if (f_out < freq_size && t_out < time_size) {
            data_out[f_out * time_size + t_out] = tile[threadIdx.x][j];
        }
    }
}
''', 'prepare_data')


_DE_DISP_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void de_disp(float* __restrict__ dm_time,
             const float* __restrict__ data_cumsum,
             const int*   __restrict__ shift_base,
             int freq_size, int split_pos, int dm_size,
             int dm_time_size,
             int data_time_size,
             int down_time_rate,
             int down_freq_rate) {

    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dm_size || y >= dm_time_size) return;

    float td_i = 0.0f, td_mid = 0.0f;
    int   data_stride = data_time_size;
    int   y_raw = y * down_time_rate;
    float inv_rate = 1.0f / (float)(down_time_rate * down_freq_rate);
    const int* shift_row = shift_base + x * freq_size;

    #pragma unroll 4
    for (int idx = 0; idx <= split_pos; idx++) {
        int t0 = __ldg(&shift_row[idx]) + y_raw;
        int t1 = t0 + down_time_rate - 1;
        if (t0 >= 0 && t1 < data_time_size) {
            const float* row = data_cumsum + idx * data_stride;
            float value = __ldg(&row[t1]);
            if (t0 > 0) value -= __ldg(&row[t0 - 1]);
            td_i += value * inv_rate;
        }
    }
    td_mid = td_i;

    #pragma unroll 4
    for (int idx = split_pos + 1; idx < freq_size; idx++) {
        int t0 = __ldg(&shift_row[idx]) + y_raw;
        int t1 = t0 + down_time_rate - 1;
        if (t0 >= 0 && t1 < data_time_size) {
            const float* row = data_cumsum + idx * data_stride;
            float value = __ldg(&row[t1]);
            if (t0 > 0) value -= __ldg(&row[t0 - 1]);
            td_i += value * inv_rate;
        }
    }

    int base = x * dm_time_size + y;
    dm_time[1 * dm_size * dm_time_size + base] = td_mid;
    dm_time[2 * dm_size * dm_time_size + base] = td_i - td_mid;
    dm_time[0 * dm_size * dm_time_size + base] = td_i;
}
''', 'de_disp')


_CUTOUT_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void make_cutout(float* __restrict__ output,
                 const unsigned short* __restrict__ raw_data,
                 const int* __restrict__ shifts,
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
                value += (float)raw_data[t * freq_size + f];
            }
        }
    }
    output[pos] = value * raw_scale / (float)(down_time_rate * down_freq_rate);
}
''', 'make_cutout')


_COARSE_TIME_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void make_coarse_time(const unsigned short* __restrict__ raw_data,
                      float* __restrict__ coarse_time,
                      int output_time_size, int freq_size,
                      int down_time_rate, int down_freq_rate) {
    int coarse_freq_size = freq_size / down_freq_rate;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total = output_time_size * coarse_freq_size;
    if (pos >= total) return;

    int y = pos / coarse_freq_size;
    int coarse_f = pos - y * coarse_freq_size;
    int t0 = y * down_time_rate;
    int f0 = coarse_f * down_freq_rate;
    float value = 0.0f;
    for (int dt = 0; dt < down_time_rate; dt++) {
        const unsigned short* row = raw_data + (t0 + dt) * freq_size + f0;
        for (int df = 0; df < down_freq_rate; df++) {
            value += (float)row[df];
        }
    }
    coarse_time[pos] = value / (float)(down_time_rate * down_freq_rate);
}
''', 'make_coarse_time')


# ---------------------------------------------------------------------------
# 3. FITS 读取 + 观测参数
# ---------------------------------------------------------------------------

def load_fits_file(file_name, reverse_flag=False):
    """读取常规 FITS；只合并前两个偏振，保留原始时间和频率采样。"""
    with fits.open(file_name, memmap=True, lazy_load_hdus=True) as f:
        h = f[1].header
        arr = f[1].data['DATA'].reshape(h['NAXIS2'] * h['NSBLK'], h['NPOL'], h['NCHAN'])
        if reverse_flag:
            arr = arr[:, :, ::-1]
        data = arr[:, 0, :].astype(np.uint16)
        np.add(data, arr[:, 1, :], out=data, casting='unsafe')
    return np.ascontiguousarray(data)


def load_2bit_fits_file(file_name):
    """读取 2-bit FITS；保留原始时间和频率采样。"""
    with fits.open(file_name) as f:
        h = f[1].header
        data = f[1].data['DATA']
    data = np.unpackbits(data.reshape(h['NAXIS2'], -1), axis=1).reshape(h['NAXIS2'], -1, 2)
    data = data[..., 0] << 1 | data[..., 1]
    return data.reshape(h['NAXIS2'] * h['NSBLK'], h['NCHAN']).astype(np.uint16, copy=False)


def get_obparams(file_name, time_factor=8.0):
    """读 fits 头并写入模块级全局变量（freq, freq_reso, time_reso, file_leng,
    down_freq_rate, down_time_rate, nbits）。"""
    global freq, freq_reso, time_reso, file_leng, down_freq_rate, down_time_rate, nbits
    with fits.open(file_name) as f:
        time_reso = f[1].header['TBIN']
        freq_reso = f[1].header['NCHAN']
        file_leng = f[1].header['NAXIS2'] * f[1].header['NSBLK']
        obs_freq  = f[0].header['OBSFREQ']
        obs_bw    = f[0].header['OBSBW']
        nbits     = f[1].header['NBITS']
        freq      = obs_freq - obs_bw / 2 + np.arange(freq_reso) * obs_bw / freq_reso
    # 这里不是 FITS 读取阶段的预下采样率，而是细通道完成时移对齐后，在 GPU 上
    # 聚合到 512 个 coarse channels 的频率 bin 宽度。
    down_freq_rate = int(freq_reso / 512)
    down_time_rate = int((49.152 * time_factor / 1e6) / time_reso)


def get_fits_start_mjd(file_name):
    """Return FITS start MJD from PSRFITS primary-header timing fields."""
    with fits.open(file_name, memmap=True, lazy_load_hdus=True) as f:
        h = f[0].header
        return float(h['STT_IMJD']) + (float(h['STT_SMJD']) + float(h['STT_OFFS'])) / 86400.0


def _dispersion_file_span(process_config):
    """覆盖最大色散延迟所需的 fits 文件数（dds_file）。

    扫到 DM 上限时，最低频相对最高频的最大延迟（时间降采样前的样本数）换算成文件数。
    trial 位于半开区间 ``[dm_min, dm_min + dm_range)``。
    依赖 :func:`get_obparams` 已设置的全局量 ``freq`` / ``time_reso`` / ``file_leng``。
    """
    if (not np.isfinite(process_config.dm_min)
            or not np.isfinite(process_config.dm_range)
            or not np.isfinite(process_config.dm_step) or process_config.dm_step <= 0):
        raise ValueError('dm_min/dm_range 必须是有限数值，dm_step 必须大于 0')
    if process_config.dm_range <= 0:
        raise ValueError(f'dm_range {process_config.dm_range} 必须大于 0')
    dm_trials_float = process_config.dm_range / process_config.dm_step
    dm_trials = int(round(dm_trials_float))
    if not np.isclose(dm_trials_float, dm_trials, rtol=0.0, atol=1e-6):
        raise ValueError(f'dm_range / dm_step = {dm_trials_float}，'
                         '必须是整数')
    dm_last_trial = process_config.dm_min + (dm_trials - 1) * process_config.dm_step
    dds_max = int(4.15 * max(0.0, dm_last_trial)
                  * (freq.min() ** -2 - freq.max() ** -2) * 1e3 / time_reso)
    return int(np.ceil(dds_max / file_leng))


def compute_dispersion_overlap(fits_file, process_config):
    """读取 ``fits_file`` 观测参数后返回相邻文件区间所需的色散重叠文件数。

    供 gate.py 按 GPU 切分单个连续观测时给每段补读取提前量，避免段尾高 DM 灵敏度下降。
    """
    get_obparams(fits_file, process_config.time_factor)
    return _dispersion_file_span(process_config)


def _validate_config(process_config, block_size, down_file_leng):
    """集中校验所有整除 / 配置约束，未通过则一次性列出全部错误并 raise。

    依赖 :func:`get_obparams` 已设置的全局量 ``file_leng`` / ``down_time_rate`` /
    ``freq_reso`` / ``down_freq_rate``。
    """
    errors = []
    dm_trials = 0

    if not np.isfinite(process_config.dm_min):
        errors.append(f'dm_min {process_config.dm_min} 必须是有限数值')
    if not np.isfinite(process_config.dm_range):
        errors.append(f'dm_range {process_config.dm_range} 必须是有限数值')
    elif process_config.dm_range <= 0:
        errors.append(f'dm_range {process_config.dm_range} 必须大于 0')
    if not np.isfinite(process_config.dm_step) or process_config.dm_step <= 0:
        errors.append(f'dm_step {process_config.dm_step} 必须大于 0')

    if (np.isfinite(process_config.dm_range) and process_config.dm_range > 0
            and np.isfinite(process_config.dm_step) and process_config.dm_step > 0):
        dm_trials_float = process_config.dm_range / process_config.dm_step
        rounded_dm_trials = int(round(dm_trials_float))
        if np.isclose(dm_trials_float, rounded_dm_trials, rtol=0.0, atol=1e-6):
            dm_trials = rounded_dm_trials
        else:
            errors.append(f'dm_range / dm_step = {dm_trials_float}，'
                          '必须是整数')

    if dm_trials > 0 and dm_trials % DM_TRIALS_PER_CHUNK != 0:
        lower_chunks = dm_trials // DM_TRIALS_PER_CHUNK
        upper_chunks = lower_chunks + 1
        suggested_range = []
        if lower_chunks > 0:
            suggested_range.append(
                lower_chunks * DM_TRIALS_PER_CHUNK * process_config.dm_step
            )
        suggested_range.append(
            upper_chunks * DM_TRIALS_PER_CHUNK * process_config.dm_step
        )
        suggestions = ' 或 '.join(f'{value:g}' for value in suggested_range)
        errors.append(f'DM trial 总数 {dm_trials} 必须是 {DM_TRIALS_PER_CHUNK} 的整数倍；'
                      f'保持 dm_step 不变时，可将 dm_range 改为 {suggestions}')
    if block_size % 512 != 0:
        errors.append(f'block_size {block_size} 必须是 512 的整数倍')
    if file_leng % down_time_rate != 0:
        errors.append(f'file_leng {file_leng} 必须能被 down_time_rate {down_time_rate} 整除')
    if freq_reso % down_freq_rate != 0:
        errors.append(f'freq_reso {freq_reso} 必须能被 down_freq_rate {down_freq_rate} 整除')
    if not 0.0 < process_config.rfi_level_ratio < 1.0:
        errors.append(f'rfi_level_ratio {process_config.rfi_level_ratio} 必须在 (0, 1) 内')
    if process_config.rfi_cv_ratio <= 1.0:
        errors.append(f'rfi_cv_ratio {process_config.rfi_cv_ratio} 必须大于 1')
    if process_config.rfi_zero_dm_sigma <= 0.0:
        errors.append(f'rfi_zero_dm_sigma {process_config.rfi_zero_dm_sigma} 必须大于 0')
    if not 0.0 < process_config.rfi_zero_dm_occupancy < 1.0:
        errors.append('rfi_zero_dm_occupancy '
                      f'{process_config.rfi_zero_dm_occupancy} 必须在 (0, 1) 内')
    if down_file_leng % block_size != 0:
        errors.append(f'down_file_leng {down_file_leng} 必须能被 block_size {block_size} 整除'
                      f'（否则每个 block 尾部 {down_file_leng % block_size} 个样本不会被检索）')
    if errors:
        raise ValueError('process_fits_list 参数自检失败:\n  - ' + '\n  - '.join(errors))
    return dm_trials, dm_trials // DM_TRIALS_PER_CHUNK


# ---------------------------------------------------------------------------
# 4. dedispersion cache 构建
# ---------------------------------------------------------------------------

def _build_dedispersion_cache(height, dm_time_width, data_width, process_config):
    """预计算细频率通道时移矩阵并分配常驻 GPU 缓冲。

    Args:
        height        : DM trial 数（dm_range / dm_step）。
        dm_time_width : 后降采样输出时间维 = block_file * file_leng / down_time_rate。
        data_width    : 原始输入时间维 = comb_file * file_leng，包含色散读取提前量。
    """
    freq_inv2     = 1.0 / (freq.astype(np.float32) * freq.astype(np.float32))
    freq_max_inv2 = float(freq_inv2[-1])

    # 全部细通道都进入 cache；不再需要旧版的有效通道索引映射。
    # 当前数据块要排除的通道/零 DM 时间 bin 全部由 _find_rfi_masks_gpu 动态决定。
    # FS1 / FS2 严格按原始频带中点二分；4096-channel、1000--1500 MHz 数据即
    # FS1=[1000, 1250) MHz，FS2=[1250, 1500) MHz。
    split_pos = freq_reso // 2 - 1

    dm_values  = (np.arange(height, dtype=np.float32) * process_config.dm_step + process_config.dm_min)
    delta_inv2 = (freq_inv2 - freq_max_inv2).astype(np.float32)
    # shift 必须保留原始采样精度；时间平均在 GPU kernel 内、完成频率对齐之后执行。
    factor     = np.float32(4.15e3 / time_reso)
    shift_base = np.trunc((dm_values[:, None] * factor) * delta_inv2[None, :]).astype(np.int32)

    return DedispersionCache(
        split_pos      = split_pos,
        shift_base_gpu = cp.asarray(shift_base),
        dm_time_gpu    = cp.zeros((3, height, dm_time_width), dtype=cp.float32),
        raw_data_gpu   = cp.empty((data_width, freq_reso), dtype=cp.uint16),
        data_cumsum_gpu = cp.empty((freq_reso, data_width), dtype=cp.float32),
        dm_time_width  = dm_time_width,
        data_width     = data_width,
    )


def _find_rfi_masks_gpu(raw_data_gpu, process_config):
    """找出不应参与 DM-time 求和的 coarse channels 和零 DM 时间 bin。

    数据已经在 GPU；融合 kernel 直接生成最终 8x8 分辨率的轻量时间序列。
    每个 coarse channel 用两项统计量判定：

    * 平均电平低于有效频段中位数的 ``rfi_level_ratio``：带边/坏通道；
    * 时间变异系数高于有效通道中位数的 ``rfi_cv_ratio``：脉冲型 RFI。

    频率 mask 长度为 512，被判坏的一项对应 8 个原始细通道。时间 mask 在
    ``time/down_time_rate`` 分辨率上，用频率归一化后的全频平均显著性与通道
    占用率联合判定，并向两侧各扩一个 bin。没有任何额外固定频段/时间掩码。
    """
    analysis_time_size = int(raw_data_gpu.shape[0])
    analysis_time_size -= analysis_time_size % down_time_rate
    coarse_count = freq_reso // down_freq_rate
    output_time_size = analysis_time_size // down_time_rate
    bad_mask_gpu = cp.zeros(coarse_count, dtype=cp.bool_)
    bad_time_gpu = cp.zeros(output_time_size, dtype=cp.bool_)
    if not process_config.rfi_filter or analysis_time_size == 0:
        return bad_mask_gpu, bad_time_gpu

    coarse_time_gpu = cp.empty((output_time_size, coarse_count), dtype=cp.float32)
    total = output_time_size * coarse_count
    _COARSE_TIME_KERNEL(
        ((total + 255) // 256,), (256,),
        (raw_data_gpu, coarse_time_gpu, output_time_size, freq_reso,
         down_time_rate, down_freq_rate),
    )
    channel_mean_gpu = coarse_time_gpu.mean(axis=0, dtype=cp.float32)
    channel_cv_gpu = coarse_time_gpu.std(axis=0, dtype=cp.float32) / (
        channel_mean_gpu + np.float32(1.0e-6)
    )
    reference_mean = cp.maximum(cp.median(channel_mean_gpu), np.float32(1.0e-6))
    cv_limit = cp.maximum(
        cp.median(channel_cv_gpu) * np.float32(process_config.rfi_cv_ratio),
        np.float32(1.0e-3),
    )
    bad_mask_gpu[:] = (
        (channel_mean_gpu < reference_mean * np.float32(process_config.rfi_level_ratio))
        | (channel_cv_gpu > cv_limit)
    )

    if process_config.rfi_zero_dm_filter:
        good_count = coarse_count - int(cp.count_nonzero(bad_mask_gpu).get())
        if good_count > 0:
            coarse_time_gpu /= channel_mean_gpu[None, :] + np.float32(1.0e-6)
            coarse_time_gpu[:, bad_mask_gpu] = np.float32(0.0)
            zero_dm_gpu = coarse_time_gpu.sum(axis=1, dtype=cp.float32) / np.float32(good_count)
            reference_cv = cp.median(channel_cv_gpu[~bad_mask_gpu])
            occupancy_gpu = cp.count_nonzero(
                coarse_time_gpu > np.float32(1.0) + np.float32(3.0) * reference_cv,
                axis=1,
            ) / np.float32(good_count)
            zero_dm_center = cp.median(zero_dm_gpu)
            zero_dm_sigma = cp.maximum(
                np.float32(1.4826) * cp.median(cp.abs(zero_dm_gpu - zero_dm_center)),
                np.float32(1.0e-4),
            )
            raw_bad_time_gpu = (
                (zero_dm_gpu > zero_dm_center
                 + np.float32(process_config.rfi_zero_dm_sigma) * zero_dm_sigma)
                & (occupancy_gpu >= np.float32(process_config.rfi_zero_dm_occupancy))
            )
            bad_time_gpu = raw_bad_time_gpu.copy()
            bad_time_gpu[1:] |= raw_bad_time_gpu[:-1]
            bad_time_gpu[:-1] |= raw_bad_time_gpu[1:]

    return bad_mask_gpu, bad_time_gpu


def d_dm_time_g(data, dd_cache, process_config):
    """在原始时间/频率分辨率消色散后融合下采样，返回常驻显存的结果。

    Args:
        data          : 单个 ``(time, freq_reso)`` uint16 numpy，或此类数组的序列；
                        均未做时间/频率下采样。
    """
    data_parts = (data,) if isinstance(data, np.ndarray) else tuple(data)
    input_time = sum(part.shape[0] for part in data_parts)
    height = int(dd_cache.dm_time_gpu.shape[1])
    dm_time_width = dd_cache.dm_time_width
    if input_time != dd_cache.data_width:
        raise ValueError(
            f'data input_time {input_time} must equal cache.data_width {dd_cache.data_width}'
        )

    if any(part.ndim != 2 or part.shape[1] != freq_reso for part in data_parts):
        raise ValueError(f'each data part must have shape (time, {freq_reso})')

    # 直接逐文件写入 GPU 对应切片，避免 CPU 端每轮重新拼接约 3 GiB 数据。
    offset = 0
    for part in data_parts:
        stop = offset + part.shape[0]
        dd_cache.raw_data_gpu[offset:stop].set(np.ascontiguousarray(part, dtype=np.uint16))
        offset = stop

    # cache 覆盖全部细通道；当前读取窗口要排除的频率完全由动态 RFI mask 决定。
    # RFI 判定使用包含色散读取提前量的完整窗口，避免未来文件中的 RFI 污染高 DM 输出。
    rfi_bad_coarse_gpu, rfi_bad_time_gpu = _find_rfi_masks_gpu(
        dd_cache.raw_data_gpu, process_config,
    )

    # 分母仍按旧 core 的 512 个 coarse channel 计算，但不平均任何待消色散样本；
    # tiled kernel 直接把细通道归一化并转成 (freq, time) 合并访问布局。
    coarse_mean_gpu = cp.mean(
        dd_cache.raw_data_gpu.reshape(input_time, freq_reso // down_freq_rate, down_freq_rate),
        axis=(0, 2), dtype=cp.float32,
    )
    _PREPARE_KERNEL(
        ((freq_reso + 31) // 32, (input_time + 31) // 32), (32, 8),
        (dd_cache.raw_data_gpu, coarse_mean_gpu,
         rfi_bad_coarse_gpu, rfi_bad_time_gpu,
         dd_cache.data_cumsum_gpu, input_time, freq_reso,
         down_time_rate, down_freq_rate),
    )
    del coarse_mean_gpu

    # raw_data_gpu 保持完整，后续 binary cutout 不受 RFI mask 影响。
    if process_config.rfi_filter and process_config.verbose:
        bad_coarse = cp.asnumpy(cp.flatnonzero(rfi_bad_coarse_gpu))
        ranges = []
        if bad_coarse.size:
            start = previous = int(bad_coarse[0])
            for coarse_index in bad_coarse[1:]:
                coarse_index = int(coarse_index)
                if coarse_index != previous + 1:
                    ranges.append((start, previous))
                    start = coarse_index
                previous = coarse_index
            ranges.append((start, previous))
        labels = [
            (f'{freq[start * down_freq_rate]:.2f}-'
             f'{freq[min((stop + 1) * down_freq_rate - 1, freq_reso - 1)]:.2f} MHz')
            for start, stop in ranges
        ]
        print(f'[RFI] dynamic mask {bad_coarse.size}/'
              f'{freq_reso // down_freq_rate} coarse channels: '
              f'{", ".join(labels) if labels else "none"}')
        bad_time_count = int(cp.count_nonzero(rfi_bad_time_gpu).get())
        print(f'[RFI] zero-DM time mask {bad_time_count}/{rfi_bad_time_gpu.size} bins '
              f'({bad_time_count * time_reso * down_time_rate * 1e3:.3f} ms)')

    # 原位前缀和避免再分配一份约 6 GB 的细通道数据。
    cp.cumsum(dd_cache.data_cumsum_gpu, axis=1, out=dd_cache.data_cumsum_gpu)

    BLOCK_TIME, BLOCK_DM = 32, 32
    nthreads = (BLOCK_TIME, BLOCK_DM)
    nblocks  = ((dm_time_width + BLOCK_TIME - 1) // BLOCK_TIME,
                (height        + BLOCK_DM  - 1) // BLOCK_DM)
    _DE_DISP_KERNEL(
        nblocks, nthreads,
        (dd_cache.dm_time_gpu, dd_cache.data_cumsum_gpu, dd_cache.shift_base_gpu,
         freq_reso, dd_cache.split_pos, height, dm_time_width, input_time,
         down_time_rate, down_freq_rate),
    )
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
    img_flat = torch.clamp(img_flat, vmin, vmax)
    imin = img_flat.amin(dim=(1, 2), keepdim=True)
    imax = img_flat.amax(dim=(1, 2), keepdim=True)
    return (img_flat - imin) / (imax - imin + 1e-8)


def preprocess_data_gpu(data: torch.Tensor, exp_cut: float = 5.0) -> torch.Tensor:
    """单张 (T, F) GPU tensor 的归一化，与 binary_data.BurstDataset.preprocess 等价。

    quantile → kthvalue 加速；FP 用法上无差异。
    """
    data  = data + 1.0
    data  = data / (data.mean(dim=0, keepdim=True) + 1e-8)
    flat  = data.reshape(-1)
    total = flat.shape[0]
    k_lo  = max(1, int(round(total * exp_cut / 100.0)))
    k_hi  = max(1, int(round(total * (1.0 - exp_cut / 100.0))))
    lo    = flat.kthvalue(k_lo).values
    hi    = flat.kthvalue(k_hi).values
    data  = torch.clamp(data, lo, hi)
    return (data - data.min()) / (data.max() - data.min() + 1e-8)


def _dedisperse_cutout_postdown_gpu(raw_data_gpu, dm_value, toa_samp, block_size,
                                    shift_unit_raw_gpu):
    """复用常驻原始数据，融合完成 binary cutout 的消色散及时间/频率平均。"""
    start_down = max(0, int(toa_samp) - block_size // 2)
    start_raw = start_down * down_time_rate
    shifts_gpu = cp.trunc(float(dm_value) * shift_unit_raw_gpu).astype(cp.int32)
    burst_gpu = cp.empty((block_size, freq_reso // down_freq_rate), dtype=cp.float32)
    total = burst_gpu.size
    _CUTOUT_KERNEL(
        ((total + 255) // 256,), (256,),
        (burst_gpu, raw_data_gpu, shifts_gpu, raw_data_gpu.shape[0], freq_reso,
         start_raw, block_size, down_time_rate, down_freq_rate,
         np.float32(1.0 if nbits == 2 else 0.5)),
    )
    return preprocess_data_gpu(torch.from_dlpack(burst_gpu))


# ---------------------------------------------------------------------------
# 6. （检测解码已移出本文件；由 caller 把 decode_fn 传给 process_fits_list）
# ---------------------------------------------------------------------------
# 期望 decode_fn 签名 :
#     decode_fn(raw_pred, conf_thr=..., iou_thr=...) -> list[(scores_np, boxes_cxcywh_np)]
#   - CenterNet 由 gate.py 用 (centers → 占位 box) 的适配器对齐到该接口


# ---------------------------------------------------------------------------
# 7. 绘图工具
# ---------------------------------------------------------------------------

def _draw_detection(vis, cx, cy, w_box, h_box, color=(0, 220, 0)):
    """根据 box 尺寸自适应：``w/h > 0`` 画矩形+实心十字（真实框）；
    否则画空心十字（CenterNet 仅中心点）。"""
    if w_box > 0 and h_box > 0:
        x1, y1 = int(cx - w_box / 2), int(cy - h_box / 2)
        cv2.rectangle(vis, (x1, y1), (x1 + w_box, y1 + h_box), color, 1)
        cv2.drawMarker(vis, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 6, 1)
    else:
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
    plt.imshow(img_display, origin='lower')
    plt.xticks(np.linspace(0, 512, 5),
               np.round(np.linspace(0, block_size, 5) * time_reso * down_time_rate * 1e3, 1))
    dm_span = DM_TRIALS_PER_CHUNK * process_config.dm_step
    dm_chunk_start = process_config.dm_min + dm_span * d
    dm_chunk_stop = dm_chunk_start + dm_span
    dm_ticks = np.linspace(dm_chunk_start, dm_chunk_stop, 5)
    plt.yticks(np.linspace(0, 512, 5), [f'{value:g}' for value in dm_ticks])
    plt.xlabel('Time (ms)')
    plt.ylabel('DM (pc cm-3)')
    dm_extent = process_config.dm_range
    num_dm_chunks = int(round(dm_extent / dm_span))
    plt.title(f'DM chunk {d + 1}/{num_dm_chunks}: {dm_chunk_start:g}-{dm_chunk_stop:g}',
              fontsize=10)

    plt.subplot(122)
    plt.imshow(new_burst.T, origin='lower', cmap='mako')
    plt.yticks(np.linspace(0, 512, 6), np.round(np.linspace(freq.min(), freq.max(), 6)).astype(np.int64))
    plt.xticks(np.linspace(0, 512, 5),
               np.round(np.linspace(0, 512, 5) * time_reso * down_time_rate * 1e3, 2))
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (MHz)')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8. 主处理流程：消色散 → 检测候选 → burst 提取 → 分类 → 保存
# ---------------------------------------------------------------------------

def process_fits_list(fits_list, save_path, model, class_model, process_config, decode_fn,
                      n_search_files=None, task_identifier=None, task_info=None,
                      bad_fits_log=None):
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
        ``decode_fn(raw_pred, conf_thr, iou_thr) -> list[(scores_np, boxes_cxcywh_np)]``。
        CenterNet 由 gate.py 把 `decode_centernet_outputs`（返回 centers）适配成
        该接口（centers → 占位 box）。
    n_search_files :
        仅对前 ``n_search_files`` 个文件启动检索 block，其余文件只作消色散读取提前量
        （gate.py 按 GPU 切分单个连续观测时用，避免段间重复检索）。``None`` 表示全部检索。
    """
    if not fits_list:
        return

    fits_list  = list(fits_list)
    input_file_count = len(fits_list)
    block_size = process_config.block_size
    os.makedirs(save_path, exist_ok=True)
    manifest_path = os.path.join(save_path, process_config.manifest_name)
    input_start_mjd = get_fits_start_mjd(fits_list[0])
    fits_list.append(fits_list[-1])                                  # 末尾复制一份避免边界 IndexError
    get_obparams(fits_list[0], process_config.time_factor)

    dds_file   = _dispersion_file_span(process_config)         # 覆盖最大色散需多读的 fits 数
    block_file = int(np.ceil(down_time_rate * block_size / file_leng))  # 一个 block 对应的 fits 数
    comb_file  = block_file + dds_file                                  # 读取窗口 = block + 色散提前量
    if process_config.verbose:
        print(block_file, comb_file)

    down_file_leng = block_file * file_leng // down_time_rate      # 输出 dm_time 的时间长度
    comb_file_leng = comb_file * file_leng                         # 原始输入时间长度（含色散提前量）

    # 集中参数自检，并把物理 DM 配置换算成内部 trial 几何。
    dm_trials, num_dm_chunks = _validate_config(
        process_config, block_size, down_file_leng,
    )

    dd_cache = _build_dedispersion_cache(
        dm_trials, down_file_leng, comb_file_leng, process_config,
    )

    # 检测图几何：[dm_min, dm_min + dm_range) 每 1024 个 trial 切一块，再平均到 512 行。
    dm_span = DM_TRIALS_PER_CHUNK * process_config.dm_step       # 每张图的物理 DM 宽度
    dm_per_pixel_trials = DM_TRIALS_PER_CHUNK // 512             # 固定为每像素 2 个 trial
    dm_per_pixel = dm_per_pixel_trials * process_config.dm_step  # 每个像素的物理 DM 宽度
    time_pool = block_size // 512                                # 时间轴 pool 的倍数（block_size → 512）
    dm_max = process_config.dm_min + process_config.dm_range
    dm_last_trial = dm_max - process_config.dm_step

    # binary cutout 与第一阶段共用原始 4096-channel 时移；输出时才平均到 512 channels。
    shift_unit_raw_gpu = cp.asarray(
        (4.15e3 * (freq ** -2 - freq.max() ** -2) / time_reso).astype(np.float32)
    )

    # 相邻 block 重叠 comb_file - block_file 个 fits；sliding-window 缓存让每个文件只读一次盘
    # 每文件只在 load_func 内把前两个偏振相加；时间和频率轴都保持 FITS 原始采样。
    file_cache = {}
    load_func  = load_2bit_fits_file if nbits == 2 else load_fits_file
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetch_futures = {}

    # n_search_files 之后的文件只作消色散读取提前量，不再为其启动检索 block
    search_limit = input_file_count if n_search_files is None else min(n_search_files, input_file_count)

    for i in range(0, search_limit, block_file):
        filename = os.path.splitext(os.path.basename(fits_list[i]))[0]
        try:
            block_start_mjd = get_fits_start_mjd(fits_list[i])
        except OSError:
            block_start_mjd = input_start_mjd + i * file_leng * time_reso / 86400.0
        block_search_fits_count = min(block_file, search_limit - i)
        if process_config.verbose:
            print(filename)

        # ── 文件读取与拼接（带 sliding-window 缓存）─────────────────────────
        needed = [fits_list[i + j] for j in range(comb_file) if i + j < len(fits_list)]
        for fp in list(file_cache):
            if fp not in needed:
                del file_cache[fp]
        for fp in needed:
            if fp not in file_cache:
                try:
                    future = prefetch_futures.pop(fp, None)
                    file_cache[fp] = future.result() if future is not None else load_func(fp)
                except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                    if bad_fits_log:
                        with open(bad_fits_log, 'a', encoding='utf-8') as f:
                            f.write(f'{task_identifier}\t{fp}\t{type(exc).__name__}: {exc}\n')
                    print(f'[Bad FITS] {fp}: {exc}; using random replacement')
                    file_cache[fp] = np.random.default_rng().integers(
                        0, 4 if nbits == 2 else 512,
                        size=(file_leng, freq_reso), dtype=np.uint16,
                    )
        raw_data = [file_cache[fp] for fp in needed]

        loaded_samples = sum(part.shape[0] for part in raw_data)
        if loaded_samples < comb_file_leng:
            padding = np.random.default_rng().integers(
                0, 4 if nbits == 2 else 512,
                size=(comb_file_leng - loaded_samples, freq_reso), dtype=np.uint16,
            )
            raw_data.append(padding)

        # GPU 处理当前 block 时，单线程预读下一窗口新增的 FITS；典型 block_file=1
        # 时只提交一个文件，使稳态耗时接近 max(I/O, GPU) 而不是两者之和。
        next_i = i + block_file
        if next_i < search_limit:
            next_needed = [
                fits_list[next_i + j] for j in range(comb_file)
                if next_i + j < len(fits_list)
            ]
            for fp in next_needed:
                if fp not in file_cache and fp not in prefetch_futures:
                    prefetch_futures[fp] = prefetch_executor.submit(load_func, fp)

        # ── 原始时间/频率精度上传 → 消色散 → 融合时间/频率平均 ───────────────
        new_data = d_dm_time_g(raw_data, dd_cache, process_config)

        # ── 全 GPU 后处理 ───────────────────────────────────────────────────
        new_data_t = torch.from_dlpack(new_data)                 # 零拷贝视图 (3, dm_trials, down_file_leng)

        num_slices = down_file_leng // block_size
        if process_config.verbose:
            print(f'Processing {num_slices} slices, {num_dm_chunks} DM chunks')

        for j in range(num_slices):
            # ── 切片 → 降采样 → 归一化（全在显存）──────────────────────────
            slice_data = new_data_t[:, :, j * block_size: (j + 1) * block_size]
            # 一次 reshape+mean 完成每个 chunk 内的 DM/时间等宽平均：
            #   DM trial 轴 → (num_dm_chunks, 512, dm_per_pixel_trials)，时间轴 → (512, time_pool)
            # 等宽均值池化满足结合律，故可一次性 pool 成 num_dm_chunks 张 512x512 检测图
            img_batch      = (slice_data
                              .reshape(3, num_dm_chunks, 512, dm_per_pixel_trials, 512, time_pool)
                              .mean(dim=(3, 5)))                     # (3, num_dm_chunks, 512, 512)
            img_batch_flat = img_batch.flatten(0, 1)                  # (N, 512, 512)
            img_batch_flat = _normalize_batch_gpu(img_batch_flat)    # kthvalue 加速版

            # ── 检测推理（数据已在显存，无 H2D；单通道输入）────────────────
            det_input = img_batch_flat.unsqueeze(1)                 # (N, 1, 512, 512)
            with torch.no_grad():
                raw_pred = model(det_input)
            det_results = decode_fn(
                raw_pred, conf_thr=process_config.det_prob, iou_thr=process_config.det_iou,
            )

            # ── 候选收集 + 融合 GPU binary cutout ──────────────────────────
            candidates = []
            for idx, (pred_scores, pred_boxes) in enumerate(det_results):
                if pred_boxes is None:
                    continue
                d = idx % num_dm_chunks                              # 该图所在的 DM chunk

                for box_idx, (cx, cy, w_box, h_box) in enumerate(pred_boxes.astype(np.int64)):
                    score_val = float(pred_scores[box_idx]) if pred_scores is not None else 0.0
                    dm_value = (process_config.dm_min
                                + d * dm_span
                                + cy * dm_per_pixel)
                    if dm_value <= process_config.dm_threshold:
                        continue

                    toa_samp   = int(cx * time_pool + j * block_size)
                    new_burst = _dedisperse_cutout_postdown_gpu(
                        dd_cache.raw_data_gpu, dm_value, toa_samp,
                        process_config.class_block_size, shift_unit_raw_gpu,
                    )

                    candidates.append({
                        'box_idx': box_idx, 'idx': idx,
                        'cx': int(cx), 'cy': int(cy), 'w_box': int(w_box), 'h_box': int(h_box),
                        'dm_value': float(dm_value), 'score': score_val, 'toa_samp': int(toa_samp),
                        'new_burst': new_burst,                       # GPU tensor
                    })

            if not candidates:
                continue

            # ── 批量分类（全在显存，无 H2D）───────────────────────────────
            burst_batch  = torch.stack([c['new_burst'] for c in candidates], dim=0)  # (N, T, F)
            class_score_batches = []
            with torch.no_grad():
                for s in range(0, len(candidates), process_config.classifier_batch_size):
                    e = s + process_config.classifier_batch_size
                    inputs = burst_batch[s:e].unsqueeze(1)            # (B, 1, T, F)
                    class_score_batches.append(
                        torch.softmax(class_model(inputs), dim=1)[:, 1]
                    )
            class_scores = torch.cat(class_score_batches).cpu().tolist()

            # ── 保存结果（仅正样本）────────────────────────────────────────
            npy_paths = {}
            for cand, cls_score in zip(candidates, class_scores):
                if cls_score <= process_config.class_threshold:
                    continue
                k, d = divmod(cand['idx'], num_dm_chunks)            # 由 idx 还原 (通道 k, DM chunk d)

                if process_config.verbose:
                    print(f"Score: {cand['score']:.3f}, DM: {cand['dm_value']:.1f}, CLS: {cls_score:.3f}")

                npy_path = None
                if process_config.save_npy:
                    npy_path = npy_paths.get(k)
                    if npy_path is None:
                        # 同一时间片/频带只回传和保存一次，避免多个候选重复搬运约 134 MB。
                        data_slice = cp.asnumpy(
                            new_data[k, :, j * block_size: (j + 1) * block_size]
                        )
                        npy_path = f"{save_path}{filename}-TS{j:0>2d}-FS{k}.npy"
                        np.save(npy_path, data_slice.astype(np.float32, copy=False))
                        npy_paths[k] = npy_path

                jpg_path = None
                if process_config.save_plot:
                    # GPU → CPU：仅画图用，12.6 MB / 次
                    img_display = (img_batch_flat[cand['idx']].cpu().numpy() * 255).astype(np.uint8)
                    img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
                    # 真实框 → 矩形+实心十字；CenterNet 仅中心点（w=h=0）→ 空心十字
                    _draw_detection(img_display, cand['cx'], cand['cy'], cand['w_box'], cand['h_box'])
                    jpg_path = (f"{save_path}{filename}-TS{j:0>2d}-FS{k}"
                                f"-BX{cand['box_idx']}-DM{cand['dm_value']:.1f}.jpg")
                    _save_plot(img_display, cand['new_burst'].cpu().numpy(),
                               jpg_path, d, block_size, process_config)

                if process_config.save_manifest:
                    toa_sec = cand['toa_samp'] * time_reso * down_time_rate
                    signal_mjd = block_start_mjd + toa_sec / 86400.0
                    task_meta = task_info or {}
                    record = {
                        'task_identifier': task_identifier,
                        'root': task_meta.get('root'),
                        'source': task_meta.get('source'),
                        'date': task_meta.get('date'),
                        'beam': task_meta.get('beam'),
                        'input_start_mjd': input_start_mjd,
                        'block_start_mjd': block_start_mjd,
                        'block_start_fits_path': fits_list[i],
                        'block_start_fits_index': int(i),
                        'block_fits_count': int(block_search_fits_count),
                        'toa_sec': toa_sec,
                        'signal_mjd': signal_mjd,
                        'toa_sample_from_block_start': int(cand['toa_samp'] * down_time_rate),
                        'toa_sample_from_input_start': int(i * file_leng + cand['toa_samp'] * down_time_rate),
                        'time_reso': float(time_reso),
                        'down_time_rate': int(down_time_rate),
                        'file_leng': int(file_leng),
                        'slice_index': int(j),
                        'freq_slice': int(k),
                        'dm_chunk': int(d),
                        'box_idx': int(cand['box_idx']),
                        'cx': int(cand['cx']),
                        'cy': int(cand['cy']),
                        'w_box': int(cand['w_box']),
                        'h_box': int(cand['h_box']),
                        'dm': float(cand['dm_value']),
                        'det_score': float(cand['score']),
                        'class_score': float(cls_score),
                        'det_prob': float(process_config.det_prob),
                        'class_threshold': float(process_config.class_threshold),
                        'dm_threshold': float(process_config.dm_threshold),
                        'dm_min': float(process_config.dm_min),
                        'dm_range': float(process_config.dm_range),
                        'dm_max': float(dm_max),
                        'dm_step': float(process_config.dm_step),
                        'block_size': int(process_config.block_size),
                        'dm_span': float(dm_span),
                        'dm_trials': int(dm_trials),
                        'dm_chunk_trials': int(DM_TRIALS_PER_CHUNK),
                        'dm_last_trial': float(dm_last_trial),
                        'dm_chunk_start': float(process_config.dm_min + d * dm_span),
                        'dm_chunk_stop': float(process_config.dm_min + (d + 1) * dm_span),
                        'jpg_path': jpg_path,
                        'npy_path': npy_path,
                    }
                    with open(manifest_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')

        del new_data_t, new_data, raw_data

    prefetch_executor.shutdown(wait=True, cancel_futures=True)
