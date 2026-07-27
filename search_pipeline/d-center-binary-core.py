"""两阶段搜索核心（CuPy 后端）：消色散 + 中心检测 + binary 分类。

数据流（全 GPU 流水线）：
    CPU → GPU（一次，仅 ~33 MB）：raw_data_down 选取有效频率通道后上传
    GPU 常驻                    ：dm_time（~805 MB at dm_range=4096），零拷贝桥接为 torch tensor
    GPU 完成                    ：reshape / normalize / 中心检测 / burst gather / 分类
    CPU 仅用于                  ：fits 读盘、时间降采样、可视化保存（仅检测命中时）

**本文件 model-agnostic**：检测模型与解码函数全部由 caller（gate.py）传入。
``decode_fn`` 必须接受 ``(raw_pred, conf_thr, iou_thr)`` 并返回每图
``(scores_np, boxes_cxcywh_np)`` 列表（CenterNet 由 gate.py 用
``centers → 占位 box`` 的小适配器对齐到该接口）。

binary 分类器使用 :mod:`binary_model.build_binary_model`（与训练侧 ``binary_classification/`` 同步）。

性能优化点：
  1. shift_base 预计算后常驻显存
  2. data 转置成 (freq, time) 布局 → kernel 内合并访问
  3. shared memory 缓存 shift_base 行
  4. split_pos 切两段循环消除分支
  5. ``__ldg`` / ``__restrict__`` 编译器提示
  6. CuPy ↔ PyTorch 零拷贝桥接
  7. normalize 阶段用 ``torch.kthvalue``（O(n) selection）替代 ``torch.quantile``（基于 sort）
"""

import os, cv2, json
cv2.setNumThreads(4)

import numpy as np
import cupy as cp
import torch
from astropy.io import fits
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
    dm_range:              int   = 4096
    dm_scale:              float = 1.0
    dm_offset:             float = 0.0
    dm_threshold:          float = 50.0
    block_size:            int   = 8192
    dm_span:               int   = 1024
    det_prob:              float = 0.45
    section_num:           int   = 5
    time_factor:           float = 8.0
    det_iou:               float = 0.5
    class_block_size:      int   = 512
    class_threshold:       float = 0.5
    classifier_batch_size: int   = 64
    save_npy:              bool  = True
    save_plot:             bool  = True
    save_manifest:         bool  = True
    manifest_name:         str   = 'candidate_manifest.jsonl'
    verbose:               bool  = False


@dataclass
class DedispersionCache:
    """预计算 + 常驻显存的消色散缓存（在多个 block_file 间复用）。

    输入 data 比输出 dm_time 宽：dm_time_width 是 block 输出长度，data_width 包含
    色散延迟所需的「读取提前量」（约为 comb_file * file_leng / down_time_rate）。
    """
    index_array:    np.ndarray   # 有效频率通道索引（CPU 端使用，下游 raw_data 选通道）
    index_len:      int
    split_pos:      int          # index 数组中点，把累加切成「低频半 / 高频半」
    shift_base_gpu: cp.ndarray   # (dm_range, index_len)        int32
    dm_time_gpu:    cp.ndarray   # (3, dm_range, dm_time_width) float32
    data_gpu:       cp.ndarray   # (index_len, data_width)      float32（转置存储）
    dm_time_width:  int          # 输出时间维（block_file 长度）
    data_width:     int          # 输入时间维（comb_file 长度，含色散提前量）


# ---------------------------------------------------------------------------
# 2. CuPy RawKernel 消色散
# ---------------------------------------------------------------------------
#
# 关键优化（不再改动；任何调整都需要重新基准）：
#   1. data 转置成 (freq, time) → 同 warp 内 shift 连续 → 完美合并访问
#   2. block 维度 (BLOCK_TIME, BLOCK_DM) = (128, 8)，threadIdx.x = 时间维（快变维）
#   3. shift_base 整块加载进 shared memory → 广播读取无 bank conflict
#   4. split_pos 切两段循环，消除 hot loop 内分支
#   5. (unsigned)shift < (unsigned)time_size 单次比较合并边界判定
#   6. __ldg / __restrict__ 编译器提示
# ---------------------------------------------------------------------------

# Kernel source is kept ASCII-only — some NVRTC builds choke on non-ASCII bytes in comments.
# 详细优化注释见上方 Python 文档与该模块顶部。
_DE_DISP_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void de_disp(float* __restrict__ dm_time,
             const float* __restrict__ data,
             const int*   __restrict__ shift_base,
             int index_len, int split_pos, int dm_size,
             int dm_time_size,
             int data_time_size) {

    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ int s_shift[];

    int tid     = threadIdx.x + threadIdx.y * blockDim.x;
    int block_n = blockDim.x  * blockDim.y;
    int dm_base = blockIdx.y  * blockDim.y;

    for (int k = tid; k < (int)(blockDim.y * index_len); k += block_n) {
        int dm_l = k / index_len;
        int f_i  = k % index_len;
        int dm_g = dm_base + dm_l;
        s_shift[k] = (dm_g < dm_size) ? shift_base[dm_g * index_len + f_i] : 0;
    }
    __syncthreads();

    if (x >= dm_size || y >= dm_time_size) return;

    float td_i = 0.0f, td_mid = 0.0f;
    int   dm_l   = threadIdx.y;
    int   base_s = dm_l * index_len;

    #pragma unroll 4
    for (int idx = 0; idx <= split_pos; idx++) {
        int shift = s_shift[base_s + idx] + y;
        if ((unsigned)shift < (unsigned)data_time_size)
            td_i += __ldg(&data[idx * data_time_size + shift]);
    }
    td_mid = td_i;

    #pragma unroll 4
    for (int idx = split_pos + 1; idx < index_len; idx++) {
        int shift = s_shift[base_s + idx] + y;
        if ((unsigned)shift < (unsigned)data_time_size)
            td_i += __ldg(&data[idx * data_time_size + shift]);
    }

    int base = x * dm_time_size + y;
    dm_time[1 * dm_size * dm_time_size + base] = td_mid;
    dm_time[2 * dm_size * dm_time_size + base] = td_i - td_mid;
    dm_time[0 * dm_size * dm_time_size + base] = td_i;
}
''', 'de_disp')


def _cp_to_torch(arr: cp.ndarray) -> torch.Tensor:
    """零拷贝：CuPy 显存数组 → PyTorch CUDA tensor（共享底层显存）。

    调用方须保证 arr（或其 base）在 tensor 使用期间保持存活。"""
    return torch.as_tensor(arr, device='cuda')


# ---------------------------------------------------------------------------
# 3. FITS 读取 + 观测参数
# ---------------------------------------------------------------------------

def _downsample_time(data):
    n_time, n_freq = data.shape
    return data.reshape(n_time // down_time_rate, down_time_rate, n_freq).mean(axis=1).astype(np.float32)


def load_fits_file(file_name, reverse_flag=False):
    """读取常规 FITS，输出已完成偏振平均和时间降采样的 (time, freq)。"""
    with fits.open(file_name, memmap=True, lazy_load_hdus=True) as f:
        h = f[1].header
        arr = f[1].data['DATA'].reshape(h['NAXIS2'] * h['NSBLK'], h['NPOL'], h['NCHAN'])
        if reverse_flag:
            arr = arr[:, :, ::-1]
        data = arr[:, 0, :].astype(np.float32)
        data += arr[:, 1, :]
        data *= 0.5
    return _downsample_time(data)


def load_2bit_fits_file(file_name):
    with fits.open(file_name) as f:
        h = f[1].header
        data = f[1].data['DATA']
    data = np.unpackbits(data.reshape(h['NAXIS2'], -1), axis=1).reshape(h['NAXIS2'], -1, 2)
    data = data[..., 0] << 1 | data[..., 1]
    data = data.reshape(h['NAXIS2'] * h['NSBLK'], h['NCHAN']).astype(np.float32, copy=False)
    return _downsample_time(data)


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
    实际 DM 范围为 dm_offset ~ dm_offset + dm_range * dm_scale。
    依赖 :func:`get_obparams` 已设置的全局量 ``freq`` / ``time_reso`` / ``file_leng``。
    """
    dm_max = process_config.dm_offset + process_config.dm_range * process_config.dm_scale
    dds_max = int(4.15 * dm_max * (freq.min() ** -2 - freq.max() ** -2) * 1e3 / time_reso)
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
    if process_config.dm_range % process_config.dm_span != 0:
        errors.append(f'dm_range {process_config.dm_range} 必须能被 dm_span '
                      f'{process_config.dm_span} 整除')
    if process_config.dm_span % 1024 != 0:
        errors.append(f'dm_span {process_config.dm_span} 必须是 1024 的整数倍')
    if block_size % 512 != 0:
        errors.append(f'block_size {block_size} 必须是 512 的整数倍')
    if file_leng % down_time_rate != 0:
        errors.append(f'file_leng {file_leng} 必须能被 down_time_rate {down_time_rate} 整除')
    if freq_reso % down_freq_rate != 0:
        errors.append(f'freq_reso {freq_reso} 必须能被 down_freq_rate {down_freq_rate} 整除')
    if down_file_leng % block_size != 0:
        errors.append(f'down_file_leng {down_file_leng} 必须能被 block_size {block_size} 整除'
                      f'（否则每个 block 尾部 {down_file_leng % block_size} 个样本不会被检索）')
    if errors:
        raise ValueError('process_fits_list 参数自检失败:\n  - ' + '\n  - '.join(errors))


# ---------------------------------------------------------------------------
# 4. dedispersion cache 构建
# ---------------------------------------------------------------------------

def _build_dedispersion_cache(height, dm_time_width, data_width, process_config):
    """预计算时移矩阵 + 分配输入/输出 GPU 缓冲。

    Args:
        height        : DM 维（dm_range）。
        dm_time_width : 输出 dm_time 的时间维 = block_file * file_leng / down_time_rate。
        data_width    : 输入 data 的时间维   = comb_file  * file_leng / down_time_rate
                        （包含色散提前量；必须 >= dm_time_width + max_dispersion_samples）。
    """
    freq_down     = np.mean(freq.reshape(freq_reso // down_freq_rate, down_freq_rate), axis=1).astype(np.float32)
    freq_inv2     = 1.0 / (freq_down * freq_down)
    freq_max_inv2 = float(freq_inv2[-1])

    index_array = np.concatenate([
        np.arange(int( 10 / 4096 * freq_reso // down_freq_rate), int( 650 / 4096 * freq_reso // down_freq_rate)),
        np.arange(int(820 / 4096 * freq_reso // down_freq_rate), int(4050 / 4096 * freq_reso // down_freq_rate)),
    ]).astype(np.int32)
    index_len = int(index_array.shape[0])
    split_pos = index_len // 2

    dm_values  = (np.arange(height, dtype=np.float32) * process_config.dm_scale + process_config.dm_offset)
    delta_inv2 = (freq_inv2[index_array] - freq_max_inv2).astype(np.float32)
    factor     = np.float32(4.15e3 / (time_reso * down_time_rate))
    shift_base = np.trunc((dm_values[:, None] * factor) * delta_inv2[None, :]).astype(np.int32)

    return DedispersionCache(
        index_array    = index_array,
        index_len      = index_len,
        split_pos      = split_pos,
        shift_base_gpu = cp.asarray(shift_base),
        dm_time_gpu    = cp.zeros((3, height, dm_time_width), dtype=cp.float32),
        data_gpu       = cp.zeros((index_len, data_width),    dtype=cp.float32),
        dm_time_width  = dm_time_width,
        data_width     = data_width,
    )


def d_dm_time_g(data, height, dm_time_width, dd_cache):
    """CuPy GPU 消色散，结果常驻显存，返回 CuPy view（不 copy_to_host）。

    Args:
        data          : ``(input_time, freq)`` numpy，``input_time = comb_file * file_leng / down_time_rate``。
        height        : DM 维。
        dm_time_width : 输出 dm_time 的时间维（block_file 长度）。
    """
    input_time = data.shape[0]
    # kernel 把 dm_time_size / data_time_size 同时用作边界判定和缓冲行步长，
    # 故二者必须与 cache 分配的宽度严格相等（不能只是 <=），否则会按错误的步长寻址。
    if dm_time_width != dd_cache.dm_time_width:
        raise ValueError(
            f'dm_time_width {dm_time_width} must equal cache.dm_time_width {dd_cache.dm_time_width}'
        )
    if input_time != dd_cache.data_width:
        raise ValueError(
            f'data input_time {input_time} must equal cache.data_width {dd_cache.data_width}'
        )

    # CPU → GPU：仅传有效频率通道（~33 MB at freq_reso=4096），转置后写入 (index_len, input_time)
    dd_cache.data_gpu[:, :input_time] = cp.asarray(data[:, dd_cache.index_array]).T

    BLOCK_TIME, BLOCK_DM = 128, 8
    nthreads = (BLOCK_TIME, BLOCK_DM)
    nblocks  = ((dm_time_width + BLOCK_TIME - 1) // BLOCK_TIME,
                (height        + BLOCK_DM  - 1) // BLOCK_DM)
    shmem    = BLOCK_DM * dd_cache.index_len * 4
    _DE_DISP_KERNEL(
        nblocks, nthreads,
        (dd_cache.dm_time_gpu, dd_cache.data_gpu, dd_cache.shift_base_gpu,
         dd_cache.index_len, dd_cache.split_pos, height, dm_time_width, input_time),
        shared_mem=shmem,
    )
    return dd_cache.dm_time_gpu[:, :, :dm_time_width]


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


# ---------------------------------------------------------------------------
# 6. （检测解码已移出本文件；由 caller 把 decode_fn 传给 process_fits_list）
# ---------------------------------------------------------------------------
# 期望 decode_fn 签名 :
#     decode_fn(raw_pred, conf_thr=..., iou_thr=...) -> list[(scores_np, boxes_cxcywh_np)]
#   - CenterNet 由 gate.py 用 (centers → 占位 box) 的适配器对齐到该接口


# ---------------------------------------------------------------------------
# 7. 绘图工具
# ---------------------------------------------------------------------------

def _draw_hollow_cross(vis, center, color, arm=9, gap=3, thickness=1):
    """中心留空的十字 + 小圆圈；用于 CenterNet 的中心点（无真实框）。"""
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.line(vis, (x - arm, y), (x - gap, y), color, thickness)
    cv2.line(vis, (x + gap, y), (x + arm, y), color, thickness)
    cv2.line(vis, (x, y - arm), (x, y - gap), color, thickness)
    cv2.line(vis, (x, y + gap), (x, y + arm), color, thickness)
    cv2.circle(vis, (x, y), gap, color, thickness)


def _draw_detection(vis, cx, cy, w_box, h_box, color=(0, 220, 0)):
    """根据 box 尺寸自适应：``w/h > 0`` 画矩形+实心十字（真实框）；
    否则画空心十字（CenterNet 仅中心点）。"""
    if w_box > 0 and h_box > 0:
        x1, y1 = int(cx - w_box / 2), int(cy - h_box / 2)
        cv2.rectangle(vis, (x1, y1), (x1 + w_box, y1 + h_box), color, 1)
        cv2.drawMarker(vis, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 6, 1)
    else:
        _draw_hollow_cross(vis, (cx, cy), color)


def _save_plot(img_display, new_burst, out_path, d, block_size, process_config):
    """左：DM-time 检测标记，右：dedispersed burst 频谱（time-freq）。"""
    plt.figure(figsize=(7.5, 3))
    plt.subplot(121)
    plt.imshow(img_display, origin='lower')
    plt.xticks(np.linspace(0, 512, 5),
               np.round(np.linspace(0, block_size, 5) * time_reso * down_time_rate * 1e3, 1))
    plt.yticks(np.linspace(0, 512, 5),
               ((np.linspace(0, process_config.dm_span, 5) + process_config.dm_span * d)
                * process_config.dm_scale + process_config.dm_offset).astype(np.int64))
    plt.xlabel('Time (ms)')
    plt.ylabel('DM (pc cm-3)')

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
    comb_file_leng = comb_file  * file_leng // down_time_rate      # 输入 data 的时间长度（含色散提前量）

    # 集中参数自检：所有整除 / 配置约束不满足则尽早 raise
    _validate_config(process_config, block_size, down_file_leng)

    dd_cache = _build_dedispersion_cache(
        process_config.dm_range, down_file_leng, comb_file_leng, process_config,
    )

    # 检测图几何：dm_range 沿 DM 轴切成 num_dm_chunks 张 512x512 检测图
    dm_span        = process_config.dm_span                    # 每张检测图覆盖的原始 DM 点数
    num_dm_chunks  = process_config.dm_range // dm_span         # DM 轴上切出的检测图数量
    dm_per_pixel   = dm_span // 512                            # 每像素 = 多少原始 DM 点（也是 DM 轴 pool 倍数）
    time_pool      = block_size // 512                         # 时间轴 pool 的倍数（block_size → 512）

    # 预计算频率相关常量，常驻显存
    dds_unit_gpu      = torch.from_numpy(
        (4.15 * (freq ** -2 - freq.max() ** -2) * 1e3 / (time_reso * down_time_rate)).astype(np.float32)
    ).to(device)                                                     # (freq_reso,)
    freq_arange_gpu   = torch.arange(freq_reso, device=device)       # (freq_reso,)
    class_time_arange = torch.arange(process_config.class_block_size, device=device)

    # 相邻 block 重叠 comb_file - block_file 个 fits；sliding-window 缓存让每个文件只读一次盘
    # 每文件已在 load_func 内完成偏振平均和时间降采样，二分类仍使用 full-frequency 数据。
    file_cache = {}
    load_func  = load_2bit_fits_file if nbits == 2 else load_fits_file

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
                    file_cache[fp] = load_func(fp)
                except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                    if bad_fits_log:
                        with open(bad_fits_log, 'a', encoding='utf-8') as f:
                            f.write(f'{task_identifier}\t{fp}\t{type(exc).__name__}: {exc}\n')
                    print(f'[Bad FITS] {fp}: {exc}; using random replacement')
                    file_cache[fp] = np.random.default_rng().random(
                        (file_leng // down_time_rate, freq_reso), dtype=np.float32,
                    )
                    file_cache[fp] -= 0.5
                    file_cache[fp] *= np.sqrt(15.0 / down_time_rate)
                    file_cache[fp] += 1.5
        raw_data = np.concatenate([file_cache[fp] for fp in needed], axis=0)

        if raw_data.shape[0] < comb_file_leng:
            m, s     = float(np.mean(raw_data)), float(np.std(raw_data))
            padding  = np.random.rand(comb_file_leng - raw_data.shape[0], freq_reso).astype(np.float32) * s + m
            raw_data = np.concatenate([raw_data, padding], axis=0)

        # 频率降采样（CPU）；时间降采样已在 per-file cache 内完成
        raw_data_down = np.mean(
            raw_data.reshape(comb_file_leng, freq_reso // down_freq_rate, down_freq_rate), axis=2,
        ).astype(np.float32)
        raw_data_down = raw_data_down / (np.mean(raw_data_down, axis=0) + 1e-8)

        # ── 消色散：CPU→GPU ~33 MB，结果留在显存 ────────────────────────────
        new_data = d_dm_time_g(raw_data_down, process_config.dm_range, down_file_leng, dd_cache)
        del raw_data_down

        # ── 全 GPU 后处理 ───────────────────────────────────────────────────
        new_data_t = _cp_to_torch(new_data)                      # 零拷贝视图 (3, dm_range, down_file_leng)

        num_slices = down_file_leng // block_size
        if process_config.verbose:
            print(f'Processing {num_slices} slices, {num_dm_chunks} DM chunks')

        for j in range(num_slices):
            # ── 切片 → 降采样 → 归一化（全在显存）──────────────────────────
            slice_data     = new_data_t[:, :, j * block_size: (j + 1) * block_size]
            # 一次 reshape+mean 同时完成「全局 DM /2」和「chunk 内 DM/时间 pool」：
            #   DM 轴 dm_range → (num_dm_chunks, 512, dm_per_pixel)，时间轴 → (512, time_pool)
            # 等宽均值池化满足结合律，故可一次性 pool 成 num_dm_chunks 张 512x512 检测图
            img_batch      = (slice_data
                              .reshape(3, num_dm_chunks, 512, dm_per_pixel, 512, time_pool)
                              .mean(dim=(3, 5)))                     # (3, num_dm_chunks, 512, 512)
            img_batch_flat = img_batch.reshape(-1, 512, 512)         # (N, 512, 512)
            img_batch_flat = _normalize_batch_gpu(img_batch_flat)    # kthvalue 加速版

            # ── 检测推理（数据已在显存，无 H2D；单通道输入）────────────────
            det_input = img_batch_flat.unsqueeze(1).contiguous()    # (N, 1, 512, 512)
            with torch.no_grad():
                raw_pred = model(det_input)
            det_results = decode_fn(raw_pred, conf_thr=process_config.det_prob, iou_thr=process_config.det_iou)

            # ── 候选收集 + burst 提取（GPU gather）────────────────────────
            candidates = []
            for idx, (pred_scores, pred_boxes) in enumerate(det_results):
                if pred_boxes is None:
                    continue
                d = idx % num_dm_chunks                              # 该图所在的 DM chunk

                for box_idx, (cx, cy, w_box, h_box) in enumerate(pred_boxes.astype(np.int64)):
                    score_val = float(pred_scores[box_idx]) if pred_scores is not None else 0.0
                    dm_value  = cy * dm_per_pixel + d * dm_span
                    dm_value  = dm_value * process_config.dm_scale + process_config.dm_offset
                    if dm_value <= process_config.dm_threshold:
                        continue

                    toa_samp   = int(cx * time_pool + j * block_size)
                    start_samp = max(0, toa_samp - process_config.class_block_size // 2)
                    dds_vals   = (dm_value * dds_unit_gpu).int()      # (freq_reso,) GPU
                    max_shift  = int(dds_vals.max().item())
                    end_samp   = min(start_samp + max_shift + process_config.class_block_size, raw_data.shape[0])

                    # 小切片 CPU→GPU（单个候选，仅几 MB）
                    raw_slice  = torch.from_numpy(
                        raw_data[start_samp:end_samp, :].copy()
                    ).to(device, non_blocking=True)
                    burst_gpu  = torch.zeros(
                        max_shift + process_config.class_block_size, freq_reso,
                        device=device, dtype=torch.float32,
                    )
                    burst_gpu[:end_samp - start_samp, :] = raw_slice

                    # 2D gather：t_idx (freq_reso, class_block_size)
                    t_idx     = dds_vals[:, None] + class_time_arange[None, :]
                    f_idx     = freq_arange_gpu[:, None].expand(-1, process_config.class_block_size)
                    new_burst = burst_gpu[t_idx, f_idx].T              # (class_block_size, freq_reso)
                    new_burst = new_burst.reshape(
                        process_config.class_block_size, -1, down_freq_rate,
                    ).mean(dim=2)                                       # (class_block_size, freq_reso//down_freq_rate)
                    new_burst = preprocess_data_gpu(new_burst)

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
            class_scores = []
            with torch.no_grad():
                for s in range(0, len(candidates), process_config.classifier_batch_size):
                    e      = s + process_config.classifier_batch_size
                    inputs = burst_batch[s:e].unsqueeze(1)            # (B, 1, T, F)
                    probs  = torch.softmax(class_model(inputs), dim=1)[:, 1]
                    class_scores.extend(probs.cpu().tolist())

            # ── 保存结果（仅正样本）────────────────────────────────────────
            for cand, cls_score in zip(candidates, class_scores):
                if cls_score <= process_config.class_threshold:
                    continue
                k, d = divmod(cand['idx'], num_dm_chunks)            # 由 idx 还原 (通道 k, DM chunk d)

                if process_config.verbose:
                    print(f"Score: {cand['score']:.3f}, DM: {cand['dm_value']:.1f}, CLS: {cls_score:.3f}")

                npy_path = None
                if process_config.save_npy:
                    # CuPy → numpy：仅在有检测时触发，~134 MB
                    data_slice = cp.asnumpy(
                        new_data[k, :, j * block_size: (j + 1) * block_size]
                    )
                    npy_path = f"{save_path}{filename}-TS{j:0>2d}-FS{k}.npy"
                    np.save(npy_path, data_slice.astype(np.float32))

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
                        'dm_range': int(process_config.dm_range),
                        'dm_scale': float(process_config.dm_scale),
                        'dm_offset': float(process_config.dm_offset),
                        'block_size': int(process_config.block_size),
                        'dm_span': int(process_config.dm_span),
                        'jpg_path': jpg_path,
                        'npy_path': npy_path,
                    }
                    with open(manifest_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')

        del new_data_t, new_data, raw_data
