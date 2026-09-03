# fmt: off

"""训练集生成器使用的 GPU 消色散与图像归一化。

本模块只保留 ``generate_dataset.py`` 实际需要的计算：构建消色散缓存、执行
CuPy kernel，以及对 detector 图逐张归一化。观测元信息和搜索流程由调用方管理。
"""

from dataclasses import dataclass

import cupy as cp
import numpy as np
import torch


@dataclass
class DedispersionCache:
    """一次生成任务中重复使用的频率索引、DM 时移和 GPU 缓冲。"""

    index_array: np.ndarray
    index_len: int
    split_pos: int
    shift_base_gpu: cp.ndarray
    dm_time_gpu: cp.ndarray
    data_gpu: cp.ndarray
    output_width: int
    input_width: int


# Kernel source 保持 ASCII，兼容对非 ASCII 注释处理不一致的 NVRTC 版本。
_DE_DISP_KERNEL = cp.RawKernel(
    r"""
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
""",
    "de_disp",
)


def build_cache(
    frequency_mhz,
    time_reso,
    down_freq_rate,
    down_time_rate,
    dm_count,
    output_width,
    input_width,
    dm_scale,
    dm_offset,
):
    """预计算 DM 时移并分配一次生成任务所需的 GPU 缓冲。"""
    frequency_mhz = np.asarray(frequency_mhz)
    freq_reso     = int(frequency_mhz.size)
    if frequency_mhz.ndim != 1 or freq_reso == 0:
        raise ValueError("frequency_mhz must be a non-empty one-dimensional array")
    if down_freq_rate <= 0 or freq_reso % down_freq_rate != 0:
        raise ValueError(
            f"frequency channels {freq_reso} must be divisible by down_freq_rate "
            f"{down_freq_rate}"
        )
    if down_time_rate <= 0:
        raise ValueError("down_time_rate must be positive")

    freq_down = np.mean(
        frequency_mhz.reshape(freq_reso // down_freq_rate, down_freq_rate),
        axis=1,
    ).astype(np.float32)
    freq_inv2     = 1.0 / (freq_down * freq_down)
    freq_max_inv2 = float(freq_inv2[-1])

    down_freq_reso = freq_reso // down_freq_rate
    index_array = np.concatenate(
        [
            np.arange(
                int(10 / 4096 * down_freq_reso),
                int(650 / 4096 * down_freq_reso),
            ),
            np.arange(
                int(820 / 4096 * down_freq_reso),
                int(4050 / 4096 * down_freq_reso),
            ),
        ]
    ).astype(np.int32)
    index_len   = int(index_array.size)
    split_pos   = index_len // 2

    dm_values  = np.arange(dm_count, dtype=np.float32) * dm_scale + dm_offset
    delta_inv2 = (freq_inv2[index_array] - freq_max_inv2).astype(np.float32)
    factor     = np.float32(4.15e3 / (time_reso * down_time_rate))
    shift_base = np.trunc(
        (dm_values[:, None] * factor) * delta_inv2[None, :]
    ).astype(np.int32)

    return DedispersionCache(
        index_array    = index_array,
        index_len      = index_len,
        split_pos      = split_pos,
        shift_base_gpu = cp.asarray(shift_base),
        dm_time_gpu    = cp.zeros((3, dm_count, output_width), dtype=cp.float32),
        data_gpu       = cp.zeros((index_len, input_width), dtype=cp.float32),
        output_width   = int(output_width),
        input_width    = int(input_width),
    )


def dedisperse(data, cache):
    """把 ``(input_width, frequency)`` 动态谱消色散到缓存中的 DM-time 画布。"""
    input_width = int(data.shape[0])
    if input_width != cache.input_width:
        raise ValueError(
            f"data input width {input_width} must equal cache input width "
            f"{cache.input_width}"
        )

    cache.data_gpu[:, :] = cp.asarray(data[:, cache.index_array]).T

    block_time = 128
    block_dm   = 8
    dm_count   = int(cache.dm_time_gpu.shape[1])
    nthreads   = (block_time, block_dm)
    nblocks    = (
        (cache.output_width + block_time - 1) // block_time,
        (dm_count + block_dm - 1) // block_dm,
    )
    _DE_DISP_KERNEL(
        nblocks,
        nthreads,
        (
            cache.dm_time_gpu,
            cache.data_gpu,
            cache.shift_base_gpu,
            cache.index_len,
            cache.split_pos,
            dm_count,
            cache.output_width,
            cache.input_width,
        ),
        shared_mem=block_dm * cache.index_len * 4,
    )
    return cache.dm_time_gpu


def normalize_batch(images):
    """逐图做 1%/99.5% 截断和 min-max 归一化，输出范围为 ``[0, 1]``。"""
    flat  = images.reshape(images.shape[0], -1)
    total = flat.shape[1]
    k1    = max(1, int(round(total * 0.010)))
    k2    = max(1, int(round(total * 0.995)))

    vmin   = flat.kthvalue(k1, dim=1).values[:, None, None]
    vmax   = flat.kthvalue(k2, dim=1).values[:, None, None]
    images = torch.clamp(images, vmin, vmax)
    imin   = images.amin(dim=(1, 2), keepdim=True)
    imax   = images.amax(dim=(1, 2), keepdim=True)
    return (images - imin) / (imax - imin + 1e-8)

# fmt: on
