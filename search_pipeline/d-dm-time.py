# fmt: off
"""单阶段搜索：固定 DM 消色散 + 多尺度时间下采样 + binary 分类器逐块判定。

工作流：
  - CPU 读取原始整数 FITS，GPU 完成偏振合并、原始采样消色散，再下采样
  - 按多个 down_sampling_rate 把数据重塑成 ``[N, 512, 512]`` 图像
  - binary 分类器逐块打分；score >= prob 的块被保存（npy + jpg 画图）

模型加载来自 :mod:`binary_model`（与 ``binary_classification/binary_model.py`` 同步）。
checkpoint 可以是纯 ``state_dict`` 也可以是完整的 ``{model_state_dict: ...}`` dict——两种都兼容。

Usage:
  CUDA_VISIBLE_DEVICES=0 python d-dm-time.py 0
  # 使用前编辑脚本底部 DM、section_num、data_path、save_base 和 beam_filter。
"""

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from astropy.io import fits
from matplotlib import gridspec

from binary_model import build_binary_model

plt.style.use("default")
sns.set_color_codes()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device("cuda")  # 本版本要求 CUDA，读取后的计算全在 GPU 上。


@dataclass
class ProcessConfig:
    """算法 / 运行时超参数。"""

    DM: float                             = 273.5  # 用来做色散修正的 DM 值（pc cm^-3）
    prob: float                           = 0.6    # 分类器判定阈值
    block_size: int                        = 512    # 每张 512×512 图对应的时间样本数（降采样后）
    section_num: int                       = 32     # 逻辑 section 总数
    down_sampling_rate_list: Sequence[int] = (8,)   # 原始分辨率消色散后再下采样
    classifier_batch_size: int             = 16     # 推理和预处理的批大小，控制峰值显存
    fits_prefetch_workers: int              = 2      # CPU 读取线程及提前量；0 为同步读取。


# ---------------------------------------------------------------------------
# 1. 路径聚合与 section 分配
# ---------------------------------------------------------------------------


def organize_file_lists(data_path: str, beam_filter: Optional[str] = None) -> List[Tuple[str, List[str], dict]]:
    """按 source/date/beam 聚合 fits 文件列表。

    Returns
    -------
    List of ``(identifier, files_list, path_info)``；``path_info`` 是
    ``{'source': str, 'date': str, 'beam': str}``。
    """
    result   = []
    path     = Path(data_path)
    all_fits = sorted([
        f.name for f in path.iterdir()
        if f.is_file() and f.suffix == ".fits" and not f.name.startswith(".")
        and "_N_" not in f.name and "_W_" not in f.name and "_F_" not in f.name
    ])

    if all_fits:
        # 按文件名里的 -Mxx_ 提取 beam；不含该标识的文件跳过，避免 .group(1) 对 None 崩溃
        beam_match = {f: re.search(r"-(M\d{2})_", f) for f in all_fits}
        skipped    = [f for f, m in beam_match.items() if m is None]
        if skipped:
            print(f"[organize] {path}: 跳过 {len(skipped)} 个无 -Mxx_ beam 标识的 fits")
        all_fits = [f for f in all_fits if beam_match[f] is not None]
        beams    = np.unique([beam_match[f].group(1) for f in all_fits]).tolist()
        if beam_filter and beam_filter != "all":
            beams = [beam_filter] if beam_filter in beams else []

        # 从 fits 往前数：当前目录是 date，父目录是 source
        date_name   = path.name
        source_name = path.parent.name

        for beam in beams:
            beam_files = sorted([str(path / f) for f in all_fits if beam in f])
            if beam_files:
                info       = {"source": source_name, "date": date_name, "beam": beam}
                identifier = f"{source_name}_{date_name}_{beam}"
                result.append((identifier, beam_files, info))
    else:
        # 递归子目录
        subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
        for subdir in subdirs:
            result.extend(organize_file_lists(str(subdir), beam_filter))

    return result


def organize_file_lists_from_roots(
    data_paths: List[str], beam_filter: Optional[str] = None
) -> List[Tuple[str, List[str], dict]]:
    """汇总多个数据根目录，保留 root 元数据用于区分任务标识。"""
    result = []
    for data_path in data_paths:
        root_label = Path(data_path).name
        for identifier, files, info in organize_file_lists(data_path, beam_filter):
            info = {**info, "root": root_label}
            result.append((f"{root_label}_{identifier}", files, info))
    return result


def distribute_file_lists(file_lists, section_num, section, config):
    """把所有 (identifier, files, info) 均匀分配给指定 section。

    单个连续观测按 section 切分时，给每段尾部补 ``dds_file`` 个文件作为消色散读取提前量
    （与下一段重叠，只读不检索），避免段间接缝处色散修正失真；``info['n_search_files']``
    记录本段实际需检索的文件数（不含重叠）。
    """
    if section_num <= 0:
        raise ValueError(f"section_num={section_num} 必须大于 0")
    if section < 0 or section >= section_num:
        raise ValueError(f"section={section} 必须位于 [0, {section_num})")
    if len(file_lists) == 1:
        identifier, fits_list, info = file_lists[0]
        section_size = (len(fits_list) + section_num - 1) // section_num
        start_idx    = section * section_size
        end_idx      = (section + 1) * section_size
        n_search     = len(fits_list[start_idx:end_idx])  # 本段实际要检索的文件数
        if n_search == 0:
            return []
        # 一次读取观测首文件参数，计算色散重叠量并保留标称长度供 TOA 换算。
        with fits.open(fits_list[0], memmap=True, lazy_load_hdus=True) as f:
            time_reso      = f[1].header["TBIN"]
            nominal_length = f[1].header["NAXIS2"] * f[1].header["NSBLK"]
            freq           = f[1].data["DAT_FREQ"][0, :].astype(np.float64)
        if freq[0] > freq[-1]:
            freq = freq[::-1]
        dds_max = int(4.15 * config.DM * (freq.min()**-2 - freq.max()**-2) * 1e3 / time_reso)
        overlap = int(np.ceil(dds_max / nominal_length))

        sub_fits = fits_list[start_idx:end_idx + overlap]  # 末尾重叠文件只读不检索。
        info     = {**info, "n_search_files": n_search, "reference_file_leng": nominal_length}
        return [(f"{identifier}_section{section}", sub_fits, info)]

    return [
        (identifier, fits_list, info)
        for idx, (identifier, fits_list, info) in enumerate(file_lists)
        if idx % section_num == section
    ]


# ---------------------------------------------------------------------------
# 2. 模型加载
# ---------------------------------------------------------------------------


def load_binary_classifier(
    ckpt_path,
    model_type  = "ConvNeXtNet",
    model_name  = "convnext_tiny",
    num_classes = 2,
    dropout     = 0.5,
):
    """构建模型 + 加载权重；兼容纯 state_dict 与完整 checkpoint dict 两种保存格式。

    ``weights_only`` 仅新 PyTorch 支持，老版本 TypeError 时退回不带该 kwarg 的调用。
    """
    model = build_binary_model(
        model_type  = model_type,
        model_name  = model_name,
        num_classes = num_classes,
        pretrained  = False,
        dropout     = dropout,
    ).to(device)

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval()


# ---------------------------------------------------------------------------
# 3. FITS 读取与 GPU 偏振合并
# ---------------------------------------------------------------------------


def read_fits_data(file_name, nbits):
    """后台线程只读取、打包整数；返回的主机数组供主线程上传。"""
    with fits.open(file_name, memmap=True, lazy_load_hdus=True) as f:
        h = f[1].header
        if h["NBITS"] != nbits:
            raise ValueError(f"NBITS 与读取方式不匹配：{file_name}")
        if nbits == 8:
            if h["NPOL"] < 2:
                raise ValueError("常规读取要求 NPOL>=2")
            # 保留 FITS 行维度；先选偏振再打包，省掉整份四偏振 CPU 副本。
            arr = f[1].data["DATA"].reshape(h["NAXIS2"], h["NSBLK"], h["NPOL"], h["NCHAN"])
            data = np.ascontiguousarray(arr[:, :, :2, :]).reshape(-1, 2, h["NCHAN"])
        elif nbits == 2:
            if h["NPOL"] != 1:
                raise ValueError("2-bit 读取要求 NPOL=1")
            data = np.ascontiguousarray(f[1].data["DATA"].reshape(h["NAXIS2"], -1))
        else:
            raise ValueError("NBITS 支持 2 或 8")
        return data, h["NCHAN"], h["NSBLK"] * h["NCHAN"]


# 每个线程直接输出偏振和；频率翻转同时完成，减少整文件 uint16 中间数组。
_POLARIZATION_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void polarization(const unsigned char* raw, unsigned short* out,
                  long long count, int nchan, int reverse) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    long long t = index / nchan;
    int f = index % nchan;
    if (reverse) f = nchan - 1 - f;
    long long offset = t * (2LL * nchan) + f;
    out[index] = (unsigned short)raw[offset] + raw[offset + nchan];
}
""", "polarization")


def load_fits_file(file_name, reverse_flag=False, host_data=None):
    """CPU 读取并打包前两个偏振；原始整数上传后在 GPU 合并。"""
    host_data = read_fits_data(file_name, 8) if host_data is None else host_data
    raw       = cp.asarray(host_data[0])

    # uint16 偏振和占用 float32 的一半；kernel 中除以 2 恢复偏振平均。
    data = cp.empty((raw.shape[0], raw.shape[2]), dtype=cp.uint16)
    _POLARIZATION_KERNEL(
        ((data.size + 255) // 256,), (256,),
        (raw, data, np.int64(data.size), np.int32(raw.shape[2]), np.int32(reverse_flag)),
    )
    return data


def load_2bit_fits_file(file_name, reverse_flag=False, host_data=None):
    """GPU 解包每字节四个 2-bit 样本，保存为双倍强度 uint16。"""
    host_data           = read_fits_data(file_name, 2) if host_data is None else host_data
    packed, nchan, count = host_data
    raw                 = cp.asarray(packed)

    shifts = cp.asarray([6, 4, 2, 0], dtype=cp.uint8)
    # 每行末尾可能有字节对齐的填充位，只保留有效样本。
    data = ((raw[..., None] >> shifts) & 3).reshape(raw.shape[0], -1)[:, :count]
    data = cp.ascontiguousarray(data.reshape(-1, nchan), dtype=cp.uint16)
    data *= 2
    if reverse_flag:
        data = cp.ascontiguousarray(data[:, ::-1])
    return data


# ---------------------------------------------------------------------------
# 4. GPU 原始采样消色散
# ---------------------------------------------------------------------------

# 一个线程输出一个粗频率/时间像素，先按细通道原始时移取样，再平均时间和频率。
# 融合 kernel 避免构建完整 float32 消色散矩阵及二维索引矩阵。
_DEDISPERSE_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void dedisperse(const unsigned short* raw, const long long* shifts, float* out,
                long long output_samples, int nchan, int freq_factor, int time_factor) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_samples * 512) return;
    long long t = index / 512;
    int f0 = (index % 512) * freq_factor;
    float sum = 0.0f;
    for (int dt = 0; dt < time_factor; ++dt) {
        for (int df = 0; df < freq_factor; ++df) {
            int f = f0 + df;
            long long sample = t * time_factor + dt + shifts[f];
            sum += raw[sample * nchan + f];
        }
    }
    out[index] = sum / (2.0f * time_factor * freq_factor);
}
""", "dedisperse")


# ---------------------------------------------------------------------------
# 5. GPU 批量归一化
# ---------------------------------------------------------------------------


def preprocess_data(data, exp_cut=5):
    """GPU 批量预处理：(N,T,F) 按图除时间基线、5/95% 截断、min-max。"""
    # 每张图按时间轴计算频率通道基线，保持训练时的预处理定义。
    data.add_(1)
    data.div_(data.mean(dim=1, keepdim=True).clamp_min_(1e-8))
    flat = data.flatten(1)

    # 一次批量排序共用两个分位点，避免四次 kthvalue 扫描；保留线性插值定义。
    ordered = flat.sort(dim=1).values
    limits = []
    for quantile in (exp_cut / 100, 1 - exp_cut / 100):
        position = (flat.shape[1] - 1) * quantile
        lower    = int(position)
        lo       = ordered[:, lower]
        hi       = ordered[:, min(lower + 1, flat.shape[1] - 1)]
        limits.append((lo + (hi - lo) * (position - lower))[:, None, None])
    del ordered

    # 截断后逐图映射到 [0, 1]；常量图通过分母 epsilon 得到全零结果。
    data.clamp_(limits[0], limits[1])
    minimum = data.amin(dim=(1, 2), keepdim=True)
    maximum = data.amax(dim=(1, 2), keepdim=True)
    data.sub_(minimum).div_(maximum - minimum + 1e-8)
    return data


# ---------------------------------------------------------------------------
# 6. 候选可视化与保存
# ---------------------------------------------------------------------------


def plot_burst(
    data, filename, block, time_reso, fits_number, block_size,
    save_path, file_leng, down_sampling_rate, freq, file_start_seconds=None,
):
    """画检测到的脉冲：上半 profile，下半 2D 频谱。"""
    w, h = data.shape
    profile    = np.mean(data, axis=1)
    peak_idx   = int(np.argmax(profile))
    time_start = ((fits_number - 1) * file_leng // down_sampling_rate + block * block_size) * time_reso
    # 短文件使用观测标称文件长度得到绝对起点，避免末文件长度改变 TOA。
    if file_start_seconds is not None:
        time_start = file_start_seconds + block * block_size * time_reso
    peak_time = time_start + peak_idx * time_reso

    plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(4, 1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(gs[0, 0])
    plt.plot(profile, color="royalblue", alpha=0.8, lw=1)
    plt.scatter(peak_idx, profile[peak_idx], color="red", s=100, marker="x")
    plt.xlim(0, w)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(gs[1:, 0])
    plt.imshow(data.T, origin="lower", cmap="mako", aspect="auto")
    plt.scatter(peak_idx, 0, color="red", s=100, marker="x")
    plt.yticks(np.linspace(0, h, 6), np.int64(np.linspace(freq.min(), freq.max(), 6)))
    plt.xticks(
        np.linspace(0, w, 6),
        np.round(time_start + np.arange(6) * time_reso * block_size / 5, 2),
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    plt.savefig(
        "{}{}-{:0>4d}-{}.jpg".format(save_path, filename, block, peak_time),
        format      = "jpg",
        dpi         = 300,
        bbox_inches = "tight",
    )
    plt.close()


# ---------------------------------------------------------------------------
# 7. 主处理函数
# ---------------------------------------------------------------------------


def process_fits_list(
    fits_list, model, config: ProcessConfig, save_path,
    n_search_files=None, reference_file_leng=None,
):
    """原始分辨率 GPU 消色散 → GPU 下采样/预处理/分类；命中后下载图像保存。"""
    if not fits_list:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("本脚本要求 CUDA GPU")

    # 校验倍率和窗口规格；n_search_files 将 section 的读取提前量排除在检索范围外。
    rates = list(config.down_sampling_rate_list)
    if config.block_size != 512 or not rates or any(int(r) != r or r <= 0 for r in rates):
        raise ValueError("block_size 必须为 512，下采样倍率必须为正整数")
    if config.DM < 0 or config.classifier_batch_size <= 0:
        raise ValueError("DM 必须非负，classifier_batch_size 必须为正整数")
    if isinstance(config.fits_prefetch_workers, bool) or not isinstance(config.fits_prefetch_workers, (int, np.integer)) or config.fits_prefetch_workers < 0:
        raise ValueError("fits_prefetch_workers 必须为非负整数")
    rates       = [int(r) for r in rates]
    common_rate = int(np.gcd.reduce(rates))  # 公共时间平均倍率，保证各尺度可由整数倍平均得到。
    if n_search_files is not None and not 0 <= n_search_files <= len(fits_list):
        raise ValueError("n_search_files 超出文件列表范围")
    search_limit = len(fits_list) if n_search_files is None else n_search_files
    os.makedirs(save_path, exist_ok=True)

    # 读取真实长度；最后一个 FITS 的 NAXIS2 可以不同，标称长度仅用于 TOA。
    lengths = []
    with fits.open(fits_list[0], memmap=True, lazy_load_hdus=True) as f:
        header    = f[1].header
        time_reso = header["TBIN"]
        freq_reso = header["NCHAN"]
        nbits     = header["NBITS"]
        freq      = f[1].data["DAT_FREQ"][0].astype(np.float64)

    for path in fits_list:
        with fits.open(path, memmap=True, lazy_load_hdus=True) as f:
            h = f[1].header
            if h["TBIN"] != time_reso or h["NCHAN"] != freq_reso or h["NBITS"] != nbits:
                raise ValueError(f"观测参数变化：{path}")
            lengths.append(h["NAXIS2"] * h["NSBLK"])
    if freq_reso % 512 or freq_reso < 512 or nbits not in (2, 8):
        raise ValueError("NCHAN 必须为 512 的倍数，NBITS 支持 2 或 8")
    if any(length <= 0 for length in lengths) or time_reso <= 0:
        raise ValueError("文件长度和 TBIN 必须为正")

    # 统一频率方向，并预计算每个细通道的原始采样时移。
    file_leng    = reference_file_leng or lengths[0]
    reverse_flag = freq[0] > freq[-1]
    if reverse_flag:
        freq = freq[::-1]
    if np.any(np.diff(freq) <= 0) or np.any(freq <= 0):
        raise ValueError("频率必须为正且单调")
    load_func = load_2bit_fits_file if nbits == 2 else load_fits_file
    shifts    = (4.15 * config.DM * (freq**-2 - freq.max()**-2) * 1e3 / time_reso).astype(np.int64)
    max_shift = int(shifts.max())
    print(f"GPU 原始分辨率消色散：DM={config.DM}, 最大时移={max_shift}, 后下采样={rates}")

    file_cache = {}
    prefetch_futures = {}
    # 共用 PyTorch 当前 CUDA stream，保证 CuPy kernel 与零拷贝 tensor 消费顺序。
    with (
        ThreadPoolExecutor(max_workers=max(1, int(config.fits_prefetch_workers))) as prefetch_executor,
        cp.cuda.Device(device.index or torch.cuda.current_device()),
        cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream),
    ):
        shifts_gpu = cp.asarray(shifts)
        i          = 0
        while i < search_limit:
            # 聚合真实检索文件，至少覆盖最大倍率的一张 512 点图像。
            end          = i
            real_samples = 0
            while end < search_limit and (end == i or real_samples < max(rates) * 512):
                real_samples += lengths[end]
                end += 1

            # 每个倍率均向上补齐 512 点窗口，最大长度覆盖全部尺度及色散余量。
            target_samples = max(((real_samples + r * 512 - 1) // (r * 512)) * r * 512 for r in rates)
            needed_samples = target_samples + max_shift

            # 后续文件只为低频色散延迟提供提前量，文件缓存随窗口滑动。
            read_end       = end
            loaded_samples = real_samples
            while read_end < len(fits_list) and loaded_samples < needed_samples:
                loaded_samples += lengths[read_end]
                read_end += 1
            needed_files = fits_list[i:read_end]

            # 淘汰当前窗口之外的 GPU 缓存，读取本轮新增文件。
            for path in list(file_cache):
                if path not in needed_files:
                    del file_cache[path]

            # 借鉴 core 的有界预读：线程只碰主机数组，上传及所有 GPU 计算仍在主线程。
            # 队列只提前少量 FITS；每个文件读一次，原始 GPU 缓存继续复用窗口重叠。
            if config.fits_prefetch_workers:
                prefetch_stop = min(len(fits_list), read_end + int(config.fits_prefetch_workers))
                for path in fits_list[i:prefetch_stop]:
                    if path not in file_cache and path not in prefetch_futures:
                        prefetch_futures[path] = prefetch_executor.submit(read_fits_data, path, nbits)
            for path in needed_files:
                if path not in file_cache:
                    if config.fits_prefetch_workers:
                        host_data = prefetch_futures.pop(path).result()
                        file_cache[path] = load_func(path, reverse_flag, host_data)
                        del host_data
                    else:
                        file_cache[path] = load_func(path, reverse_flag)

            # 缓存保留相邻窗口重叠文件；连续输入缓冲和补齐噪声都在 GPU 上生成。
            raw_data = cp.empty((needed_samples, freq_reso), dtype=cp.uint16)
            offset   = 0
            for path in needed_files:
                count = min(file_cache[path].shape[0], needed_samples - offset)
                if count <= 0:
                    break
                raw_data[offset:offset + count] = file_cache[path][:count]
                offset += count

            if offset < needed_samples:
                # 观测结束后用末尾真实样本的统计量生成噪声，补足窗口及色散余量。
                tail  = raw_data[max(0, offset - 4096):offset].astype(cp.float32)
                noise = cp.random.random((needed_samples - offset, freq_reso), dtype=cp.float32)
                noise *= tail.std()
                noise += tail.mean()
                raw_data[offset:] = cp.clip(noise, 0, 65535).astype(cp.uint16)
                del tail, noise

            # 保留旧版文件命名：多文件窗口以首文件为名，block 编号从零开始。
            filename           = Path(fits_list[i]).stem
            fits_number        = int(re.search(r"(\d{4})\.fits$", fits_list[i]).group(1))
            file_start_seconds = (fits_number - 1) * file_leng * time_reso
            print(f"Processing file {fits_number}: {filename}; real_samples={real_samples}, padded_samples={target_samples}")

            # 每个输入块只消色散一次；kernel 内先应用原始采样时移，再做公共倍率平均。
            common_samples = target_samples // common_rate
            common_gpu     = cp.empty((common_samples, 512), dtype=cp.float32)
            _DEDISPERSE_KERNEL(
                ((common_gpu.size + 255) // 256,), (256,),
                (raw_data, shifts_gpu, common_gpu, np.int64(common_samples),
                 np.int32(freq_reso), np.int32(freq_reso // 512), np.int32(common_rate)),
            )
            common_data = torch.utils.dlpack.from_dlpack(common_gpu)
            del raw_data  # 同一 CUDA stream 顺序执行，消色散后释放连续输入缓冲。

            for rate in rates:
                nblocks        = (real_samples + rate * 512 - 1) // (rate * 512)
                output_samples = nblocks * 512
                extra_rate     = rate // common_rate

                # 各倍率使用独立图像缓冲；后续原位归一化不会修改公共消色散结果。
                if extra_rate == 1:
                    images = common_data[:output_samples].reshape(nblocks, 512, 512).clone()
                else:
                    images = common_data[:output_samples * extra_rate].reshape(nblocks, 512, extra_rate, 512).mean(dim=2)

                save_path_down = os.path.join(save_path, f"{rate:04d}")
                os.makedirs(save_path_down, exist_ok=True)

                # 每个倍率独立记录命中结果，分批推理以限制模型和归一化的显存峰值。
                detected_blocks = []
                detected_scores = []
                for batch_start in range(0, nblocks, config.classifier_batch_size):
                    batch_end = min(batch_start + config.classifier_batch_size, nblocks)
                    with torch.inference_mode():
                        batch  = preprocess_data(images[batch_start:batch_end])
                        scores = model(batch[:, None]).softmax(dim=1)[:, 1]

                        # 分数和编号体积很小；完整图像只在命中时下载。
                        hits = torch.nonzero(scores >= config.prob).flatten().cpu().tolist()
                        for hit in hits:
                            block      = batch_start + hit
                            candidate  = batch[hit].cpu().numpy()
                            valid_time = min(512, (real_samples - block * rate * 512 + rate - 1) // rate)
                            # 补齐部分禁止成为候选峰值；真实末尾半窗口仍参与检测。
                            if int(candidate.mean(axis=1).argmax()) >= valid_time:
                                continue
                            score = float(scores[hit].item())
                            detected_blocks.append(block)
                            detected_scores.append(score)

                            # 保存归一化后的 float32 图像；npy 与 jpg 沿用旧版名称和目录。
                            np.save(os.path.join(save_path_down, f"{filename}-{block:04d}.npy"), candidate)
                            plot_burst(
                                candidate, filename, block, time_reso * rate, fits_number, 512,
                                save_path_down + os.sep, file_leng, rate, freq, file_start_seconds,
                            )
                    del batch, scores

                print(f"{rate:04d} Detected blocks: {detected_blocks} Score: {detected_scores}")
                del images

            # 公共消色散结果在全部倍率完成后释放，随后推进到下一输入块。
            del common_data, common_gpu
            i = end


# ---------------------------------------------------------------------------
# 8. 入口配置与任务执行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- section 编号；提交器传入该参数，直接运行时默认处理 section 0 ----
    section = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config  = ProcessConfig(
        DM                      = 1376.5,
        prob                    = 0.6,
        block_size              = 512,
        section_num             = 32,  # 逻辑 section 总数
        # 支持多个倍率，例如 [4, 8, 16, 32]；公共数据按最大公约数复用。
        down_sampling_rate_list = np.array([8]),
        classifier_batch_size   = 32,  # 与远端 190417 旧入口的推理批大小一致。
        fits_prefetch_workers   = 2,   # 借鉴 core 的后台 FITS 预读；0 可用于串行对照。
    )

    # ---- 模型 ----
    # 把训练得到的 best_model_ema.pth 拷到此目录（或改成绝对路径）
    classifier_ckpt = "./models/binary_best_model_conv_small_ema.pth"
    model           = load_binary_classifier(
        classifier_ckpt, model_type="ConvNeXtNet", model_name="convnext_small"
    )
    print(f"[Model] loaded classifier from {classifier_ckpt}")

    # ---- 路径配置 ----
    # data_path 既可以是 source/date/ 直接含 fits 的目录，也可以是 source/ 含多 date 子目录的目录
    data_path   = "/data32/PT2026_0040/FRB20190417A/20260930/"  # 非 None 时覆盖 data_paths
    data_paths  = [data_path]
    save_base   = "/home/ykzhang/low.iops.files/"
    beam_filter = "M01"                                 # 'M01' / 'M02' / ... / 'all' / None

    # 备用配置示例
    # data_path   = '/path/to/observations/another_source/date/'
    # data_path   = '/path/to/observations/source/'
    # 多目标：data_path = None; data_paths = ['/path/to/root_a/', '/path/to/root_b/', ...]
    if data_path is not None:
        data_paths = [data_path]

    # 收集文件
    print(f"Organizing file lists from {data_paths}")
    all_file_lists = organize_file_lists_from_roots(data_paths, beam_filter)
    print(f"Total file lists: {len(all_file_lists)}")

    # 当前 section 处理的子集
    section_file_lists = distribute_file_lists(all_file_lists, config.section_num, section, config)
    print(f"Section {section} processing {len(section_file_lists)} file lists")

    # 文件组之间继续处理；累计失败用于最终 PBS 作业退出状态。
    failures = 0
    for identifier, fits_list, info in section_file_lists:
        print(f"\n{'=' * 60}\nProcessing: {identifier}\nFiles: {len(fits_list)}\n{'=' * 60}")

        # save_base 可以是公共输出根目录，也可以已经包含 source；避免重复追加同名目录。
        save_root = Path(save_base)
        if save_root.name != info["source"]:
            save_root = save_root / info["source"]
        save_path = str(save_root / "CalData" / info["date"] / info["beam"]) + os.sep

        try:
            process_fits_list(
                fits_list, model, config, save_path,
                n_search_files      = info.get("n_search_files"),
                reference_file_leng = info.get("reference_file_leng"),
            )
            print(f"Successfully processed {identifier}")
        except Exception as e:
            failures += 1
            print(f"Error processing {identifier}: {e}")
            import traceback

            traceback.print_exc()
    sys.exit(1 if failures else 0)
# fmt: on
