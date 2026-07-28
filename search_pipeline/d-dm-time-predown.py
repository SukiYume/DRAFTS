"""单阶段搜索：固定 DM 消色散 + 多尺度时间下采样 + binary 分类器逐块判定。

工作流：
  - 在指定 DM 下对 fits 文件序列做色散修正
  - 按多个 down_sampling_rate 把数据重塑成 ``[N, 512, 512]`` 图像
  - binary 分类器逐块打分；score >= prob 的块被保存（npy + jpg 画图）

模型加载来自 :mod:`binary_model`（与 ``binary_classification/binary_model.py`` 同步）。
checkpoint 可以是纯 ``state_dict`` 也可以是完整的 ``{model_state_dict: ...}`` dict——两种都兼容。

Usage:
  CUDA_VISIBLE_DEVICES=0 python d-dm-time-predown.py 0
  # 使用前编辑脚本底部 DM、section_num、data_path、save_base 和 beam_filter。
"""

import os, re, sys
import numpy as np
from pathlib import Path
from astropy.io import fits
from dataclasses import dataclass
from typing import List, Tuple, Optional

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('default')
sns.set_color_codes()

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from binary_model import build_binary_model


@dataclass
class ProcessConfig:
    """算法 / 运行时超参数。"""
    DM: float = 273.5                       # 用来做色散修正的 DM 值（pc cm^-3）
    prob: float = 0.5                       # 分类器判定阈值
    block_size: int = 512                   # 每张 512×512 图对应的时间样本数（降采样后）
    section_num: int = 19                    # section 总数（= GPU 数 x workers_per_gpu，见 s-pbsspt.py）
    down_sampling_rate_list: np.ndarray = None  # 多尺度搜索（不同时间分辨率）

    def __post_init__(self):
        if self.down_sampling_rate_list is None:
            self.down_sampling_rate_list = np.array([8])


# ---------------------------------------------------------------------------
# 1. FITS 读取
# ---------------------------------------------------------------------------

def _downsample_time(data, rate):
    if rate <= 1:
        return data.astype(np.float32, copy=False)
    rem = data.shape[0] % rate
    if rem:
        padding = np.random.rand(rate - rem, data.shape[1]).astype(np.float32)
        padding = padding * np.std(data) + np.mean(data)
        data = np.concatenate([data, padding], axis=0)
    return data.reshape(-1, rate, data.shape[1]).mean(axis=1).astype(np.float32)


def load_fits_file(file_name, reverse_flag=False, down_time_rate=1):
    """读取常规 FITS，输出已完成偏振平均和可选时间预下采样的 (time, freq)。"""
    with fits.open(file_name, memmap=True, lazy_load_hdus=True) as f:
        h = f[1].header
        arr = f[1].data['DATA'].reshape(h['NAXIS2'] * h['NSBLK'], h['NPOL'], h['NCHAN'])
        if reverse_flag:
            arr = arr[:, :, ::-1]
        data = arr[:, 0, :].astype(np.float32)
        data += arr[:, 1, :]
        data *= 0.5
    return _downsample_time(data, down_time_rate)


def load_2bit_fits_file(file_name, reverse_flag=False, down_time_rate=1):
    """读取 2-bit 压缩 fits 文件，输出 (time, freq) 二维数据。"""
    with fits.open(file_name) as f:
        h = f[1].header
        try:
            data = f[1].data['DATA']
        except Exception:
            # 缺 DATA 字段时用占位随机数据，便于流水线 sanity-test
            data = np.random.randint(0, 3,
                (h['NAXIS2'], h['NSBLK'] // 4, h['NPOL'], h['NCHAN'], 1), dtype=np.uint8)
    data = np.unpackbits(data.reshape(h['NAXIS2'], -1), axis=1).reshape(h['NAXIS2'], -1, 2)
    data = data[..., 0] << 1 | data[..., 1]
    data = data.reshape(h['NAXIS2'] * h['NSBLK'], h['NCHAN'])
    if reverse_flag:
        data = data[:, ::-1]
    return _downsample_time(data.astype(np.float32, copy=False), down_time_rate)


# ---------------------------------------------------------------------------
# 2. 单图归一化（CPU numpy 版本）
# ---------------------------------------------------------------------------

def preprocess_data(data, exp_cut=5):
    """归一化：减基线 → 分位截断 → min-max 到 [0, 1]。与训练时 BurstDataset.preprocess 等价。"""
    data       = data.copy() + 1
    data      /= np.mean(data, axis=0)
    vmin, vmax = np.nanpercentile(data, [exp_cut, 100 - exp_cut])
    data       = np.clip(data, vmin, vmax)
    data       = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return data


# ---------------------------------------------------------------------------
# 3. 可视化
# ---------------------------------------------------------------------------

def plot_burst(data, filename, block, time_reso, fits_number, block_size, save_path,
               file_leng, down_sampling_rate, freq):
    """画检测到的脉冲：上半 profile，下半 2D 频谱。"""
    w, h       = data.shape
    profile    = np.mean(data, axis=1)
    peak_idx   = int(np.argmax(profile))
    time_start = ((fits_number - 1) * file_leng // down_sampling_rate + block * block_size) * time_reso
    peak_time  = time_start + peak_idx * time_reso

    plt.figure(figsize=(5, 5))
    gs  = gridspec.GridSpec(4, 1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(gs[0, 0])
    plt.plot(profile, color='royalblue', alpha=0.8, lw=1)
    plt.scatter(peak_idx, profile[peak_idx], color='red', s=100, marker='x')
    plt.xlim(0, w)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(gs[1:, 0])
    plt.imshow(data.T, origin='lower', cmap='mako', aspect='auto')
    plt.scatter(peak_idx, 0, color='red', s=100, marker='x')
    plt.yticks(np.linspace(0, h, 6), np.int64(np.linspace(freq.min(), freq.max(), 6)))
    plt.xticks(np.linspace(0, w, 6),
               np.round(time_start + np.arange(6) * time_reso * block_size / 5, 2))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.savefig('{}{}-{:0>4d}-{}.jpg'.format(save_path, filename, block, peak_time),
                format='jpg', dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 4. 主处理函数
# ---------------------------------------------------------------------------

def process_fits_list(fits_list, model, config: ProcessConfig, save_path, n_search_files=None):
    """对 fits 文件列表做：消色散 → 多尺度下采样 → 分类器推理 → 保存正样本。

    n_search_files :
        仅对前 ``n_search_files`` 个文件启动检索 block，其余文件只作消色散读取提前量
        （section 切分时的尾部重叠，与 d-center-binary-core.process_fits_list 一致）。
        ``None`` 表示全部检索。
    """
    if not fits_list:
        return
    os.makedirs(save_path, exist_ok=True)

    # 读第一个 fits 拿观测参数
    with fits.open(fits_list[0]) as f:
        time_reso = f[1].header['TBIN']
        freq_reso = f[1].header['NCHAN']
        file_leng = f[1].header['NAXIS2'] * f[1].header['NSBLK']
        nbits     = f[1].header['NBITS']
        freq      = f[1].data['DAT_FREQ'][0, :].astype(np.float64)

    # 频率轴方向（如低频在右，需要反转）
    reverse_flag = False
    if freq[0] > freq[-1]:
        reverse_flag = True
        freq         = freq[::-1]

    load_func = load_2bit_fits_file if nbits == 2 else load_fits_file
    print(f"Using {'2-bit' if load_func is load_2bit_fits_file else 'standard'} FITS loader (NBITS={nbits})")

    # 预下采样：按最小 down_sampling_rate（上限 16）压缩时间维，减小后续消色散计算量
    pre_down_rate = int(min(np.min(config.down_sampling_rate_list), 16))

    # 色散延迟（原始采样点）
    dds_raw  = (4.15 * config.DM * (freq ** -2 - freq.max() ** -2) * 1e3 / time_reso).astype(np.int64)
    dds_file = int(np.ceil(dds_raw.max() / file_leng))
    # 在预下采样后的采样点上的色散延迟
    dds      = (dds_raw // pre_down_rate).astype(np.int64)

    max_down_rate = config.down_sampling_rate_list.max()
    block_file    = int(np.ceil(max_down_rate * config.block_size / file_leng))
    comb_file     = block_file + dds_file
    print(f"Processing {len(fits_list)} files, block_file={block_file}, comb_file={comb_file}")

    # 预计算消色散索引（避免循环重算）
    target_length = block_file * file_leng // pre_down_rate
    time_indices  = dds[:, None] + np.arange(target_length)
    freq_indices  = np.arange(freq_reso)[:, None]

    # 与 d-center-binary-core.py 一样保留滑动窗口缓存，避免相邻窗口重复读盘。
    # 例如单 rate=8 时 block_file=1, comb_file=2，内部 FITS 文件不缓存会被读两遍。
    file_cache = {}

    search_limit = len(fits_list) if n_search_files is None else min(n_search_files, len(fits_list))
    for i in range(0, search_limit, block_file):
        fits_number = int(re.search(r'(\d{4})\.fits', fits_list[i]).group(1))
        filename    = os.path.basename(fits_list[i]).split('.fits')[0]
        print(f"Processing file {fits_number}: {filename}")

        # 读盘 + 拼接
        needed_files = [fits_list[i + j] for j in range(comb_file) if i + j < len(fits_list)]
        for fp in list(file_cache):
            if fp not in needed_files:
                del file_cache[fp]
        for fp in needed_files:
            if fp not in file_cache:
                file_cache[fp] = load_func(fp, reverse_flag, pre_down_rate)
        raw_data = np.concatenate([file_cache[fp] for fp in needed_files], axis=0)

        # 不足时随机填充
        expected_samples = comb_file * file_leng // pre_down_rate
        if raw_data.shape[0] < expected_samples:
            padding  = np.random.rand(expected_samples - raw_data.shape[0], freq_reso).astype(np.float32)
            padding  = padding * np.std(raw_data) + np.mean(raw_data)
            raw_data = np.concatenate([raw_data, padding], axis=0).astype(np.float32)

        # 消色散（向量化）
        new_data = raw_data[time_indices, freq_indices].T
        del raw_data

        # 多尺度循环
        for down_sampling_rate in config.down_sampling_rate_list:
            save_path_down = f'{save_path}{down_sampling_rate:0>4d}/'
            os.makedirs(save_path_down, exist_ok=True)

            eff_down_rate = max(1, int(down_sampling_rate // pre_down_rate))
            pred_data     = np.mean(
                new_data.reshape(block_file * file_leng // down_sampling_rate, eff_down_rate, 512, freq_reso // 512),
                axis=(1, 3),
            ).reshape(-1, 512, 512)

            # 逐块预处理
            for j in range(pred_data.shape[0]):
                pred_data[j] = preprocess_data(pred_data[j])

            # 分批推理
            batch_size  = 32
            predict_res = []
            for batch_start in range(0, pred_data.shape[0], batch_size):
                batch_end  = min(batch_start + batch_size, pred_data.shape[0])
                batch_data = pred_data[batch_start: batch_end]
                inputs     = torch.from_numpy(batch_data[:, np.newaxis, :, :]).float().to(device)
                with torch.no_grad():
                    batch_res = model(inputs).softmax(dim=1)[:, 1].cpu().numpy()
                predict_res.extend(batch_res)
            predict_res = np.array(predict_res)

            # 保存正样本
            blocks = np.where(predict_res >= config.prob)[0]
            print(f'{down_sampling_rate:0>4d}', 'Detected blocks:', blocks.tolist(),
                  'Score:', predict_res[blocks].tolist())
            for block in blocks:
                np.save(f'{save_path_down}{filename}-{block:0>4d}.npy', pred_data[block])
                plot_burst(
                    pred_data[block], filename, block, time_reso * down_sampling_rate,
                    fits_number, config.block_size, save_path_down, file_leng,
                    down_sampling_rate, freq,
                )

        del new_data


# ---------------------------------------------------------------------------
# 路径聚合 + 多 section 分配（与 d-center-binary-gate.py 一致）
# ---------------------------------------------------------------------------

def organize_file_lists(data_path: str, beam_filter: Optional[str] = None) -> List[Tuple[str, List[str], dict]]:
    """按 source/date/beam 聚合 fits 文件列表。

    Returns
    -------
    List of ``(identifier, files_list, path_info)``；``path_info`` 是
    ``{'source': str, 'date': str, 'beam': str}``。
    """
    result = []
    path   = Path(data_path)
    all_fits = sorted([
        f.name for f in path.iterdir()
        if f.is_file() and f.suffix == '.fits' and not f.name.startswith('.')
        and '_N_' not in f.name and '_W_' not in f.name and '_F_' not in f.name
    ])

    if all_fits:
        # 按文件名里的 -Mxx_ 提取 beam；不含该标识的文件跳过，避免 .group(1) 对 None 崩溃
        beam_match = {f: re.search(r'-(M\d{2})_', f) for f in all_fits}
        skipped    = [f for f, m in beam_match.items() if m is None]
        if skipped:
            print(f'[organize] {path}: 跳过 {len(skipped)} 个无 -Mxx_ beam 标识的 fits')
        all_fits   = [f for f in all_fits if beam_match[f] is not None]
        beams      = np.unique([beam_match[f].group(1) for f in all_fits]).tolist()
        if beam_filter and beam_filter != 'all':
            beams = [beam_filter] if beam_filter in beams else []

        # 从 fits 往前数：当前目录是 date，父目录是 source
        date_name   = path.name
        source_name = path.parent.name

        for beam in beams:
            beam_files = sorted([str(path / f) for f in all_fits if beam in f])
            if beam_files:
                info       = {'source': source_name, 'date': date_name, 'beam': beam}
                identifier = f"{source_name}_{date_name}_{beam}"
                result.append((identifier, beam_files, info))
    else:
        # 递归子目录
        subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
        for subdir in subdirs:
            result.extend(organize_file_lists(str(subdir), beam_filter))

    return result


def organize_file_lists_from_roots(data_paths: List[str], beam_filter: Optional[str] = None) -> List[Tuple[str, List[str], dict]]:
    """Collect file lists from multiple data roots and keep the root in task metadata."""
    result = []
    for data_path in data_paths:
        root_label = Path(data_path).name
        for identifier, files, info in organize_file_lists(data_path, beam_filter):
            info = {**info, 'root': root_label}
            result.append((f"{root_label}_{identifier}", files, info))
    return result


def compute_dispersion_overlap(fits_file, config):
    """读取 ``fits_file`` 观测参数后返回相邻 section 边界所需的色散重叠文件数。

    与 d-center-binary-core.compute_dispersion_overlap 逻辑一致，只是这里换算最大色散延迟
    用的是固定 ``config.DM``，而不是 dm_range/dm_scale/dm_offset。
    """
    with fits.open(fits_file, memmap=True, lazy_load_hdus=True) as f:
        time_reso = f[1].header['TBIN']
        file_leng = f[1].header['NAXIS2'] * f[1].header['NSBLK']
        freq      = f[1].data['DAT_FREQ'][0, :].astype(np.float64)
    if freq[0] > freq[-1]:
        freq = freq[::-1]
    dds_max = int(4.15 * config.DM * (freq.min() ** -2 - freq.max() ** -2) * 1e3 / time_reso)
    return int(np.ceil(dds_max / file_leng))


def distribute_file_lists(file_lists, section_num, section, config):
    """把所有 (identifier, files, info) 均匀分配给指定 section。

    单个连续观测按 section 切分时，给每段尾部补 ``dds_file`` 个文件作为消色散读取提前量
    （与下一段重叠，只读不检索），避免段间接缝处色散修正失真；``info['n_search_files']``
    记录本段实际需检索的文件数（不含重叠）。
    """
    if len(file_lists) == 1:
        identifier, fits_list, info = file_lists[0]
        section_size = len(fits_list) // section_num + 1
        start_idx    = section * section_size
        end_idx      = (section + 1) * section_size
        n_search     = len(fits_list[start_idx: end_idx])               # 本段实际要检索的文件数
        if n_search == 0:
            return []
        overlap  = compute_dispersion_overlap(fits_list[0], config)
        sub_fits = fits_list[start_idx: end_idx + overlap]              # 末尾多带 overlap 个重叠文件
        info     = {**info, 'n_search_files': n_search}
        return [(f"{identifier}_section{section}", sub_fits, info)]

    return [
        (identifier, fits_list, info)
        for idx, (identifier, fits_list, info) in enumerate(file_lists)
        if idx % section_num == section
    ]


# ---------------------------------------------------------------------------
# 6. 模型加载（与 binary_classification 新训练输出兼容）
# ---------------------------------------------------------------------------

def load_binary_classifier(ckpt_path, model_type='ConvNeXtNet', model_name='convnext_tiny',
                           num_classes=2, dropout=0.5):
    """构建模型 + 加载权重；兼容纯 state_dict 与完整 checkpoint dict 两种保存格式。

    ``weights_only`` 仅新 PyTorch 支持，老版本 TypeError 时退回不带该 kwarg 的调用。
    """
    model = build_binary_model(
        model_type=model_type, model_name=model_name,
        num_classes=num_classes, pretrained=False, dropout=dropout,
    ).to(device)

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=True)
    return model.eval()


# ---------------------------------------------------------------------------
# 7. 入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # ---- section 编号（由 s-pbsspt.py 生成的 PBS 脚本注入；一块 GPU 可能对应多个 section）----
    section = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config  = ProcessConfig(
        DM                      = 273.5,
        prob                    = 0.5,
        block_size              = 512,
        section_num             = 15,        # 与 PBS submission 展开后的总 section 数一致（= GPU 数 x workers_per_gpu）
        # 多尺度搜索：覆盖窄/宽脉冲
        down_sampling_rate_list = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
    )

    # ---- 模型 ----
    # 把训练得到的 best_model_ema.pth 拷到此目录（或改成绝对路径）
    classifier_ckpt = './models/binary_best_model_conv_small_ema.pth'
    model = load_binary_classifier(classifier_ckpt,
                                   model_type='ConvNeXtNet', model_name='convnext_small')
    print(f'[Model] loaded classifier from {classifier_ckpt}')

    # ---- 路径配置 ----
    # data_path 既可以是 source/date/ 直接含 fits 的目录，也可以是 source/ 含多 date 子目录的目录
    data_path   = '/path/to/observations/source/date/'       # 单目标模板替换用；非 None 时覆盖 data_paths
    data_paths  = [data_path]
    save_base   = '/path/to/observations/'
    beam_filter = 'M01'                                  # 'M01' / 'M02' / ... / 'all' / None

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

    for identifier, fits_list, info in section_file_lists:
        print(f"\n{'=' * 60}\nProcessing: {identifier}\nFiles: {len(fits_list)}\n{'=' * 60}")
        # root 只用于区分输入任务，不参与输出目录拼接。
        save_path = str(Path(save_base) / info['source'] / 'CalData' / info['date'] / info['beam']) + os.sep

        try:
            process_fits_list(fits_list, model, config, save_path, n_search_files=info.get('n_search_files'))
            print(f"Successfully processed {identifier}")
        except Exception as e:
            print(f"Error processing {identifier}: {e}")
            import traceback
            traceback.print_exc()
