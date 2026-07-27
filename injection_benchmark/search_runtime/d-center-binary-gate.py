"""两阶段搜索的 gate / 调度入口。

职责：
  1. 解析数据路径（支持 source/date/ 直接含 fits 或 source/ 含多个 date 子目录）
  2. 按 source/date/beam 聚合后切片分配给当前 GPU section
  3. 加载 CenterNet 检测模型 + binary 分类器
  4. 调 :func:`d_center_binary_core.process_fits_list` 处理每个任务

当前文件只提供可导入函数；实际命令行入口是 ``t-blind-section.py``。
"""

import os
import re
import sys
import numpy as np
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional

import torch

from binary_model      import build_binary_model


# ---------------------------------------------------------------------------
# 加载 core 模块（文件名带短横线，需要 importlib spec_from_file_location）
# ---------------------------------------------------------------------------

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.join(BASE_DIR, 'd-center-binary-core.py')
spec      = importlib.util.spec_from_file_location('d_center_binary_core', CORE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f'Unable to load core module from {CORE_PATH}')
core = importlib.util.module_from_spec(spec)
sys.modules['d_center_binary_core'] = core
spec.loader.exec_module(core)

ProcessConfig              = core.ProcessConfig
process_fits_list          = core.process_fits_list
compute_dispersion_overlap = core.compute_dispersion_overlap
device                     = core.device


# ---------------------------------------------------------------------------
# 模型加载：CenterNet → 统一返回 (model, decode_fn)
# decode_fn 签名:  (raw_pred, conf_thr, iou_thr) -> list[(scores_np, boxes_cxcywh_np)]
# ---------------------------------------------------------------------------

CENTERNET_BACKBONES = {
    'centernet_conv_tiny': 'convnext_tiny',
    'centernet_conv_small': 'convnext_small',
}
VALID_DETECTORS = ' / '.join(CENTERNET_BACKBONES)


def _load_checkpoint(ckpt_path):
    """兼容纯 state_dict / 完整 checkpoint dict。"""
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    return state


def _load_eval_model(model, ckpt_path):
    """加载权重并切到 eval。"""
    state = _load_checkpoint(ckpt_path)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _make_centernet_decode_fn(decode_centernet_outputs):
    """把 CenterNet 的 centers 解码适配成统一候选格式。"""
    def decode_fn(raw_pred, conf_thr=0.3, iou_thr=None, topk=100):
        res = decode_centernet_outputs(raw_pred, conf_thr=conf_thr, topk=topk, down_ratio=4)
        out = []
        for s, c in res:
            if c is None:
                out.append((None, None))
                continue
            zeros = np.zeros(len(c), dtype=np.float32)
            boxes = np.column_stack([c[:, 0], c[:, 1], zeros, zeros]).astype(np.float32)
            out.append((s.astype(np.float32), boxes))
        return out

    return decode_fn


def load_detector(detector_type, ckpt_path):
    """根据 type 字符串构建检测模型并加载权重。返回 ``(model, decode_fn)``。

    detector_type:
        - ``centernet_conv_tiny``      → CenterNet(convnext_tiny) + 中心点解码
        - ``centernet_conv_small``     → CenterNet(convnext_small) + 中心点解码
    """
    name = detector_type.lower()
    if name in CENTERNET_BACKBONES:
        from centernet_eval import decode_centernet_outputs
        from centernet_model import build_centernet_model

        model = build_centernet_model(
            backbone=CENTERNET_BACKBONES[name],
            pretrained=False,
            head_ch=128,
        )
        return _load_eval_model(model, ckpt_path), _make_centernet_decode_fn(decode_centernet_outputs)

    raise ValueError(f"Unknown detector_type {detector_type!r}; valid: {VALID_DETECTORS}")


def load_binary_classifier(ckpt_path, model_type='ConvNeXtNet', model_name='convnext_tiny',
                           num_classes=2, dropout=0.5):
    """构建 binary 分类器 + 加载权重；兼容纯 state_dict 与完整 checkpoint dict。"""
    model = build_binary_model(
        model_type=model_type, model_name=model_name,
        num_classes=num_classes, pretrained=False, dropout=dropout,
    )
    return _load_eval_model(model, ckpt_path)


# ---------------------------------------------------------------------------
# 路径聚合 + 多卡分配（与 d-dm-time-predown.py 一致）
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


def distribute_file_lists(file_lists, gpu_num, section, process_config):
    """把所有 (identifier, files, info) 均匀分配给指定 section。

    单个连续观测按 GPU 切分时，给每段尾部补 ``dds_file`` 个文件作为消色散读取提前量
    （与下一段重叠，只读不检索），避免段间接缝处高 DM 灵敏度下降；``info['n_search_files']``
    记录本段实际需检索的文件数（不含重叠）。
    """
    if len(file_lists) == 1:
        identifier, fits_list, info = file_lists[0]
        section_size = len(fits_list) // gpu_num + 1
        start_idx    = section * section_size
        end_idx      = (section + 1) * section_size
        n_search     = len(fits_list[start_idx: end_idx])               # 本段实际要检索的文件数
        if n_search == 0:
            return []
        overlap  = compute_dispersion_overlap(fits_list[0], process_config)
        sub_fits = fits_list[start_idx: end_idx + overlap]              # 末尾多带 overlap 个重叠文件
        info     = {**info, 'n_search_files': n_search}
        return [(f"{identifier}_section{section}", sub_fits, info)]

    return [
        (identifier, fits_list, info)
        for idx, (identifier, fits_list, info) in enumerate(file_lists)
        if idx % gpu_num == section
    ]
