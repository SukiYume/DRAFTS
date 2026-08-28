"""原始分辨率 GPU 两阶段搜索的 gate / 调度入口。

职责：
  1. 解析数据路径（支持 source/date/ 直接含 FITS 或 source/ 含多个 date 子目录）
  2. 按 source/date/beam 聚合后切片分配给当前 section
  3. 加载 CenterNet detector 和 binary 分类器
  4. 调用 :mod:`d-center-binary-core-new.py` 完成动态 RFI、细通道消色散和两阶段推理
  5. 维护 ``processing_log.txt`` 实现任务级断点续跑

DM 参数均为真实物理量。默认盲搜区间为 ``[0, 4096)`` pc cm^-3，步长为
1 pc cm^-3；原始时间和频率数据上传 GPU 后才做消色散与下采样。

Usage:
  CUDA_VISIBLE_DEVICES=0 python d-center-binary-gate-new.py 0
  # PBS 生产运行通过 s-pbsspt.py 提交；section_num 应等于 GPU 数 x workers_per_gpu。
"""

import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from binary_model import build_binary_model

torch.set_num_threads(4)
torch.set_num_interop_threads(1)


# ---------------------------------------------------------------------------
# 加载 new core（文件名带短横线，需要 importlib spec_from_file_location）
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_PATH = os.path.join(BASE_DIR, 'd-center-binary-core-new.py')
spec = importlib.util.spec_from_file_location('d_center_binary_core_new', CORE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f'Unable to load core module from {CORE_PATH}')
core = importlib.util.module_from_spec(spec)
sys.modules['d_center_binary_core_new'] = core
spec.loader.exec_module(core)

ProcessConfig = core.ProcessConfig
process_fits_list = core.process_fits_list
compute_dispersion_overlap = core.compute_dispersion_overlap
device = core.device


# ---------------------------------------------------------------------------
# 模型加载：CenterNet → 统一返回 (model, decode_fn)
# ---------------------------------------------------------------------------

CENTERNET_BACKBONES = {
    'centernet_conv_tiny': 'convnext_tiny',
    'centernet_conv_small': 'convnext_small',
}
VALID_DETECTORS = ' / '.join(CENTERNET_BACKBONES)


def _load_checkpoint(ckpt_path):
    """兼容纯 state_dict、EMA checkpoint 和 DataParallel 前缀。"""
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'ema_state_dict' in state:
        state = state['ema_state_dict']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    if isinstance(state, dict):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
    return state


def _load_eval_model(model, ckpt_path):
    """加载权重并切换到 GPU eval 模式。"""
    model.load_state_dict(_load_checkpoint(ckpt_path), strict=True)
    return model.to(device).eval()


def _make_centernet_decode_fn(decode_centernet_outputs):
    """把 CenterNet 中心点输出适配为 core 需要的 boxes 接口。"""
    def decode_fn(raw_pred, conf_thr=0.3, iou_thr=None, topk=100):
        results = decode_centernet_outputs(
            raw_pred, conf_thr=conf_thr, topk=topk, down_ratio=4,
        )
        output = []
        for scores, centers in results:
            if centers is None:
                output.append((None, None))
                continue
            zeros = np.zeros(len(centers), dtype=np.float32)
            boxes = np.column_stack(
                [centers[:, 0], centers[:, 1], zeros, zeros],
            ).astype(np.float32)
            output.append((scores.astype(np.float32), boxes))
        return output

    return decode_fn


def load_detector(detector_type, ckpt_path):
    """构建 CenterNet detector，加载权重并返回统一解码函数。"""
    name = detector_type.lower()
    if name in CENTERNET_BACKBONES:
        from centernet_eval import decode_centernet_outputs
        from centernet_model import build_centernet_model

        model = build_centernet_model(
            backbone=CENTERNET_BACKBONES[name],
            pretrained=False,
            head_ch=128,
        )
        return (
            _load_eval_model(model, ckpt_path),
            _make_centernet_decode_fn(decode_centernet_outputs),
        )
    raise ValueError(f'Unknown detector_type {detector_type!r}; valid: {VALID_DETECTORS}')


def load_binary_classifier(ckpt_path, model_type='ConvNeXtNet', model_name='convnext_tiny',
                           num_classes=2, dropout=0.5):
    """构建 binary 分类器并加载权重。"""
    model = build_binary_model(
        model_type=model_type,
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=dropout,
    )
    return _load_eval_model(model, ckpt_path)


# ---------------------------------------------------------------------------
# 路径聚合 + 多 section 分配
# ---------------------------------------------------------------------------

def organize_file_lists(data_path: str,
                        beam_filter: Optional[str] = None) -> List[Tuple[str, List[str], dict]]:
    """按 source/date/beam 聚合 FITS 文件列表。"""
    result = []
    path = Path(data_path)
    all_fits = sorted([
        item.name for item in path.iterdir()
        if item.is_file() and item.suffix == '.fits' and not item.name.startswith('.')
        and '_N_' not in item.name and '_W_' not in item.name and '_F_' not in item.name
    ])

    if all_fits:
        beam_match = {name: re.search(r'-(M\d{2})_', name) for name in all_fits}
        skipped = [name for name, match in beam_match.items() if match is None]
        if skipped:
            print(f'[organize] {path}: 跳过 {len(skipped)} 个无 -Mxx_ beam 标识的 FITS')
        all_fits = [name for name in all_fits if beam_match[name] is not None]
        beams = np.unique([beam_match[name].group(1) for name in all_fits]).tolist()
        if beam_filter and beam_filter != 'all':
            beams = [beam_filter] if beam_filter in beams else []

        date_name = path.name
        source_name = path.parent.name
        for beam in beams:
            beam_files = sorted([str(path / name) for name in all_fits if beam in name])
            if beam_files:
                info = {'source': source_name, 'date': date_name, 'beam': beam}
                identifier = f'{source_name}_{date_name}_{beam}'
                result.append((identifier, beam_files, info))
    else:
        for subdir in sorted([item for item in path.iterdir() if item.is_dir()]):
            result.extend(organize_file_lists(str(subdir), beam_filter))

    return result


def organize_file_lists_from_roots(
        data_paths: List[str],
        beam_filter: Optional[str] = None) -> List[Tuple[str, List[str], dict]]:
    """从多个数据根收集任务，并把输入根名称写入任务元数据。"""
    result = []
    for data_path in data_paths:
        root_label = Path(data_path).name
        for identifier, files, info in organize_file_lists(data_path, beam_filter):
            info = {**info, 'root': root_label}
            result.append((f'{root_label}_{identifier}', files, info))
    return result


def load_file_lists_from_manifest(
        manifest_path: str) -> List[Tuple[str, List[str], dict]]:
    """读取 :mod:`c-manifest-build.py` 生成的任务清单。"""
    with open(manifest_path, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)

    tasks = payload.get('tasks', payload) if isinstance(payload, dict) else payload
    result = []
    for index, task in enumerate(tasks):
        try:
            identifier = task['identifier']
            files = task['files']
            info = task.get('info', {})
        except (TypeError, KeyError) as exc:
            raise ValueError(f'Invalid manifest task at index {index}: {task!r}') from exc
        result.append((identifier, files, info))
    return result


def distribute_file_lists(file_lists, section_num, section, process_config):
    """为一个 section 分配任务，并为连续观测补足色散读取重叠。"""
    if len(file_lists) == 1:
        identifier, fits_list, info = file_lists[0]
        section_size = len(fits_list) // section_num + 1
        start_idx = section * section_size
        end_idx = (section + 1) * section_size
        n_search = len(fits_list[start_idx:end_idx])
        if n_search == 0:
            return []
        overlap = compute_dispersion_overlap(fits_list[0], process_config)
        sub_fits = fits_list[start_idx:end_idx + overlap]
        info = {**info, 'n_search_files': n_search}
        return [(f'{identifier}_section{section}', sub_fits, info)]

    return [
        (identifier, fits_list, info)
        for index, (identifier, fits_list, info) in enumerate(file_lists)
        if index % section_num == section
    ]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    section = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # 盲搜默认区间：[0, 4096) pc cm^-3，共 4096 个 trial、4 张 DM-time 图。
    process_config = ProcessConfig(
        dm_min       = 0,
        dm_range     = 4096,
        dm_step      = 1,
        dm_threshold = 10,
        block_size   = 4096,
        det_prob     = 0.45,
        section_num  = 32,
        time_factor  = 8,
        rfi_filter            = True,
        rfi_level_ratio       = 0.05,
        rfi_cv_ratio          = 3.0,
        rfi_zero_dm_filter    = True,
        rfi_zero_dm_sigma     = 8.0,
        rfi_zero_dm_occupancy = 0.03,
    )

    # ---- 检测模型（默认使用 v10 CenterNet ConvNeXt Tiny）----
    DETECTOR_TYPE = 'centernet_conv_tiny'
    DETECTOR_CKPT = './models/object_best_model_centernet_conv_tiny_ema_v10.pth'
    model, decode_fn = load_detector(DETECTOR_TYPE, DETECTOR_CKPT)
    print(f'[Model] detector={DETECTOR_TYPE}  ckpt={DETECTOR_CKPT}')

    # ---- binary 分类器 ----
    CLASSIFIER_MODEL_NAME = 'convnext_small'
    CLASSIFIER_CKPT = './models/binary_best_model_conv_small_ema.pth'
    class_model = load_binary_classifier(
        CLASSIFIER_CKPT,
        model_type='ConvNeXtNet',
        model_name=CLASSIFIER_MODEL_NAME,
    )
    print(f'[Model] classifier ckpt={CLASSIFIER_CKPT}')

    # ---- 路径配置 ----
    log_file = './processing_log_new.txt'
    bad_fits_log = os.path.splitext(log_file)[0] + '_bad_fits.log'
    task_manifest = '/path/to/observations/CRAFTS/task_manifest.json'
    data_path = None
    data_paths = [
        '/path/to/CRAFTS/',
    ]
    save_base = '/path/to/observations/CRAFTS/'
    beam_filter = 'all'

    # 单目标直接设置 data_path；它会覆盖 data_paths 和 task_manifest。
    # data_path = '/path/to/observations/source/date/'
    # save_base = '/path/to/observations/'
    # beam_filter = 'M01'
    if data_path is not None:
        data_paths = [data_path]
        task_manifest = None

    completed_tasks = set()
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as stream:
            completed_tasks = {line.strip() for line in stream if line.strip()}
    print(f'Loaded {len(completed_tasks)} completed tasks from log')

    if task_manifest and os.path.exists(task_manifest):
        print(f'Loading file lists from manifest: {task_manifest}')
        all_file_lists = load_file_lists_from_manifest(task_manifest)
    else:
        print(f'Organizing file lists from {data_paths}')
        all_file_lists = organize_file_lists_from_roots(data_paths, beam_filter)
    print(f'Total file lists: {len(all_file_lists)}')

    section_file_lists = distribute_file_lists(
        all_file_lists, process_config.section_num, section, process_config,
    )
    print(f'Section {section} processing {len(section_file_lists)} file lists')

    if section_file_lists:
        print(f'\nTask Summary for Section {section}:')
        for identifier, fits_list, _ in section_file_lists:
            status = 'COMPLETED' if identifier in completed_tasks else 'PENDING'
            print(f'  [{status}] {identifier}: {len(fits_list)} files')
        print()

    failures = []
    for identifier, fits_list, info in section_file_lists:
        if identifier in completed_tasks:
            print(f'Skipping completed task: {identifier}')
            continue

        print(f"\n{'=' * 60}\nProcessing: {identifier}\nFiles: {len(fits_list)}\n{'=' * 60}")
        save_path = str(
            Path(save_base) / info['source'] / 'CentData' / info['date'] / info['beam'],
        ) + os.sep

        try:
            if info.get('n_search_files') is None and os.path.isdir(save_path):
                print(f'Cleaning previous beam outputs: {save_path}')
                shutil.rmtree(save_path)
            process_fits_list(
                fits_list,
                save_path,
                model,
                class_model,
                process_config,
                decode_fn=decode_fn,
                n_search_files=info.get('n_search_files'),
                task_identifier=identifier,
                task_info=info,
                bad_fits_log=bad_fits_log,
            )
            print(f'Successfully processed {identifier}')
            with open(log_file, 'a', encoding='utf-8') as stream:
                stream.write(f'{identifier}\n')
            completed_tasks.add(identifier)
        except Exception as exc:
            print(f'Error processing {identifier}: {exc}')
            import traceback
            traceback.print_exc()
            failures.append((identifier, str(exc)))

    if failures:
        print(f'Failed tasks: {len(failures)}', file=sys.stderr)
        for identifier, reason in failures:
            print(f'  {identifier}: {reason}', file=sys.stderr)
        sys.exit(1)
