"""两阶段搜索的 gate / 调度入口。

职责：
  1. 解析数据路径（支持 source/date/ 直接含 fits 或 source/ 含多个 date 子目录）
  2. 按 source/date/beam 聚合后切片分配给当前 section（一块 GPU 可能同时跑多个 section，
     取决于 s-pbsspt.py 的 workers_per_gpu）
  3. 加载检测模型（CenterNet）+ binary 分类器
  4. 调 :func:`d_center_binary_core.process_fits_list` 处理每个任务
  5. 维护 ``processing_log.txt`` 实现任务级断点续跑

切换模型：编辑下方 ``DETECTOR_TYPE`` / ``DETECTOR_CKPT`` /
``CLASSIFIER_MODEL_NAME`` / ``CLASSIFIER_CKPT``。

Usage:
  CUDA_VISIBLE_DEVICES=0 python d-center-binary-gate.py 0
  # PBS 生产运行通过 s-pbsspt.py 提交；section_num 应等于 GPU 数 x workers_per_gpu。
"""

import os, sys, re, json, shutil
import numpy as np
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

from binary_model import build_binary_model


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
    """把 CenterNet 的 centers 解码适配成统一的 boxes 接口。"""
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
# 路径聚合 + 多 section 分配（与 d-dm-time-predown.py 一致）
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


def load_file_lists_from_manifest(manifest_path: str) -> List[Tuple[str, List[str], dict]]:
    """Load prebuilt task lists produced by c-manifest-build.py."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    tasks = payload.get('tasks', payload) if isinstance(payload, dict) else payload
    result = []
    for idx, task in enumerate(tasks):
        try:
            identifier = task['identifier']
            files      = task['files']
            info       = task.get('info', {})
        except (TypeError, KeyError) as exc:
            raise ValueError(f'Invalid manifest task at index {idx}: {task!r}') from exc
        result.append((identifier, files, info))
    return result


def distribute_file_lists(file_lists, section_num, section, process_config):
    """把所有 (identifier, files, info) 均匀分配给指定 section。

    单个连续观测按 section 切分时，给每段尾部补 ``dds_file`` 个文件作为消色散读取提前量
    （与下一段重叠，只读不检索），避免段间接缝处高 DM 灵敏度下降；``info['n_search_files']``
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
        overlap  = compute_dispersion_overlap(fits_list[0], process_config)
        sub_fits = fits_list[start_idx: end_idx + overlap]              # 末尾多带 overlap 个重叠文件
        info     = {**info, 'n_search_files': n_search}
        return [(f"{identifier}_section{section}", sub_fits, info)]

    return [
        (identifier, fits_list, info)
        for idx, (identifier, fits_list, info) in enumerate(file_lists)
        if idx % section_num == section
    ]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # ---- section 编号（由 s-pbsspt.py 生成的 PBS 脚本注入；一块 GPU 可能对应多个 section）----
    section = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # ---- 算法配置 ----
    process_config = ProcessConfig(
        dm_range     = 4096,        # 总 DM 扫描范围 / (1/dm_scale)
        dm_scale     = 1,           # 每个 DM 索引对应的 DM 步长（pc cm^-3）
        dm_offset    = 0,           # DM = idx * dm_scale + dm_offset
        dm_threshold = 10,          # 低于该 DM 直接丢弃（视为 RFI）
        block_size   = 4096,        # 每个时间切片的样本数（降采样后）
        dm_span      = 1024,        # 每张 512x512 检测图覆盖的原始 DM 点数
        det_prob     = 0.45,        # 中心检测置信度阈值
        section_num  = 32,          # 与 PBS submission 展开后的总 section 数一致（= GPU 数 x workers_per_gpu）
        time_factor  = 8,           # 控制 down_time_rate
    )

    # ---- 检测模型（默认使用 v10 CenterNet ConvNeXt Tiny）----
    # detector 可选：centernet_conv_tiny / centernet_conv_small
    # 默认权重：models/object_best_model_centernet_conv_tiny_ema_v10.pth
    DETECTOR_TYPE = 'centernet_conv_tiny'
    DETECTOR_CKPT = './models/object_best_model_centernet_conv_tiny_ema_v10.pth'

    model, decode_fn = load_detector(DETECTOR_TYPE, DETECTOR_CKPT)
    print(f'[Model] detector={DETECTOR_TYPE}  ckpt={DETECTOR_CKPT}')

    # ---- binary 分类器（默认使用最近验证的 ConvNeXt Small）----
    # classifier model_name 可选：convnext_tiny / convnext_small
    CLASSIFIER_MODEL_NAME = 'convnext_small'
    CLASSIFIER_CKPT = './models/binary_best_model_conv_small_ema.pth'
    class_model = load_binary_classifier(CLASSIFIER_CKPT,
                                          model_type='ConvNeXtNet', model_name=CLASSIFIER_MODEL_NAME)
    print(f'[Model] classifier ckpt={CLASSIFIER_CKPT}')

    # ---- 路径配置 ----
    log_file    = './processing_log_zd202x_1_1_2bit.txt'
    bad_fits_log = os.path.splitext(log_file)[0] + '_bad_fits.log'
    task_manifest = '/path/to/observations/CRAFTS/task_manifest_zd202x_1_1_2bit.json'
    data_path   = None                                      # 单目标模板替换用；非 None 时覆盖 data_paths
    data_paths  = [
        '/path/to/CRAFTS/',
    ]
    save_base   = '/path/to/observations/CRAFTS/'
    beam_filter = 'all'                                     # 'M01' / 'M02' / 'all' / None

    # ---- 路径配置示例 ----
    # log_file    = './processing_log.txt'
    # data_path   = '/path/to/observations/source/date/'
    # save_base   = '/path/to/observations/'
    # beam_filter = 'M01'                                     # 'M01' / 'M02' / 'all' / None

    # 备用配置示例
    # data_path   = '/path/to/another/source/date/'
    # data_path   = '/path/to/CRAFTS/'
    if data_path is not None:
        data_paths = [data_path]

    # ---- 读取已完成任务（断点续跑）----
    completed_tasks = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            completed_tasks = set(line.strip() for line in f if line.strip())
    print(f'Loaded {len(completed_tasks)} completed tasks from log')

    # ---- 收集 + 分配 ----
    if task_manifest and os.path.exists(task_manifest):
        print(f'Loading file lists from manifest: {task_manifest}')
        all_file_lists = load_file_lists_from_manifest(task_manifest)
    else:
        print(f'Organizing file lists from {data_paths}')
        all_file_lists = organize_file_lists_from_roots(data_paths, beam_filter)
    print(f'Total file lists: {len(all_file_lists)}')

    section_file_lists = distribute_file_lists(all_file_lists, process_config.section_num, section,
                                               process_config)
    print(f'Section {section} processing {len(section_file_lists)} file lists')

    if section_file_lists:
        print(f"\nTask Summary for Section {section}:")
        for identifier, fits_list, info in section_file_lists:
            status = "COMPLETED" if identifier in completed_tasks else "PENDING"
            print(f"  [{status}] {identifier}: {len(fits_list)} files")
        print()

    # ---- 主循环 ----
    failures = []
    for identifier, fits_list, info in section_file_lists:
        if identifier in completed_tasks:
            print(f'Skipping completed task: {identifier}')
            continue

        print(f"\n{'=' * 60}\nProcessing: {identifier}\nFiles: {len(fits_list)}\n{'=' * 60}")
        # root 只用于区分输入任务，不参与输出目录拼接。
        save_path = str(Path(save_base) / info['source'] / 'CentData' / info['date'] / info['beam']) + os.sep

        try:
            if info.get('n_search_files') is None and os.path.isdir(save_path):
                print(f'Cleaning previous beam outputs: {save_path}')
                shutil.rmtree(save_path)
            process_fits_list(fits_list, save_path, model, class_model, process_config,
                              decode_fn=decode_fn, n_search_files=info.get('n_search_files'),
                              task_identifier=identifier, task_info=info,
                              bad_fits_log=bad_fits_log)
            print(f'Successfully processed {identifier}')
            # 记录完成
            with open(log_file, 'a') as f:
                f.write(f'{identifier}\n')
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
