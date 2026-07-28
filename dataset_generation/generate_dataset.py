"""Build CenterNet H5 training data from raw FAST injection + dedispersion.

The generator does not draw signals directly in time-DM space.  It injects the
existing multifitting dynamic-spectrum model into real raw8 FAST data, runs the
DRAFTS CuPy dedispersion kernel over a large 0-4096 DM canvas, then crops and
area-resamples varied DM/time windows into 512x512 detector images.

Large intermediates are never saved.  Data is processed in batches of injected
signals; each batch is split into smaller scenes, dedispersed, cropped, and
streamed into the output H5.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from astropy.io import fits

try:
    from scipy import ndimage as ndi
except ImportError:  # Remote generation can still run with the centered physical fallback.
    ndi = None


# 默认路径全部相对脚本所在目录（dataset_generation）。
# 需要的代码文件（simulation_utils.py、d-center-binary-core.py）和数据（rawdata/、channel_std 缓存）
# 都放在脚本同目录或其子目录下即可。命令行仍可覆盖。
_HERE = Path(__file__).resolve().parent      # 自包含：代码 + 数据都在脚本同目录或其子目录
DEFAULT_RAWDATA_DIR = _HERE / "rawdata"      # 原始 FAST 背景 FITS（可用 --rawdata-dir 覆盖）
DEFAULT_OUTPUT = _HERE / "multifit_tdm_200k.h5"
DEFAULT_WORK_DIR = _HERE                     # std 缓存等中间产物
# simulation_utils.py、d-center-binary-core.py 固定在脚本同目录 _HERE，不再单独配置 injection/runtime 路径。
DEFAULT_COUNT = 200_000  # h5 images = unique_signals(50000) * crops_per_signal(4)
DEFAULT_SIGNALS_PER_BATCH = 500
DEFAULT_SIGNALS_PER_SCENE = 3
DEFAULT_CROPS_PER_SIGNAL = 4
DEFAULT_MAX_OBJECTS_PER_IMAGE = 3
DEFAULT_FILE_FIRST = 11
DEFAULT_FILE_LAST = 275
DEFAULT_SEED = 20260614

# Signal distributions.  Deliberately broader than the paper injection run:
# this dataset trains detector robustness (wide DM/time/SNR generalization,
# many weak signals), not survey completeness.
#   - DM lower bound dropped 100 -> 5 so low-DM bowties actually populate the
#     small (512) DM-span crops; without this DM<100 was background-only.
#   - SNR floor 2 -> 1.5 and ceiling 100 -> 120 (log-uniform) to push more weak
#     signals.  Note: large DM/time crop spans further dilute the *relative* S/N
#     via area-resample, so "low DM x large span x low SNR" weak cases arise
#     automatically without coupling SNR to the (per-crop) span.
DM_RANGE_PC_CM3 = (5.0, 4000.0)
SNR_RANGE = (1.5, 120.0)
WIDTH_FWHM_MS_RANGE = (0.5, 50.0)
BANDWIDTH_FWHM_MHZ_RANGE = (50.0, 500.0)
SCATTERING_ZERO_FRACTION = 0.15
SCATTERING_TAU_1GHZ_MS_RANGE = (0.03, 80.0)
TAU_FREQ_INDEX = 4.0

CANVAS_DM_RANGE = 4096
CANVAS_DM_SCALE = 1.0
CANVAS_DM_OFFSET = 0.0
DEFAULT_TIME_FACTOR = 8.0
DEFAULT_SCENE_OUTPUT_DOWN = 8192
DEFAULT_TAIL_GUARD_DOWN = 4096
DEFAULT_RAW_CHUNK_DOWN = 4096

DM_CROP_SPANS = (512, 768, 1024, 1536, 2048, 3072, 4096)
TIME_CROP_WIDTHS = (1024, 2048, 4096, 8192, 16384)
SPLIT_NAMES = ("full", "lower", "upper")
MIN_SPLIT_EFFECTIVE_SNR = 1.5

# 标注策略（与训练 Dataset 的 keypoint 中心监督配套）：
# 1. 中心永远来自注入真值，避免 CenterNet heatmap 被响应亮度峰或 RFI 带偏。
# 2. 框以真值中心为几何中心、用真实形态/物理宽高，**允许越界**。训练端用 keypoint 跟踪
#    中心 + 角点重建 wh，越界部分天然无副作用，靠边目标因此能拿到与真实尺度相称的高斯半径。
# 3. 框大小来自局部形态分割，且用 nearest-truth ownership 避免相邻爆发互相借能量。
# 4. 形态分割失败时，用物理宽高作保底（仅当有效 S/N >= PHYSICAL_FALLBACK_MIN_SNR），
#    避免对图上看不见的弱信号硬标框；宽高只影响 Gaussian 半径，中心才是主监督。

# 物理 fallback 的有效 S/N 门槛（独立于可见性门槛 MIN_SPLIT_EFFECTIVE_SNR）。
# 低于该值且形态分割找不到响应的候选不硬标——它们在图上基本不可见，硬标会污染标签。
PHYSICAL_FALLBACK_MIN_SNR = 3.0

# 边缘样本占比（提升靠边/被切信号的检测能力）：约 4~5 成的样本把信号中心放到贴边区，
# 贴边范围见 EDGE_PIX_RANGE。
EDGE_X_FRACTION = 0.42
EDGE_Y_FRACTION = 0.48
EDGE_PIX_RANGE = (0.0, 56.0)
CENTER_PIX_RANGE = (96.0, 416.0)
CHANNEL_CHUNK = 128
MODEL_SUPPORT_SIGMA = 8.0


# ---------------------------------------------------------------------------
# 生成计划的数据结构：先把所有随机参数一次性固定，再逐 scene 读背景、注入、裁切
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SceneConfig:
    """单个 scene 的时间/频率下采样配置（同一次运行内所有 scene 共用）。"""
    scene_output_down: int    # 消色散输出的时间 bin 数（下采样后）
    input_down: int           # 输入窗口长度 = 输出 + 最大色散位移 + 尾部 guard
    max_dm_shift_down: int    # 最高 DM 在下采样时间单位下的色散位移
    down_time_rate: int       # 时间下采样率（raw -> down）
    down_freq_rate: int       # 频率下采样率（raw 通道 -> 下采样通道）
    time_factor: float        # runtime 的 time_factor 参数


@dataclass(frozen=True)
class SignalPlan:
    """一个注入信号的全部计划参数 + 它在某个 crop 中的裁切/落位信息（一份即一张图）。"""
    signal_index: int                    # 全局唯一信号序号
    injection_id: str                    # 形如 <run_label>_injNNNNN
    scene_local_index: int               # 在所属 scene 内的序号（0..signals_per_scene-1）
    toa_down: float                      # 参考频率到达时间（下采样时间 bin，连续）
    toa_raw: int                         # 同上，raw 采样单位（= round(toa_down*rate)）
    dm_pc_cm3: float                     # 色散量
    dm_index: float                      # DM 在 canvas 上的行坐标（scale=1,offset=0 时即 = DM）
    snr: float                           # 频率积分 nominal S/N
    width_ms_fwhm: float                 # 本征时间宽度 FWHM
    sigma_time_ms: float                 # = width_fwhm / 2.355
    bandwidth_mhz_fwhm: float            # 频率带宽 FWHM
    sigma_freq_mhz: float                # = bandwidth_fwhm / 2.355
    center_freq_mhz: float               # 频率包络中心
    scattering_ms_at_1ghz: float         # 1GHz 处散射时标（0 表示无散射）
    scattering_ms_at_highest_freq: float # 频带最高频处的散射时标（注入用参考）
    per_channel_peak_snr: float          # = snr / sqrt(有效通道数)，单通道峰值幅度
    effective_channels_fwhm: float       # 频率包络 >0.5 的有效通道数
    crop_dm_span: int                    # 本 crop 的 DM 跨度（源像素）
    crop_dm_start: int                   # 本 crop 的 DM 起点（canvas 行）
    crop_time_width: int                 # 本 crop 的时间跨度（源像素）
    crop_time_start: int                 # 本 crop 的时间起点（scene 输出列）
    split_index: int                     # 0=full / 1=lower / 2=upper
    split_name: str                      # split 的名字
    target_cx: float                     # 期望/实际落点 x（仅用于标注排序）
    target_cy: float                     # 期望/实际落点 y
    placement_mode_x: str                # x 方向布局模式（center/low_edge/high_edge）
    placement_mode_y: str                # y 方向布局模式


@dataclass(frozen=True)
class ScenePlan:
    """一个 scene：一段连续 raw 背景 + 注入其中的若干信号 + 由这些信号派生的所有 crop。"""
    batch_index: int                     # 所属 batch（仅用于日志/flush 节奏）
    scene_index: int                     # scene 序号
    start_global_raw: int                # 背景起始 raw 采样（已对齐下采样率）
    start_file_number: int               # 起始 FITS 文件编号
    input_raw_samples: int               # 输入窗口 raw 采样数
    signals: tuple[SignalPlan, ...]      # 本 scene 物理注入的信号（<= signals_per_scene）
    crops: tuple[SignalPlan, ...]        # 每个信号派生的 crop 视角（每个 crop 出一张图）


# ---------------------------------------------------------------------------
# 基础工具：文件编号解析、JSON 写出、动态 import 注入/runtime 模块
# ---------------------------------------------------------------------------
def file_number(path_or_stem: str) -> int:
    """从形如 ``*_Mxx_0123`` 的文件名里解析出 4 位 FITS 编号（如 0123 -> 123）。"""
    match = re.search(r"[-_]M\d+_(\d{4})$", Path(path_or_stem).stem)
    if not match:
        raise ValueError(f"Cannot parse Mxx file number from {path_or_stem}")
    return int(match.group(1))


def write_json(path: Path, payload: dict) -> None:
    """缩进 + key 排序写出单个 JSON（用于 config / inspect 汇总）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """逐行写 JSONL（每张图一行 metadata，便于流式读取/relabel）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_simulation_utils():
    """导入合并后的支撑库 simulation_utils，返回 (inj, model)——两者指向同一模块。

    simulation_utils 同时包含 FRB 模型（model 用：gaussian/scattered/DM_DELAY）和 FAST 背景工具
    （inj 用：list_background_fits/read_metadata/estimate_global_channel_std 等），所以
    返回两份同一引用即可，主流程里 ``inj.``/``model.`` 的写法保持不变。
    """
    sys.path.insert(0, str(_HERE))
    import simulation_utils  # noqa: PLC0415

    return simulation_utils, simulation_utils


def load_runtime_core():
    """从脚本同目录动态加载 DRAFTS runtime 核心；文件名含连字符无法常规 import 故按路径加载。"""
    core_path = _HERE / "d-center-binary-core.py"
    spec = importlib.util.spec_from_file_location("drafts_d_center_binary_core", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import runtime core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 随机采样和 crop 布局：只生成「计划」（参数 + 裁切窗口），不读大文件
# ---------------------------------------------------------------------------
def log_uniform(lo: float, hi: float, u: float) -> float:
    """对数均匀采样：给定 [0,1) 的 u，返回 [lo, hi] 上 log 均匀分布的值。"""
    return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))


def sample_pixel(rng: np.random.Generator, edge_fraction: float) -> tuple[float, str]:
    """采样信号在 512 图上的目标像素位置：以 edge_fraction 概率贴边（低/高边），否则居中区。

    返回 (像素坐标, 模式名)；贴边样本用来训练靠边 / 被边界切断的检测能力。
    """
    if rng.random() < edge_fraction:
        if rng.random() < 0.5:
            return float(rng.uniform(*EDGE_PIX_RANGE)), "low_edge"
        return float(511.0 - rng.uniform(*EDGE_PIX_RANGE)), "high_edge"
    return float(rng.uniform(*CENTER_PIX_RANGE)), "center"


def choose_crop_start(center_value: float, span: int, target_pixel: float, canvas_size: int) -> tuple[int, float]:
    """由「希望信号落到的目标像素」反推 crop 起点，并把起点夹到合法 canvas 范围内。

    返回 (crop 起点, 实际落点像素)。因为起点被夹紧，实际落点可能与目标像素略有出入，
    故一并返回实际值，供标注/排序使用。
    """
    scale = span / 512.0
    start = int(round(center_value - target_pixel * scale))
    start = max(0, min(canvas_size - span, start))
    actual_pixel = (center_value - start) / scale
    return start, float(actual_pixel)


def choose_split(center_freq: float, freq_min: float, freq_max: float, rng: np.random.Generator) -> int:
    """采样该 crop 使用哪个 detector 频带平面：0=full / 1=lower / 2=upper。

    55% 用 full；其余按信号中心频率偏向对应半带，增强模型对子带的鲁棒性。
    """
    if rng.random() < 0.55:
        return 0
    mid = 0.5 * (freq_min + freq_max)
    if center_freq < mid:
        return int(rng.choice([0, 1], p=[0.25, 0.75]))
    return int(rng.choice([0, 2], p=[0.25, 0.75]))


def sample_crop_fields(
    rng: np.random.Generator,
    dm_index: float,
    toa_down: float,
    center_freq: float,
    scene_cfg: SceneConfig,
    freq_min: float,
    freq_max: float,
) -> dict:
    """采样一个 crop 视角：DM 跨度 / 时间跨度 / 贴边落点 / split，并反推裁切起点。

    时间跨度只从能放进当前 scene 的候选里选。返回的 dict 直接喂给 SignalPlan 的 crop 字段。
    """
    crop_dm_span = int(rng.choice(DM_CROP_SPANS))
    valid_time_widths = [width for width in TIME_CROP_WIDTHS if width <= scene_cfg.scene_output_down]
    if not valid_time_widths:
        raise ValueError(f"No TIME_CROP_WIDTHS fit scene_output_down={scene_cfg.scene_output_down}")
    crop_time_width = int(rng.choice(valid_time_widths))
    target_y, y_mode = sample_pixel(rng, EDGE_Y_FRACTION)
    target_x, x_mode = sample_pixel(rng, EDGE_X_FRACTION)
    crop_dm_start, actual_y = choose_crop_start(dm_index, crop_dm_span, target_y, CANVAS_DM_RANGE)
    crop_time_start, actual_x = choose_crop_start(
        toa_down,
        crop_time_width,
        target_x,
        scene_cfg.scene_output_down,
    )
    split_index = choose_split(center_freq, freq_min, freq_max, rng)
    return {
        "crop_dm_span": crop_dm_span,
        "crop_dm_start": crop_dm_start,
        "crop_time_width": crop_time_width,
        "crop_time_start": crop_time_start,
        "split_index": split_index,
        "split_name": SPLIT_NAMES[split_index],
        "target_cx": float(actual_x),
        "target_cy": float(actual_y),
        "placement_mode_x": x_mode,
        "placement_mode_y": y_mode,
    }


def crop_variant(
    signal: SignalPlan,
    rng: np.random.Generator,
    scene_cfg: SceneConfig,
    freq_min: float,
    freq_max: float,
) -> SignalPlan:
    """复制一个信号，只重新采样它的 crop 字段——得到同一信号的另一个 DM/time 视角。"""
    values = asdict(signal)
    values.update(sample_crop_fields(
        rng,
        signal.dm_index,
        signal.toa_down,
        signal.center_freq_mhz,
        scene_cfg,
        freq_min,
        freq_max,
    ))
    return SignalPlan(**values)


def scene_time_reso_seconds(scene_cfg: SceneConfig) -> float:
    """scene guard 用的保守下采样时间分辨率：FAST search-mode TBIN≈49.152us × 下采样率。"""
    return 49.152e-6 * scene_cfg.down_time_rate


def args_guard_down(width_ms: float, scat_high_ms: float, scene_cfg: SceneConfig) -> int:
    """估计信号在时间方向占据的半宽（下采样 bin），用作 scene 边界 guard。

    只护 scene 边界；crop 边界**故意**允许切断信号以制造靠边样本。此处真实元信息尚未应用，
    用保守 TBIN 估计；真实 time_reso 由 rebuild_signal_timing 再夹一次。
    """
    approx_ms = MODEL_SUPPORT_SIGMA * (width_ms / 2.355 + 2.0 * scat_high_ms)
    return int(math.ceil((approx_ms / 1000.0) / scene_time_reso_seconds(scene_cfg))) + 512


def sample_signal_params(
    rng: np.random.Generator,
    signal_index: int,
    scene_local_index: int,
    scene_cfg: SceneConfig,
    freq_min: float,
    freq_max: float,
    model,
    run_label: str,
) -> SignalPlan:
    """采样一个信号的全部物理参数（DM/宽度/带宽/SNR/散射/中心频率/到达时间）+ 首个 crop 视角。

    SNR/宽度/带宽/散射用对数均匀，DM/中心频率用均匀。到达时间夹在 scene 边界 guard 内，
    保证整段响应（含色散拖尾）落在输出窗口里。
    """
    dm = float(rng.uniform(DM_RANGE_PC_CM3[0], DM_RANGE_PC_CM3[1]))
    dm_index = (dm - CANVAS_DM_OFFSET) / CANVAS_DM_SCALE
    width_ms = log_uniform(WIDTH_FWHM_MS_RANGE[0], WIDTH_FWHM_MS_RANGE[1], rng.random())
    bandwidth_mhz = log_uniform(BANDWIDTH_FWHM_MHZ_RANGE[0], BANDWIDTH_FWHM_MHZ_RANGE[1], rng.random())
    snr = log_uniform(SNR_RANGE[0], SNR_RANGE[1], rng.random())
    if rng.random() < SCATTERING_ZERO_FRACTION:
        scat_1ghz = 0.0
    else:
        scat_1ghz = log_uniform(
            SCATTERING_TAU_1GHZ_MS_RANGE[0],
            SCATTERING_TAU_1GHZ_MS_RANGE[1],
            rng.random(),
        )
    center_freq = float(rng.uniform(freq_min, freq_max))
    sigma_freq = bandwidth_mhz / 2.355
    sigma_time = width_ms / 2.355
    scat_high = scat_1ghz * (freq_max / 1000.0) ** (-TAU_FREQ_INDEX)

    guard = max(args_guard_down(width_ms, scat_high, scene_cfg), 256)
    earliest = guard
    latest = scene_cfg.scene_output_down - guard - 1
    toa_down = float(rng.uniform(earliest, latest))
    toa_raw = int(round(toa_down * scene_cfg.down_time_rate))

    freqs = np.linspace(freq_min, freq_max, 4096, dtype=np.float64)
    envelope = model.gaussian_profile(freqs, center_freq, sigma_freq)
    effective_channels = max(1.0, float(np.sum(envelope > 0.5)))
    per_channel_peak_snr = snr / math.sqrt(effective_channels)
    crop_fields = sample_crop_fields(rng, dm_index, toa_down, center_freq, scene_cfg, freq_min, freq_max)

    return SignalPlan(
        signal_index=signal_index,
        injection_id=f"{run_label}_inj{signal_index + 1:05d}",
        scene_local_index=scene_local_index,
        toa_down=toa_down,
        toa_raw=toa_raw,
        dm_pc_cm3=dm,
        dm_index=float(dm_index),
        snr=float(snr),
        width_ms_fwhm=float(width_ms),
        sigma_time_ms=float(sigma_time),
        bandwidth_mhz_fwhm=float(bandwidth_mhz),
        sigma_freq_mhz=float(sigma_freq),
        center_freq_mhz=float(center_freq),
        scattering_ms_at_1ghz=float(scat_1ghz),
        scattering_ms_at_highest_freq=float(scat_high),
        per_channel_peak_snr=float(per_channel_peak_snr),
        effective_channels_fwhm=float(effective_channels),
        **crop_fields,
    )


def rebuild_signal_timing(plan: SignalPlan, scene_cfg: SceneConfig, meta: dict) -> SignalPlan:
    """用真实元信息（真实 time_reso）重算时间 guard，把到达时间夹回 scene 内并修正 crop 起点。"""
    guard = max(
        int(math.ceil(
            MODEL_SUPPORT_SIGMA
            * (plan.sigma_time_ms + 2.0 * plan.scattering_ms_at_highest_freq)
            / 1000.0
            / (meta["time_reso_seconds"] * scene_cfg.down_time_rate)
        )) + 512,
        256,
    )
    toa_down = min(max(plan.toa_down, guard), scene_cfg.scene_output_down - guard - 1)
    crop_time_start, actual_x = choose_crop_start(
        toa_down,
        plan.crop_time_width,
        plan.target_cx,
        scene_cfg.scene_output_down,
    )
    values = asdict(plan)
    values.update({
        "toa_down": float(toa_down),
        "toa_raw": int(round(toa_down * scene_cfg.down_time_rate)),
        "crop_time_start": crop_time_start,
        "target_cx": float(actual_x),
    })
    return SignalPlan(**values)


def build_scene_plans(
    args: argparse.Namespace,
    files: list[Path],
    meta: dict,
    freqs_mhz: np.ndarray,
    scene_cfg: SceneConfig,
    model,
) -> list[ScenePlan]:
    """规划全部 scene：每个 scene 随机取一段背景、采样 signals_per_scene 个信号，
    每个信号派生 crops_per_signal 个 crop（每 crop 一张图），直到凑够 args.count 张图。

    只产出计划（参数 + 裁切窗口），不读背景 / 不注入 / 不消色散——那些在主循环里做。
    """
    rng = np.random.default_rng(args.seed)
    freq_min = float(np.nanmin(freqs_mhz))
    freq_max = float(np.nanmax(freqs_mhz))
    total_raw = len(files) * meta["samples_per_file"]
    input_raw = scene_cfg.input_down * scene_cfg.down_time_rate
    if input_raw >= total_raw:
        raise ValueError(f"Input scene {input_raw} raw samples is longer than selected data {total_raw}")

    scenes: list[ScenePlan] = []
    signal_index = 0
    image_count = 0
    while image_count < args.count:
        remaining_images = args.count - image_count
        scene_count = min(args.signals_per_scene, math.ceil(remaining_images / args.crops_per_signal))
        batch_index = image_count // args.signals_per_batch
        max_start = total_raw - input_raw - 1
        start_global = int(rng.integers(0, max_start // scene_cfg.down_time_rate + 1)) * scene_cfg.down_time_rate
        start_file = start_global // meta["samples_per_file"]
        signals = []
        crops = []
        for scene_local in range(scene_count):
            sig = sample_signal_params(
                rng,
                signal_index,
                scene_local,
                scene_cfg,
                freq_min,
                freq_max,
                model,
                args.run_label,
            )
            sig = rebuild_signal_timing(sig, scene_cfg, meta)
            signals.append(sig)
            signal_index += 1

        for sig in signals:
            for crop_idx in range(args.crops_per_signal):
                if image_count + len(crops) >= args.count:
                    break
                crops.append(sig if crop_idx == 0 else crop_variant(sig, rng, scene_cfg, freq_min, freq_max))
            if image_count + len(crops) >= args.count:
                break

        scenes.append(ScenePlan(
            batch_index=batch_index,
            scene_index=len(scenes),
            start_global_raw=start_global,
            start_file_number=file_number(files[start_file].stem),
            input_raw_samples=input_raw,
            signals=tuple(signals),
            crops=tuple(crops),
        ))
        image_count += len(crops)
    return scenes


# ---------------------------------------------------------------------------
# 真实 FAST 背景读取、raw 域注入、时间下采样
# ---------------------------------------------------------------------------
def estimate_channel_std(inj, files: list[Path], meta: dict, args: argparse.Namespace) -> np.ndarray:
    """估计每通道背景噪声 std（注入幅度的标度）；优先读缓存，没有就从少量文件估计并缓存。"""
    cache = args.work_dir / "channel_std_raw8.npy"
    if args.std_cache is not None:
        cache = args.std_cache
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] != meta["nchan"]:
            raise ValueError(f"Bad std cache shape {arr.shape}; expected {meta['nchan']}")
        return arr.astype(np.float32)
    sample_files = files if args.std_file_limit <= 0 else files[: args.std_file_limit]
    std = inj.estimate_global_channel_std(
        sample_files,
        meta,
        max_samples_per_file=args.std_samples_per_file,
        workers=args.std_workers,
    ).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, std)
    return std


def load_raw_stokes_samples(files: list[Path], meta: dict, start_global: int, length: int) -> np.ndarray:
    """从（可能跨多个）FITS 文件读出连续 raw 时间窗口，前两路偏振取平均作为背景 [length, nchan]。"""
    out = np.empty((length, meta["nchan"]), dtype=np.float32)
    written = 0
    sample = start_global
    while written < length:
        file_idx = sample // meta["samples_per_file"]
        local = sample - file_idx * meta["samples_per_file"]
        take = min(length - written, meta["samples_per_file"] - local)
        with fits.open(files[file_idx], memmap=True, lazy_load_hdus=True) as hdul:
            arr = hdul[1].data["DATA"].reshape(meta["samples_per_file"], meta["npol"], meta["nchan"])
            sl = arr[local:local + take]
            out[written:written + take] = 0.5 * (
                sl[:, 0, :].astype(np.float32) + sl[:, 1, :].astype(np.float32)
            )
        written += take
        sample += take
    return out


def load_downsampled_stokes_window(
    files: list[Path],
    meta: dict,
    start_global: int,
    scene_cfg: SceneConfig,
    signals: tuple[SignalPlan, ...],
    freqs_mhz: np.ndarray,
    chan_std: np.ndarray,
    model,
    chunk_down: int,
) -> np.ndarray:
    """分块读 raw 背景 → 块内逐信号注入 → 每 down_time_rate 个 raw 采样求平均，
    流式拼出整段下采样 time-freq [input_down, nchan]，避免一次性把超长 raw 读进内存。
    """
    if start_global % scene_cfg.down_time_rate != 0:
        raise ValueError("start_global must be aligned to down_time_rate")
    out = np.empty((scene_cfg.input_down, meta["nchan"]), dtype=np.float32)
    for down_start in range(0, scene_cfg.input_down, chunk_down):
        down_stop = min(scene_cfg.input_down, down_start + chunk_down)
        raw_start = start_global + down_start * scene_cfg.down_time_rate
        raw_len = (down_stop - down_start) * scene_cfg.down_time_rate
        raw = load_raw_stokes_samples(files, meta, raw_start, raw_len)
        chunk_local_raw = down_start * scene_cfg.down_time_rate
        for signal in signals:
            inject_signal_raw_stokes(raw, chunk_local_raw, signal, freqs_mhz, chan_std, meta, model)
        out[down_start:down_stop] = raw.reshape(
            down_stop - down_start,
            scene_cfg.down_time_rate,
            meta["nchan"],
        ).mean(axis=1)
    return out


def support_samples_raw(signal: SignalPlan, freqs_mhz: np.ndarray, meta: dict) -> int:
    """信号在时间方向需要覆盖的半窗（raw 采样）：取本征宽度 + 最低频处散射尾的若干 sigma。"""
    ref = float(np.nanmax(freqs_mhz))
    min_freq = float(np.nanmin(freqs_mhz))
    max_scat_ms = signal.scattering_ms_at_highest_freq * (min_freq / ref) ** (-TAU_FREQ_INDEX)
    support_s = max(0.08, MODEL_SUPPORT_SIGMA * (signal.sigma_time_ms + max_scat_ms) / 1000.0)
    return int(math.ceil(support_s / meta["time_reso_seconds"]))


def inject_signal_raw_stokes(
    raw_stokes: np.ndarray,
    chunk_local_raw_start: int,
    signal: SignalPlan,
    freqs_mhz: np.ndarray,
    chan_std: np.ndarray,
    meta: dict,
    model,
) -> None:
    """把一个 FRB 信号原位加进 raw time-freq 块（就地修改 raw_stokes）。

    逐通道：高斯频率包络 × 散射高斯时间 profile，按 DM 给每通道的到达时间加色散延迟，
    幅度 = per_channel_peak_snr × envelope × pulse × 每通道 std，最后并入背景并 clip 到 [0,255]。
    只在信号实际覆盖的时间/通道范围内计算，块外/包络可忽略处直接跳过。
    """
    n_time = int(raw_stokes.shape[0])
    nchan = int(meta["nchan"])
    ref_freq = float(np.nanmax(freqs_mhz))
    sigma_bins = (signal.sigma_time_ms / 1000.0) / meta["time_reso_seconds"]
    tau_ref_bins = (signal.scattering_ms_at_highest_freq / 1000.0) / meta["time_reso_seconds"]
    support = support_samples_raw(signal, freqs_mhz, meta)

    all_delays = model.DM_DELAY_SECONDS * signal.dm_pc_cm3 * (freqs_mhz ** -2 - ref_freq ** -2)
    all_centers = signal.toa_raw + all_delays / meta["time_reso_seconds"]
    local_start = max(0, int(math.floor(float(np.nanmin(all_centers)) - support)) - chunk_local_raw_start)
    local_stop = min(n_time, int(math.ceil(float(np.nanmax(all_centers)) + support)) + 1 - chunk_local_raw_start)
    if local_stop <= local_start:
        return

    for chan_start in range(0, nchan, CHANNEL_CHUNK):
        chan_stop = min(chan_start + CHANNEL_CHUNK, nchan)
        freqs = freqs_mhz[chan_start:chan_stop]
        envelope = model.gaussian_profile(freqs, signal.center_freq_mhz, signal.sigma_freq_mhz).astype(np.float32)
        if float(np.nanmax(envelope)) <= 1e-5:
            continue
        delays = model.DM_DELAY_SECONDS * signal.dm_pc_cm3 * (freqs ** -2 - ref_freq ** -2)
        centers = signal.toa_raw + delays / meta["time_reso_seconds"]
        chunk_start = max(local_start, int(math.floor(float(np.nanmin(centers)) - support)) - chunk_local_raw_start)
        chunk_stop = min(local_stop, int(math.ceil(float(np.nanmax(centers)) + support)) + 1 - chunk_local_raw_start)
        if chunk_stop <= chunk_start:
            continue

        times = chunk_local_raw_start + np.arange(chunk_start, chunk_stop, dtype=np.float64)
        tau_by_freq = tau_ref_bins * (freqs / ref_freq) ** (-TAU_FREQ_INDEX)
        pulse = model.scattered_gaussian_profile(times, centers, sigma_bins, tau_by_freq).astype(np.float32)
        delta = signal.per_channel_peak_snr * envelope[:, None] * pulse
        delta = (delta.T * chan_std[chan_start:chan_stop][None, :]).astype(np.float32)
        raw_stokes[chunk_start:chunk_stop, chan_start:chan_stop] = np.clip(
            raw_stokes[chunk_start:chunk_stop, chan_start:chan_stop] + delta,
            0,
            255,
        )


# ---------------------------------------------------------------------------
# 频率下采样 + 消色散准备
# ---------------------------------------------------------------------------
def make_process_config(core, time_factor: float):
    """构造 runtime 的消色散 ProcessConfig，固定到 0-4096 DM canvas。"""
    return core.ProcessConfig(
        dm_range=CANVAS_DM_RANGE,
        dm_scale=CANVAS_DM_SCALE,
        dm_offset=CANVAS_DM_OFFSET,
        dm_threshold=90.0,
        block_size=4096,
        dm_span=1024,
        dm_stride=512,
        time_factor=time_factor,
        save_npy=False,
        save_plot=False,
    )


def build_scene_config(core, first_file: Path, args: argparse.Namespace) -> SceneConfig:
    """读首文件元信息，算出最高 DM 的色散位移，推导输入窗口长度，组装 SceneConfig。"""
    core.get_obparams(str(first_file), args.time_factor)
    freq_down = np.mean(
        core.freq.reshape(core.freq_reso // core.down_freq_rate, core.down_freq_rate),
        axis=1,
    ).astype(np.float32)
    dm_max = CANVAS_DM_OFFSET + CANVAS_DM_SCALE * (CANVAS_DM_RANGE - 1)
    delay = 4.15e3 * dm_max * (freq_down.min() ** -2 - freq_down.max() ** -2)
    max_shift = int(math.ceil(delay / (core.time_reso * core.down_time_rate)))
    input_down = int(args.scene_output_down + max_shift + args.tail_guard_down)
    return SceneConfig(
        scene_output_down=int(args.scene_output_down),
        input_down=input_down,
        max_dm_shift_down=max_shift,
        down_time_rate=int(core.down_time_rate),
        down_freq_rate=int(core.down_freq_rate),
        time_factor=float(args.time_factor),
    )


def stokes_to_freq_down(stokes: np.ndarray, core, scene_cfg: SceneConfig) -> np.ndarray:
    """频率方向每 down_freq_rate 个通道求平均：[input_down, nchan] -> [input_down, nchan/rate]。"""
    return np.mean(
        stokes.reshape(scene_cfg.input_down, core.freq_reso // scene_cfg.down_freq_rate, scene_cfg.down_freq_rate),
        axis=2,
    ).astype(np.float32)


def prepare_dedispersion(core, first_file: Path, args: argparse.Namespace, scene_cfg: SceneConfig):
    """预建 runtime 的消色散缓存（DM 试探表等），整个运行复用一次。"""
    pc = make_process_config(core, args.time_factor)
    core.get_obparams(str(first_file), args.time_factor)
    return core._build_dedispersion_cache(
        CANVAS_DM_RANGE,
        scene_cfg.scene_output_down,
        scene_cfg.input_down,
        pc,
    )


# ---------------------------------------------------------------------------
# 标注生成：在当前 512x512 detector 图里给每个可见注入信号生成检测框
#   中心 = 注入真值（signal_center_in_crop）；框 = 形态分割尺寸优先、物理宽高兜底，
#   一律以真值中心为几何中心、允许越界（centered_box）。
# ---------------------------------------------------------------------------
def signal_center_in_crop(signal: SignalPlan, crop_owner: SignalPlan) -> tuple[float, float] | None:
    """把注入信号的 (toa_down, dm_index) 映射到当前 crop 的 512 像素坐标；中心出界返回 None。"""
    # 线性边沿映射：crop 内源坐标 [start, start+span) -> 输出像素 [0, 512)。
    # 注意：area-resample 的像素中心约定会带来 <0.5px 的系统性偏移（随 span 增大趋近
    # 0.5px），远小于评估匹配阈值 dist_thr=8px，故此处不做半像素修正；若要严格对齐
    # d_dm_time_g 的轴约定，需对照 runtime 实测响应峰再决定是否加 (i+0.5)*512/N-0.5。
    x = (signal.toa_down - crop_owner.crop_time_start) / crop_owner.crop_time_width * 512.0
    y = (signal.dm_index - crop_owner.crop_dm_start) / crop_owner.crop_dm_span * 512.0
    if x < 0.0 or x >= 512.0 or y < 0.0 or y >= 512.0:
        return None
    return float(x), float(y)


def centered_box(cx: float, cy: float, width: float, height: float) -> list[float] | None:
    """以真值中心为几何中心的 COCO 框，**允许越界**（不裁剪到图内）。

    中心必须落在图像 [0,512) 内（否则目标不可见，返回 None）；宽高至少 1px。输出真实尺度
    的（可越界）框、且 x+w/2 严格等于真值中心——训练 Dataset 用 keypoint 跟踪中心 + 角点
    重建 wh，能直接消化越界框，靠边目标因此能拿到与真实尺度相称的高斯半径。
    """
    if cx < 0.0 or cx >= 512.0 or cy < 0.0 or cy >= 512.0:
        return None
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    return [float(cx - w / 2.0), float(cy - h / 2.0), w, h]


def clipped_coco_extent(left: float, top: float, right: float, bottom: float) -> list[float] | None:
    """把形态分割测到的像素范围裁到图内——仅用于*测量*响应实际占据的像素跨度。"""
    left = float(np.clip(left, 0.0, 511.0))
    top = float(np.clip(top, 0.0, 511.0))
    right = float(np.clip(right, 1.0, 512.0))
    bottom = float(np.clip(bottom, 1.0, 512.0))
    if right - left < 1.0:
        right = min(512.0, left + 1.0)
        left = max(0.0, right - 1.0)
    if bottom - top < 1.0:
        bottom = min(512.0, top + 1.0)
        top = max(0.0, bottom - 1.0)
    if right - left < 1.0 or bottom - top < 1.0:
        return None
    return [left, top, right - left, bottom - top]


def split_snr_fraction(signal: SignalPlan, owner_split: int, freq_min: float, freq_max: float) -> float:
    """返回信号频谱落在当前 detector split 中的 L2 能量比例。"""
    if owner_split == 0:
        return 1.0
    freqs = np.linspace(freq_min, freq_max, 1024, dtype=np.float64)
    envelope = np.exp(-0.5 * ((freqs - signal.center_freq_mhz) / max(signal.sigma_freq_mhz, 1e-6)) ** 2)
    denom = float(np.sqrt(np.sum(envelope * envelope)))
    if denom <= 0.0:
        return 0.0
    mid = 0.5 * (freq_min + freq_max)
    if owner_split == 1:
        mask = freqs <= mid
    elif owner_split == 2:
        mask = freqs >= mid
    else:
        return 1.0
    return float(np.sqrt(np.sum(envelope[mask] * envelope[mask])) / denom)


def shrink_box_for_visible_fraction(
    box: list[float],
    center: tuple[float, float],
    visible_fraction: float,
) -> list[float] | None:
    """子带只看到谱尾时，把框按可见能量比例（中心不变）缩小，不沿用 full-band 的完整尺寸。"""
    if visible_fraction >= 0.98:
        return box
    _, _, w, h = box
    scale = float(np.clip(0.55 + 0.45 * np.sqrt(max(visible_fraction, 0.0)), 0.55, 1.0))
    return centered_box(center[0], center[1], w * scale, h * scale)


def signal_bbox(signal: SignalPlan, crop_owner: SignalPlan, meta: dict, scene_cfg: SceneConfig) -> tuple[float, float]:
    """形态分割失败时使用的保底物理框宽高。"""
    dt_ms = crop_owner.crop_time_width * scene_cfg.down_time_rate * meta["time_reso_seconds"] * 1000.0 / 512.0
    temporal_ms = signal.width_ms_fwhm + 2.5 * signal.scattering_ms_at_highest_freq
    w = temporal_ms / max(dt_ms, 1e-6) + 16.0
    dm_per_pixel = crop_owner.crop_dm_span / 512.0
    h = 520.0 / (max(signal.bandwidth_mhz_fwhm, 25.0) * max(dm_per_pixel, 1e-6)) + 16.0
    w = float(np.clip(w, 14.0, 120.0))
    h = float(np.clip(h, 14.0, 140.0))
    return float(w), float(h)


def component_box(mask: np.ndarray, label_id: int, x0: int, y0: int) -> list[float] | None:
    """取某个连通域的外接框（含小 margin），(x0,y0) 是局部 patch 在整图中的偏移。"""
    yy, xx = np.nonzero(mask == label_id)
    if yy.size == 0:
        return None
    left = float(xx.min() + x0)
    right = float(xx.max() + x0 + 1)
    top = float(yy.min() + y0)
    bottom = float(yy.max() + y0 + 1)
    width = right - left
    height = bottom - top
    mx = max(3.0, 0.05 * width)
    my = max(3.0, 0.08 * height)
    return clipped_coco_extent(left - mx, top - my, right + mx, bottom + my)


def local_stats(smooth: np.ndarray) -> tuple[float, float, float]:
    """用 patch 四周边缘像素估计背景分布，返回 (中位数, P85, P90) 作为阈值/噪声标度。"""
    edge = max(1, min(smooth.shape) // 8)
    background = np.concatenate(
        [
            smooth[:edge, :].ravel(),
            smooth[-edge:, :].ravel(),
            smooth[:, :edge].ravel(),
            smooth[:, -edge:].ravel(),
        ]
    )
    med = float(np.median(background))
    p85 = float(np.percentile(background, 85))
    p90 = float(np.percentile(background, 90))
    return med, p85, p90


def ownership_mask(
    shape: tuple[int, int],
    x0: int,
    y0: int,
    center: tuple[float, float],
    other_centers: list[tuple[float, float]],
) -> np.ndarray:
    """最近真值归属 mask：只保留「离本目标中心比离任何其它目标更近」的像素，
    防止多目标 scene 里相邻爆发在阈值分割时互相借能量、连成一个大框。
    """
    if not other_centers:
        return np.ones(shape, dtype=bool)
    yy, xx = np.indices(shape, dtype=np.float32)
    xx = xx + float(x0)
    yy = yy + float(y0)
    cx, cy = center
    own = (xx - cx) ** 2 + (yy - cy) ** 2
    keep = np.ones(shape, dtype=bool)
    for ox, oy in other_centers:
        other = (xx - ox) ** 2 + (yy - oy) ** 2
        keep &= own <= other * 0.92
    return keep


def clamp_dims_to_limits(width: float, height: float, max_w: float, max_h: float) -> tuple[float, float]:
    """把宽高夹到 [2, max] 区间。"""
    return min(max(width, 2.0), max_w), min(max(height, 2.0), max_h)


def choose_component(
    labels: np.ndarray,
    count: int,
    seed: tuple[int, int],
    sx: int,
    sy: int,
    max_seed_distance: float = 28.0,
) -> int:
    """选连通域：种子像素所在连通域优先；否则取离种子最近（且足够近）的连通域，0 表示没有。"""
    seed_label = int(labels[seed])
    if seed_label > 0:
        return seed_label

    best: tuple[float, int] | None = None
    for label_id in range(1, count + 1):
        yy, xx = np.nonzero(labels == label_id)
        if yy.size < 3:
            continue
        min_dist = float(np.sqrt(np.min((xx - sx) ** 2 + (yy - sy) ** 2)))
        if min_dist > max_seed_distance:
            continue
        score = min_dist - 0.01 * float(np.sqrt(yy.size))
        if best is None or score < best[0]:
            best = (score, label_id)
    return 0 if best is None else best[1]


def morphology_box(
    image: np.ndarray,
    center: tuple[float, float],
    effective_snr: float,
    other_centers: list[tuple[float, float]],
    min_size: tuple[float, float],
) -> list[float] | None:
    """从实际 detector 图中提取目标框。

    这个函数只回答一个问题：真值中心附近是否能在当前图像中稳定分割出亮响应。
    它不看 nominal S/N 门槛；低 split S/N 的候选如果图上真的可见，也应该被标。
    """
    if ndi is None:
        return None

    # 只看真值中心附近的局部窗口，避免远处强 RFI 或背景梯度影响标注；
    # 多目标时再用 nearest-truth ownership mask 防止不同 burst 互相借能量。
    height, width = image.shape
    cx, cy = center
    half_w = 150.0
    half_h = 110.0
    nearest = min((math.hypot(cx - ox, cy - oy) for ox, oy in other_centers), default=999.0)
    if nearest < 180.0:
        half_w = max(36.0, min(half_w, 0.58 * nearest))
        half_h = max(30.0, min(half_h, 0.50 * nearest))
    x0 = max(0, int(np.floor(cx - half_w)))
    x1 = min(width, int(np.ceil(cx + half_w + 1)))
    y0 = max(0, int(np.floor(cy - half_h)))
    y1 = min(height, int(np.ceil(cy + half_h + 1)))
    if x1 <= x0 or y1 <= y0:
        return None

    patch = image[y0:y1, x0:x1].astype(np.float32, copy=False)
    smooth = ndi.gaussian_filter(patch, sigma=1.15)
    sx = int(np.clip(round(center[0]) - x0, 0, smooth.shape[1] - 1))
    sy = int(np.clip(round(center[1]) - y0, 0, smooth.shape[0] - 1))

    # 在真值中心附近找最亮点作为连通域种子，允许响应峰与真值中心有小偏移。
    seed_radius = 15
    seed_y = slice(max(0, sy - seed_radius), min(smooth.shape[0], sy + seed_radius + 1))
    seed_x = slice(max(0, sx - seed_radius), min(smooth.shape[1], sx + seed_radius + 1))
    seed_patch = smooth[seed_y, seed_x]
    if seed_patch.size == 0:
        return None
    seed_local = np.unravel_index(int(np.argmax(seed_patch)), seed_patch.shape)
    seed = (int(seed_local[0] + seed_y.start), int(seed_local[1] + seed_x.start))

    med, p85, p90 = local_stats(smooth)
    contrast = float(smooth[seed]) - med
    noise_scale = max(p90 - med, 1e-4)
    local_snr = contrast / noise_scale
    if contrast < max(0.035, 1.05 * noise_scale) or local_snr < 1.25:
        return None

    owner_mask = ownership_mask(smooth.shape, x0, y0, center, other_centers)
    patch_pixels = smooth.shape[0] * smooth.shape[1]
    min_w, min_h = min_size
    max_w = min(230.0, max(42.0, 7.5 * min_w))
    max_h = min(210.0, max(42.0, 7.5 * min_h))
    if nearest < 999.0:
        max_w = min(max_w, max(34.0, 0.88 * nearest))
        max_h = min(max_h, max(30.0, 0.78 * nearest))

    if local_snr < 2.2 or effective_snr < 3.0:
        threshold_fracs = (0.64, 0.72, 0.80, 0.88)
    elif local_snr < 4.0 or effective_snr < 8.0:
        threshold_fracs = (0.56, 0.64, 0.74, 0.84)
    else:
        threshold_fracs = (0.44, 0.52, 0.62, 0.74)

    best: list[float] | None = None
    for frac in threshold_fracs:
        threshold = max(p85, med + frac * contrast)
        if local_snr < 2.2:
            threshold = max(threshold, p90)
        mask = (smooth >= threshold) & owner_mask
        mask = ndi.binary_closing(mask, structure=np.ones((2, 3), dtype=bool), iterations=1)
        labels, count = ndi.label(mask)
        label_id = choose_component(labels, int(count), seed, sx, sy)
        if label_id <= 0:
            continue
        yy, xx = np.nonzero(labels == label_id)
        pixels = int(yy.size)
        raw_width = float(xx.max() - xx.min() + 1)
        raw_height = float(yy.max() - yy.min() + 1)
        if pixels > 0.18 * patch_pixels and frac < threshold_fracs[-1]:
            continue
        if raw_width > max_w * 1.35 or raw_height > max_h * 1.35:
            continue
        aspect = raw_width / max(raw_height, 1.0)
        if aspect > 9.0 or aspect < 0.12:
            continue

        box = component_box(labels, label_id, x0, y0)
        if box is None:
            continue
        _, _, bw, bh = box
        bw, bh = clamp_dims_to_limits(max(bw, min_w), max(bh, min_h), max_w, max_h)
        best = centered_box(cx, cy, bw, bh)
        if best is not None:
            return best
    return best


def annotation_for_signal(
    signal: SignalPlan,
    crop_owner: SignalPlan,
    image: np.ndarray,
    meta: dict,
    scene_cfg: SceneConfig,
    freq_min: float,
    freq_max: float,
    min_effective_snr: float,
    physical_fallback_min_snr: float,
    other_centers: list[tuple[float, float]],
) -> tuple[list[float], float] | None:
    """为一个物理注入信号生成当前 crop 中的检测框。

    返回 `(box, dist)`，dist 用来把 crop owner 排最前、其余同 scene 目标按离 owner 的距离排序。
    流程：有效 S/N >= min_effective_snr 且可见才考虑 → 形态分割优先取尺寸 → 失败且有效 S/N
    >= physical_fallback_min_snr 才用物理宽高兜底 → 子带按可见能量缩放。框一律以真值中心
    为几何中心、允许越界。
    """
    center = signal_center_in_crop(signal, crop_owner)
    if center is None:
        return None

    visible_fraction = split_snr_fraction(signal, crop_owner.split_index, freq_min, freq_max)
    effective_snr = signal.snr * visible_fraction
    if visible_fraction < 0.04 or effective_snr < min_effective_snr:
        return None

    # 第一优先级：实际 time-DM 图中能稳定分割出亮响应。
    min_size = signal_bbox(signal, crop_owner, meta, scene_cfg)
    box = morphology_box(image, center, effective_snr, other_centers, min_size)

    # 第二优先级：有效 S/N 足够高（>= physical_fallback_min_snr）但形态分割失败时，
    # 才用保底物理宽高；更弱的候选不硬标，只接受形态分割能看到的响应。
    if box is None and effective_snr >= physical_fallback_min_snr:
        w, h = min_size
        box = centered_box(center[0], center[1], max(w, 18.0), max(h, 18.0))

    if box is not None:
        box = shrink_box_for_visible_fraction(box, center, visible_fraction)
    if box is None:
        return None

    dist = float((center[0] - crop_owner.target_cx) ** 2 + (center[1] - crop_owner.target_cy) ** 2)
    return box, dist


def crop_scene_images(
    core,
    dm_time_gpu,
    crop_plans: tuple[SignalPlan, ...],
    injected_signals: tuple[SignalPlan, ...],
    meta: dict,
    scene_cfg: SceneConfig,
    freq_min: float,
    freq_max: float,
    min_effective_snr: float,
    physical_fallback_min_snr: float,
    max_objects_per_image: int,
):
    """把一个 scene 的所有 crop 从 GPU 上的 dm_time canvas 裁出、area-resample 成 512²、批归一化，
    再为每张图标注框（含同 scene 其它信号），每图最多保留 max_objects_per_image 个、owner 排最前。

    返回 (images[N,512,512], 每图的框列表)。
    """
    torch = core.torch
    crops = []
    for signal in crop_plans:
        slab = dm_time_gpu[
            signal.split_index,
            signal.crop_dm_start: signal.crop_dm_start + signal.crop_dm_span,
            signal.crop_time_start: signal.crop_time_start + signal.crop_time_width,
        ]
        tensor = torch.as_tensor(slab, device="cuda").float()[None, None, :, :]
        crop = torch.nn.functional.interpolate(tensor, size=(512, 512), mode="area")[0, 0]
        crops.append(crop)
    batch = torch.stack(crops, dim=0)
    batch = core._normalize_batch_gpu(batch)
    images = batch.detach().float().cpu().numpy().astype(np.float32)

    annotations_by_image: list[list[list[float]]] = []
    for image, owner in zip(images, crop_plans):
        candidates = []
        centers: dict[int, tuple[float, float]] = {}
        for signal in injected_signals:
            center = signal_center_in_crop(signal, owner)
            if center is None:
                continue
            visible_fraction = split_snr_fraction(signal, owner.split_index, freq_min, freq_max)
            effective_snr = signal.snr * visible_fraction
            if visible_fraction < 0.04 or effective_snr < min_effective_snr:
                continue
            centers[signal.signal_index] = center

        for signal in injected_signals:
            if signal.signal_index not in centers:
                continue
            other_centers = [
                center for idx, center in centers.items() if idx != signal.signal_index
            ]
            box = annotation_for_signal(
                signal,
                owner,
                image,
                meta,
                scene_cfg,
                freq_min,
                freq_max,
                min_effective_snr,
                physical_fallback_min_snr,
                other_centers,
            )
            if box is not None:
                dist = -1.0 if signal.signal_index == owner.signal_index else box[1]
                candidates.append((dist, box[0]))
        candidates.sort(key=lambda item: item[0])
        annotations_by_image.append([box for _, box in candidates[:max_objects_per_image]])
    return images, annotations_by_image


# ---------------------------------------------------------------------------
# H5 和配套 JSON 输出
# ---------------------------------------------------------------------------
def create_h5(path: Path, image_count: int, gzip_level: int):
    """新建输出 H5，预分配 images / original_* 数据集（annotations 在最后一次性写）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    h5 = h5py.File(path, "w")
    compression = "gzip" if gzip_level > 0 else None
    opts = gzip_level if gzip_level > 0 else None
    h5.create_dataset(
        "images",
        shape=(image_count, 512, 512),
        dtype="float32",
        chunks=(1, 512, 512),
        compression=compression,
        compression_opts=opts,
    )
    string_dtype = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset("original_filename", shape=(image_count,), dtype=string_dtype)
    h5.create_dataset("original_path", shape=(image_count,), dtype=string_dtype)
    h5.create_dataset("original_slice", shape=(image_count,), dtype="int32")
    return h5


def write_annotations(h5, annotation_rows: list[list[float]], gzip_level: int) -> None:
    """把全部标注行 [image_idx, left, top, w, h] 一次性写成 (N,5) annotations 数据集。"""
    compression = "gzip" if gzip_level > 0 else None
    opts = gzip_level if gzip_level > 0 else None
    data = np.asarray(annotation_rows, dtype=np.float32)
    if data.size == 0:
        data = np.empty((0, 5), dtype=np.float32)
    h5.create_dataset(
        "annotations",
        data=data,
        dtype="float32",
        chunks=(min(max(len(data), 1), 4096), 5),
        compression=compression,
        compression_opts=opts,
    )


def write_annotations_json(path: Path, image_count: int, annotation_rows: list[list[float]]) -> None:
    """写配套的 per-image 标注 JSON；无框的图用占位行 [-1,-1,-1,-1]（仅 JSON 有占位，H5 没有）。"""
    by_image = {str(i): [[-1.0, -1.0, -1.0, -1.0]] for i in range(image_count)}
    for row in annotation_rows:
        image_idx = int(row[0])
        box = [float(v) for v in row[1:5]]
        if by_image[str(image_idx)] == [[-1.0, -1.0, -1.0, -1.0]]:
            by_image[str(image_idx)] = []
        by_image[str(image_idx)].append(box)
    path.write_text(json.dumps(by_image, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """解析命令行参数；远端生成通常只覆盖路径/规模/scene 长度，不需要改代码默认值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rawdata-dir", type=Path, default=DEFAULT_RAWDATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-label", default="objdet_multifit")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument(
        "--unique-signals",
        type=int,
        default=None,
        help="Number of unique injected signals. Overrides --count as unique_signals * crops_per_signal.",
    )
    parser.add_argument("--signals-per-batch", type=int, default=DEFAULT_SIGNALS_PER_BATCH)
    parser.add_argument("--signals-per-scene", type=int, default=DEFAULT_SIGNALS_PER_SCENE)
    parser.add_argument("--crops-per-signal", type=int, default=DEFAULT_CROPS_PER_SIGNAL)
    parser.add_argument("--max-objects-per-image", type=int, default=DEFAULT_MAX_OBJECTS_PER_IMAGE)
    parser.add_argument("--file-first", type=int, default=DEFAULT_FILE_FIRST)
    parser.add_argument("--file-last", type=int, default=DEFAULT_FILE_LAST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--time-factor", type=float, default=DEFAULT_TIME_FACTOR)
    parser.add_argument("--scene-output-down", type=int, default=DEFAULT_SCENE_OUTPUT_DOWN)
    parser.add_argument("--tail-guard-down", type=int, default=DEFAULT_TAIL_GUARD_DOWN)
    parser.add_argument("--raw-chunk-down", type=int, default=DEFAULT_RAW_CHUNK_DOWN)
    parser.add_argument("--gzip-level", type=int, default=4)
    parser.add_argument("--std-file-limit", type=int, default=8)
    parser.add_argument("--std-samples-per-file", type=int, default=16384)
    parser.add_argument("--std-workers", type=int, default=4)
    parser.add_argument("--std-cache", type=Path, default=None)
    parser.add_argument("--min-split-effective-snr", type=float, default=MIN_SPLIT_EFFECTIVE_SNR,
                        help="可见性门槛：低于此有效 S/N 的候选不参与标注")
    parser.add_argument("--physical-fallback-min-snr", type=float, default=PHYSICAL_FALLBACK_MIN_SNR,
                        help="物理 fallback 门槛：形态分割失败时，仅当有效 S/N >= 此值才硬标物理框")
    parser.add_argument("--validate-headers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """主流程：解析参数 → 规划全部 scene → 写 config/metadata →（非 dry-run 则）
    逐 scene 读背景、注入、下采样、消色散、裁切标注，流式写入 H5，最后写 annotations + JSON。
    """
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.unique_signals is not None:
        if args.unique_signals < 1:
            raise SystemExit("--unique-signals must be positive")
        args.count = int(args.unique_signals) * int(args.crops_per_signal)
    if args.signals_per_batch < 1 or args.signals_per_scene < 1:
        raise SystemExit("--signals-per-batch and --signals-per-scene must be positive")
    if args.crops_per_signal < 1:
        raise SystemExit("--crops-per-signal must be positive")
    if args.max_objects_per_image < 1:
        raise SystemExit("--max-objects-per-image must be positive")
    if args.signals_per_scene > args.max_objects_per_image:
        raise SystemExit("--signals-per-scene cannot exceed --max-objects-per-image")

    inj, model = load_simulation_utils()
    core = load_runtime_core()
    files = inj.list_background_fits(args.rawdata_dir, args.file_first, args.file_last)
    meta = inj.read_metadata(files[0])
    if args.validate_headers:
        inj.validate_consistent_headers(files, meta)
    freqs_mhz = inj.frequency_axis(meta)
    freq_min = float(np.nanmin(freqs_mhz))
    freq_max = float(np.nanmax(freqs_mhz))
    scene_cfg = build_scene_config(core, files[0], args)
    scenes = build_scene_plans(args, files, meta, freqs_mhz, scene_cfg, model)
    unique_signal_count = int(sum(len(scene.signals) for scene in scenes))

    config = {
        "run_label": args.run_label,
        "count": args.count,
        "count_units": "h5_images",
        "requested_unique_signals": args.unique_signals,
        "signals_per_batch": args.signals_per_batch,
        "signals_per_scene": args.signals_per_scene,
        "crops_per_signal": args.crops_per_signal,
        "max_objects_per_image": args.max_objects_per_image,
        "unique_injected_signals": unique_signal_count,
        "seed": args.seed,
        "rawdata_dir": str(args.rawdata_dir),
        "work_dir": str(args.work_dir),
        "file_first": args.file_first,
        "file_last": args.file_last,
        "meta": meta,
        "scene_config": asdict(scene_cfg),
        "parameter_ranges": {
            "dm_pc_cm3": DM_RANGE_PC_CM3,
            "snr": SNR_RANGE,
            "width_fwhm_ms": WIDTH_FWHM_MS_RANGE,
            "bandwidth_fwhm_mhz": BANDWIDTH_FWHM_MHZ_RANGE,
            "scattering_zero_fraction": SCATTERING_ZERO_FRACTION,
            "scattering_tau_1ghz_ms": SCATTERING_TAU_1GHZ_MS_RANGE,
            "dm_crop_spans": DM_CROP_SPANS,
            "time_crop_widths": TIME_CROP_WIDTHS,
        },
        "annotation_rules": {
            "box_center": "truth_center",
            "candidate_center_bounds": "[0, 512)",
            "out_of_bounds_boxes": "allowed; box centered on truth, may exceed [0,512); training dataset reconstructs center+wh from keypoints",
            "edge_box_policy": "true-size box centered on truth, no shrink/clip; center must stay in [0,512)",
            "min_split_effective_snr": args.min_split_effective_snr,
            "physical_fallback_min_snr": args.physical_fallback_min_snr,
            "image_morphology_first": ndi is not None,
            "morphology_window": "truth-centered local adaptive window",
            "morphology_seed": "brightest pixel within 15px of truth center",
            "multi_signal_ownership": "nearest truth center mask before connected components",
            "fallback": "truth-centered physical width/height if effective SNR >= physical_fallback_min_snr",
            "subband_box_scaling": "centered sqrt spectral energy fraction; center remains truth",
        },
    }
    write_json(args.output.with_suffix(".config.json"), config)
    metadata_rows = []
    image_idx = 0
    for scene in scenes:
        for signal in scene.crops:
            metadata_rows.append({
                **asdict(signal),
                "image_index": image_idx,
                "batch_index": scene.batch_index,
                "scene_index": scene.scene_index,
                "start_global_raw": scene.start_global_raw,
                "start_file_number": scene.start_file_number,
            })
            image_idx += 1
    write_jsonl(args.output.with_suffix(".metadata.jsonl"), metadata_rows)
    print(
        f"[plan] images={args.count} unique_signals={unique_signal_count} "
        f"scenes={len(scenes)} batches="
        f"{math.ceil(args.count / args.signals_per_batch)} input_down={scene_cfg.input_down} "
        f"output={args.output}",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps({"preview": metadata_rows[:5]}, indent=2, sort_keys=True), flush=True)
        return

    chan_std = estimate_channel_std(inj, files, meta, args)
    dd_cache = prepare_dedispersion(core, files[0], args, scene_cfg)
    annotation_rows: list[list[float]] = []
    image_idx = 0
    with create_h5(args.output, args.count, args.gzip_level) as h5:
        h5.attrs["generator"] = "generate_dataset.py"
        h5.attrs["run_label"] = args.run_label
        h5.attrs["seed"] = args.seed
        for scene in scenes:
            stokes = load_downsampled_stokes_window(
                files,
                meta,
                scene.start_global_raw,
                scene_cfg,
                scene.signals,
                freqs_mhz,
                chan_std,
                model,
                chunk_down=args.raw_chunk_down,
            )
            raw_data_down = stokes_to_freq_down(stokes, core, scene_cfg)
            raw_data_down = raw_data_down / (np.mean(raw_data_down, axis=0) + np.float32(1e-8))
            dm_time = core.d_dm_time_g(raw_data_down, CANVAS_DM_RANGE, scene_cfg.scene_output_down, dd_cache)
            images, boxes_by_image = crop_scene_images(
                core,
                dm_time,
                scene.crops,
                scene.signals,
                meta,
                scene_cfg,
                freq_min,
                freq_max,
                args.min_split_effective_snr,
                args.physical_fallback_min_snr,
                args.max_objects_per_image,
            )

            for local_idx, (image, boxes) in enumerate(zip(images, boxes_by_image)):
                h5["images"][image_idx] = image
                first_file_idx = scene.start_global_raw // meta["samples_per_file"]
                h5["original_filename"][image_idx] = files[first_file_idx].name
                h5["original_path"][image_idx] = str(files[first_file_idx])
                h5["original_slice"][image_idx] = int(scene.scene_index)
                for box in boxes:
                    annotation_rows.append([float(image_idx), *box])
                image_idx += 1

            h5.flush()
            print(
                f"[scene] {scene.scene_index + 1}/{len(scenes)} "
                f"batch={scene.batch_index} images={image_idx}/{args.count} "
                f"annotations={len(annotation_rows)}",
                flush=True,
            )
            if image_idx % args.signals_per_batch == 0 or image_idx == args.count:
                h5.flush()
                print(f"[progress] images={image_idx}/{args.count} annotations={len(annotation_rows)}", flush=True)

            del stokes, raw_data_down, images

        write_annotations(h5, annotation_rows, args.gzip_level)
        h5.flush()
    write_annotations_json(args.output.with_name(f"{args.output.stem}_annotations.json"), args.count, annotation_rows)
    print(f"[done] h5={args.output} annotations={len(annotation_rows)}", flush=True)


if __name__ == "__main__":
    main()
