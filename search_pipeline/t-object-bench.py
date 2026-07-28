"""Object-detection-only backend benchmark.

This test entry runs the center detector without the binary-classification
stage. It is meant for comparing CuPy and Numba dedispersion backends across
detector models, data paths, and GPU assignments.

Example, production-like CRAFTS detector settings:
    CUDA_VISIBLE_DEVICES=0 python t-object-bench.py \
        --backend cupy \
        --detector-type centernet_conv_tiny \
        --detector-ckpt models/object_best_model_centernet_conv_tiny_ema_v10.pth \
        --data-path /path/to/observations/source/date \
        --output-root /path/to/drafts_runs/object_backend \
        --dm-range 4096 --dm-scale 1 --dm-offset 0 \
        --dm-threshold 10 --block-size 4096 --detect-dm-span 1024 --det-prob 0.45
"""

import argparse
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
GATE_PATH = BASE_DIR / "d-center-binary-gate.py"

cv2 = None
plt = None
np = None
torch = None
cp = None
cuda = None
gate = None
core = None
device = None
_numba_de_disp_kernel = None


def load_gate():
    spec = importlib.util.spec_from_file_location("d_center_binary_gate", GATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load gate module from {GATE_PATH}")
    gate = importlib.util.module_from_spec(spec)
    sys.modules["d_center_binary_gate"] = gate
    spec.loader.exec_module(gate)
    return gate


def load_runtime_dependencies():
    global cv2, plt, np, torch, cp, gate, core, device
    if gate is not None:
        return

    import cv2 as cv2_mod
    import matplotlib.pyplot as plt_mod
    import numpy as np_mod
    import torch as torch_mod

    try:
        import cupy as cp_mod
    except ImportError:
        cp_mod = None

    cv2 = cv2_mod
    plt = plt_mod
    np = np_mod
    torch = torch_mod
    cp = cp_mod
    gate = load_gate()
    core = gate.core
    device = core.device


@dataclass
class NumbaCache:
    freq_inv2_gpu: object
    index_gpu: object
    freq_max_inv2: float
    split_idx: int


def ensure_numba_kernel():
    global cuda, _numba_de_disp_kernel
    if cuda is None:
        try:
            from numba import cuda as cuda_mod
        except ImportError as exc:
            raise RuntimeError("Numba backend requested, but numba.cuda is not available") from exc
        cuda = cuda_mod
    if _numba_de_disp_kernel is not None:
        return _numba_de_disp_kernel

    @cuda.jit
    def kernel(dm_time, data, freq_inv2, freq_max_inv2, index,
               time_reso_v, down_time_rate_v, dm_scale, dm_offset,
               split_idx):
        x, y = cuda.grid(2)
        if x < dm_time.shape[1] and y < dm_time.shape[2]:
            td_i = 0.0
            td_mid = 0.0
            dm_value = x * dm_scale + dm_offset
            factor = 4.15 * dm_value * 1e3 / (time_reso_v * down_time_rate_v)
            for idx in range(index.shape[0]):
                i = index[idx]
                shift = int(factor * (freq_inv2[i] - freq_max_inv2) + y)
                if shift < data.shape[0]:
                    td_i += data[shift, i]
                if idx == split_idx:
                    td_mid = td_i
            dm_time[1, x, y] = td_mid
            dm_time[2, x, y] = td_i - td_mid
            dm_time[0, x, y] = td_i

    _numba_de_disp_kernel = kernel
    return _numba_de_disp_kernel


def default_detector_ckpt(detector_type):
    if detector_type.lower() == "centernet_conv_tiny":
        return BASE_DIR / "models" / "object_best_model_centernet_conv_tiny_ema_v10.pth"
    return BASE_DIR / "models" / f"object_best_model_{detector_type.lower()}_ema.pth"


def resolve_runtime_path(path):
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


def build_process_config(args):
    return gate.ProcessConfig(
        dm_range=args.dm_range,
        dm_scale=args.dm_scale,
        dm_offset=args.dm_offset,
        dm_threshold=args.dm_threshold,
        block_size=args.block_size,
        dm_span=args.detect_dm_span,
        det_prob=args.det_prob,
        det_iou=args.det_iou,
        section_num=args.gpu_num,
        time_factor=args.time_factor,
        save_plot=args.save_plot,
        save_npy=False,
        verbose=args.verbose,
    )


def build_numba_cache(height):
    ensure_numba_kernel()

    freq_down = np.mean(
        core.freq.reshape(core.freq_reso // core.down_freq_rate, core.down_freq_rate),
        axis=1,
    ).astype(np.float32)
    freq_inv2 = 1.0 / (freq_down * freq_down)
    index_array = np.concatenate([
        np.arange(int(10 / 4096 * core.freq_reso // core.down_freq_rate),
                  int(650 / 4096 * core.freq_reso // core.down_freq_rate)),
        np.arange(int(820 / 4096 * core.freq_reso // core.down_freq_rate),
                  int(4050 / 4096 * core.freq_reso // core.down_freq_rate)),
    ]).astype(np.int32)
    return NumbaCache(
        freq_inv2_gpu=cuda.to_device(freq_inv2),
        index_gpu=cuda.to_device(index_array),
        freq_max_inv2=float(freq_inv2[-1]),
        split_idx=len(index_array) // 2,
    )


def dedisperse_numba(data, height, width, config, cache):
    kernel = ensure_numba_kernel()
    dm_time_gpu = cuda.device_array((3, height, width), dtype=np.float32)
    data_gpu = cuda.to_device(data)
    nthreads = (8, 128)
    nblocks = (
        (height + nthreads[0] - 1) // nthreads[0],
        (width + nthreads[1] - 1) // nthreads[1],
    )
    kernel[nblocks, nthreads](
        dm_time_gpu, data_gpu, cache.freq_inv2_gpu, cache.freq_max_inv2,
        cache.index_gpu, core.time_reso, core.down_time_rate, config.dm_scale,
        config.dm_offset, cache.split_idx,
    )
    cuda.synchronize()
    return dm_time_gpu.copy_to_host()


def normalize_images_cpu(img_flat):
    vmin = np.percentile(img_flat, 1.0, axis=(1, 2), keepdims=True)
    vmax = np.percentile(img_flat, 99.5, axis=(1, 2), keepdims=True)
    img_flat = np.clip(img_flat, vmin, vmax)
    img_min = img_flat.min(axis=(1, 2), keepdims=True)
    img_max = img_flat.max(axis=(1, 2), keepdims=True)
    return (img_flat - img_min) / (img_max - img_min + 1e-8)


def draw_hollow_cross(vis, cx, cy, color=(0, 220, 0), arm=9, gap=3, thickness=1):
    x, y = int(round(cx)), int(round(cy))
    cv2.line(vis, (x - arm, y), (x - gap, y), color, thickness)
    cv2.line(vis, (x + gap, y), (x + arm, y), color, thickness)
    cv2.line(vis, (x, y - arm), (x, y - gap), color, thickness)
    cv2.line(vis, (x, y + gap), (x, y + arm), color, thickness)
    cv2.circle(vis, (x, y), gap, color, thickness)


def save_detection_plot(img, cx, cy, w_box, h_box, score, dm_value,
                        d, j, k, box_idx, filename, save_path, block_size,
                        config):
    img_disp = (img * 255).astype(np.uint8)
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
    color = (0, 220, 0)

    if w_box > 0 and h_box > 0:
        cxi, cyi = int(round(cx)), int(round(cy))
        wi, hi = max(2, int(round(w_box))), max(2, int(round(h_box)))
        x1, y1 = cxi - wi // 2, cyi - hi // 2
        cv2.rectangle(img_disp, (x1, y1), (x1 + wi, y1 + hi), color, 1)
        cv2.drawMarker(img_disp, (cxi, cyi), color, cv2.MARKER_CROSS, 8, 1)
    else:
        draw_hollow_cross(img_disp, cx, cy, color)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_disp, origin="lower")
    plt.xticks(
        np.linspace(0, 512, 5),
        np.round(np.linspace(0, block_size, 5) * core.time_reso * core.down_time_rate * 1e3, 1),
    )
    plt.yticks(
        np.linspace(0, 500, 5),
        ((np.linspace(0, config.dm_span, 5) + config.dm_span * d) *
         config.dm_scale + config.dm_offset).astype(np.int64),
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("DM (pc cm-3)")
    plt.title(f"{filename}  band={k}  score={score:.2f}  DM={dm_value:.0f}")
    out_path = Path(save_path) / f"{filename}-TS{j:0>2d}-FS{k}-BX{box_idx}-DM{dm_value:.0f}.jpg"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def image_to_numpy(img_flat, idx):
    if isinstance(img_flat, torch.Tensor):
        return img_flat[idx].detach().cpu().numpy()
    return img_flat[idx]


def handle_detection_results(img_flat, det_results, num_dm_chunks, dm_per_pixel,
                             j, filename, save_path, block_size, config):
    n_burst = 0
    for idx in range(len(img_flat)):
        k = idx // num_dm_chunks
        d = idx % num_dm_chunks
        pred_scores, pred_boxes = det_results[idx] if idx < len(det_results) else (None, None)
        if pred_boxes is None:
            continue

        for box_idx, (cx, cy, w_box, h_box) in enumerate(pred_boxes.astype(np.float32)):
            dm_value = (cy * dm_per_pixel + d * config.dm_span) * config.dm_scale + config.dm_offset
            if dm_value <= config.dm_threshold:
                continue

            n_burst += 1
            score = float(pred_scores[box_idx]) if pred_scores is not None else 0.0
            if config.verbose:
                print(
                    f"  [Det] band={k} dm_chunk={d} score={score:.3f} "
                    f"DM={dm_value:.1f} cx={cx:.1f} cy={cy:.1f}",
                    flush=True,
                )
            if config.save_plot:
                save_detection_plot(
                    image_to_numpy(img_flat, idx), cx, cy, w_box, h_box, score,
                    dm_value, d, j, k, box_idx, filename, save_path, block_size,
                    config,
                )
    return n_burst


def detect_cupy(new_data, model, decode_fn, config, filename, save_path, block_size,
                down_file_leng):
    new_data_t = core._cp_to_torch(new_data)
    num_slices = down_file_leng // block_size
    num_dm_chunks = config.dm_range // config.dm_span
    dm_per_pixel = config.dm_span // 512
    time_pool = block_size // 512
    n_burst = 0

    for j in range(num_slices):
        slice_data = new_data_t[:, :, j * block_size: (j + 1) * block_size]
        img_batch = (
            slice_data
            .reshape(3, num_dm_chunks, 512, dm_per_pixel, 512, time_pool)
            .mean(dim=(3, 5))
        )
        img_flat = img_batch.reshape(-1, 512, 512)
        img_flat = core._normalize_batch_gpu(img_flat)
        det_input = img_flat.unsqueeze(1).contiguous()
        with torch.no_grad():
            raw_pred = model(det_input)
        det_results = decode_fn(raw_pred, conf_thr=config.det_prob, iou_thr=config.det_iou)
        n_burst += handle_detection_results(
            img_flat, det_results, num_dm_chunks, dm_per_pixel, j, filename,
            save_path, block_size, config,
        )
    torch.cuda.synchronize()
    return n_burst


def detect_numba(new_data, model, decode_fn, config, filename, save_path, block_size,
                 down_file_leng):
    num_slices = down_file_leng // block_size
    num_dm_chunks = config.dm_range // config.dm_span
    dm_per_pixel = config.dm_span // 512
    time_pool = block_size // 512
    n_burst = 0

    for j in range(num_slices):
        slice_data = new_data[:, :, j * block_size: (j + 1) * block_size]
        img_batch = np.mean(
            slice_data
            .reshape(3, num_dm_chunks, 512, dm_per_pixel, 512, time_pool),
            axis=(3, 5),
        )
        img_flat = normalize_images_cpu(img_batch.reshape(-1, 512, 512))
        det_input = torch.from_numpy(img_flat).unsqueeze(1).to(device).float()
        with torch.no_grad():
            raw_pred = model(det_input)
        det_results = decode_fn(raw_pred, conf_thr=config.det_prob, iou_thr=config.det_iou)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        n_burst += handle_detection_results(
            img_flat, det_results, num_dm_chunks, dm_per_pixel, j, filename,
            save_path, block_size, config,
        )
    return n_burst


def build_backend_cache(backend, config, down_file_leng, comb_file_leng):
    if backend == "cupy":
        if cp is None:
            raise RuntimeError("CuPy backend requested, but cupy is not available")
        return core._build_dedispersion_cache(
            config.dm_range, down_file_leng, comb_file_leng, config,
        )
    return build_numba_cache(config.dm_range)


def dedisperse(backend, raw_data_down, config, down_file_leng, cache):
    if backend == "cupy":
        new_data = core.d_dm_time_g(raw_data_down, config.dm_range, down_file_leng, cache)
        cp.cuda.runtime.deviceSynchronize()
        return new_data
    return dedisperse_numba(raw_data_down, config.dm_range, down_file_leng, config, cache)


def detect(backend, new_data, model, decode_fn, config, filename, save_path, block_size,
           down_file_leng):
    if backend == "cupy":
        return detect_cupy(
            new_data, model, decode_fn, config, filename, save_path, block_size,
            down_file_leng,
        )
    return detect_numba(
        new_data, model, decode_fn, config, filename, save_path, block_size,
        down_file_leng,
    )


def process_fits_list(fits_list, save_path, model, decode_fn, config, backend,
                      n_search_files=None):
    if not fits_list:
        print("[Warn] no FITS files to process", flush=True)
        return

    fits_list = list(fits_list)
    block_size = config.block_size
    save_path = str(save_path)
    os.makedirs(save_path, exist_ok=True)

    search_limit = len(fits_list) if n_search_files is None else min(n_search_files, len(fits_list))
    if search_limit <= 0:
        print("[Warn] selected section has no anchor files", flush=True)
        return
    fits_list.append(fits_list[-1])

    core.get_obparams(fits_list[0], config.time_factor)
    dds_file = core._dispersion_file_span(config)
    block_file = int(np.ceil(core.down_time_rate * block_size / core.file_leng))
    comb_file = block_file + dds_file
    down_file_leng = block_file * core.file_leng // core.down_time_rate
    comb_file_leng = comb_file * core.file_leng // core.down_time_rate

    core._validate_config(config, block_size, down_file_leng)
    cache = build_backend_cache(backend, config, down_file_leng, comb_file_leng)

    print(
        f"[Setup] backend={backend} freq_reso={core.freq_reso} time_reso={core.time_reso:.3e}s "
        f"file_leng={core.file_leng} down_freq_rate={core.down_freq_rate} "
        f"down_time_rate={core.down_time_rate} nbits={core.nbits}",
        flush=True,
    )
    print(
        f"[Setup] dm_range={config.dm_range} dm_scale={config.dm_scale} "
        f"dm_offset={config.dm_offset} block_file={block_file} "
        f"dds_file={dds_file} comb_file={comb_file}",
        flush=True,
    )

    load_func = core.load_2bit_fits_file if core.nbits == 2 else core.load_fits_file
    file_cache = {}
    total_burst = 0
    n_blocks = (search_limit + block_file - 1) // block_file
    t_total_start = time.time()

    for block_idx, i in enumerate(range(0, search_limit, block_file), start=1):
        filename = os.path.splitext(os.path.basename(fits_list[i]))[0]
        t_block = time.time()
        print(f"\n[Block {block_idx}/{n_blocks}] anchor={filename}", flush=True)

        t0 = time.time()
        needed = [fits_list[i + j] for j in range(comb_file) if i + j < len(fits_list)]
        for fp in list(file_cache):
            if fp not in needed:
                del file_cache[fp]
        n_hit = 0
        for fp in needed:
            if fp in file_cache:
                n_hit += 1
            else:
                file_cache[fp] = load_func(fp)
        raw_data = np.concatenate([file_cache[fp] for fp in needed], axis=0)
        if raw_data.shape[0] < comb_file_leng:
            mean = float(np.mean(raw_data))
            std = float(np.std(raw_data))
            padding = np.random.rand(comb_file_leng - raw_data.shape[0], core.freq_reso).astype(np.float32)
            raw_data = np.concatenate([raw_data, padding * std + mean], axis=0)
        t_load = time.time() - t0

        t0 = time.time()
        raw_data_down = np.mean(
            raw_data.reshape(comb_file_leng, core.freq_reso // core.down_freq_rate, core.down_freq_rate),
            axis=2,
        ).astype(np.float32)
        raw_data_down = raw_data_down / (np.mean(raw_data_down, axis=0) + 1e-8)
        del raw_data
        t_prepare = time.time() - t0

        t0 = time.time()
        new_data = dedisperse(backend, raw_data_down, config, down_file_leng, cache)
        del raw_data_down
        t_dedisp = time.time() - t0

        t0 = time.time()
        n_burst_block = detect(
            backend, new_data, model, decode_fn, config, filename, save_path,
            block_size, down_file_leng,
        )
        t_detect = time.time() - t0

        total_burst += n_burst_block
        print(
            f"[Time] load={t_load:.2f}s (hit {n_hit}/{len(needed)}) "
            f"freq_ds+norm={t_prepare:.2f}s dedisp={t_dedisp:.2f}s "
            f"detect+save={t_detect:.2f}s total={time.time() - t_block:.2f}s "
            f"bursts={n_burst_block}",
            flush=True,
        )
        del new_data

    elapsed = time.time() - t_total_start
    print(
        f"\n[Done] total_bursts={total_burst} total_time={elapsed:.1f}s "
        f"avg_per_block={elapsed / max(n_blocks, 1):.2f}s",
        flush=True,
    )


def build_output_path(output_root, run_label, detector_type, backend, info):
    parts = [Path(output_root)]
    if run_label:
        parts.append(run_label)
    parts.extend([detector_type, backend, info["source"], info["date"], info["beam"]])
    return Path(*parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run object detection only, switching CuPy or Numba dedispersion backend.",
    )
    parser.add_argument("--backend", choices=("cupy", "numba"), required=True)
    parser.add_argument("--detector-type", default="centernet_conv_tiny")
    parser.add_argument("--detector-ckpt", default=None)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--beam", default="M01", help="M01, M02, all, or empty for all beams")
    parser.add_argument("--section", type=int, default=0)
    parser.add_argument("--gpu-num", type=int, default=1)
    parser.add_argument("--dm-range", type=int, default=4096)
    parser.add_argument("--dm-scale", type=float, default=1.0)
    parser.add_argument("--dm-offset", type=float, default=0.0)
    parser.add_argument("--dm-threshold", type=float, default=50.0)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--detect-dm-span", type=int, default=1024)
    parser.add_argument("--det-prob", type=float, default=0.45)
    parser.add_argument("--det-iou", type=float, default=0.5)
    parser.add_argument("--time-factor", type=float, default=8.0)
    parser.add_argument("--no-save-plot", dest="save_plot", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(save_plot=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(BASE_DIR)
    load_runtime_dependencies()

    if args.section < 0 or args.section >= args.gpu_num:
        raise ValueError(f"--section must be in [0, gpu_num), got {args.section}/{args.gpu_num}")

    detector_type = args.detector_type.lower()
    detector_ckpt = resolve_runtime_path(args.detector_ckpt) or default_detector_ckpt(detector_type)
    if not detector_ckpt.exists():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {detector_ckpt}. "
            "Pass --detector-ckpt explicitly for models not copied to search_pipeline/models."
        )

    output_root = args.output_root
    if output_root is None:
        output_root = BASE_DIR.parent / "output" / "search" / "object_detection_backend"
    output_root = Path(output_root).resolve()

    config = build_process_config(args)
    beam_filter = args.beam if args.beam else None
    all_file_lists = gate.organize_file_lists(args.data_path, beam_filter)
    section_file_lists = gate.distribute_file_lists(
        all_file_lists, config.section_num, args.section, config,
    )

    print("=" * 80, flush=True)
    print(f"  Backend       : {args.backend}", flush=True)
    print(f"  Detector      : {detector_type}", flush=True)
    print(f"  Checkpoint    : {detector_ckpt}", flush=True)
    print(f"  Data path     : {args.data_path}", flush=True)
    print(f"  Beam          : {beam_filter}", flush=True)
    print(f"  Section       : {args.section}/{args.gpu_num}", flush=True)
    print(f"  Device        : {device}", flush=True)
    print(f"  Output root   : {output_root}", flush=True)
    print("=" * 80, flush=True)
    print(
        f"[Data] total_file_lists={len(all_file_lists)} "
        f"section_file_lists={len(section_file_lists)}",
        flush=True,
    )

    if args.dry_run:
        for identifier, fits_list, info in section_file_lists:
            save_path = build_output_path(output_root, args.run_label, detector_type, args.backend, info)
            print(
                f"[DryRun] {identifier}: files={len(fits_list)} "
                f"n_search_files={info.get('n_search_files', len(fits_list))} "
                f"save_path={save_path}",
                flush=True,
            )
        return

    t0 = time.time()
    model, decode_fn = gate.load_detector(detector_type, str(detector_ckpt))
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] loaded in {time.time() - t0:.2f}s "
        f"params={n_params / 1e6:.1f}M device={next(model.parameters()).device}",
        flush=True,
    )

    for identifier, fits_list, info in section_file_lists:
        save_path = build_output_path(output_root, args.run_label, detector_type, args.backend, info)
        print(
            f"\n[Run] {identifier}: files={len(fits_list)} "
            f"n_search_files={info.get('n_search_files', len(fits_list))} "
            f"save_path={save_path}",
            flush=True,
        )
        process_fits_list(
            fits_list, save_path, model, decode_fn, config, args.backend,
            n_search_files=info.get("n_search_files"),
        )
        print(f"[Run] done {identifier}", flush=True)


if __name__ == "__main__":
    main()
