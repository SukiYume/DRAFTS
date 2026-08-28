"""
FITS 数据预处理脚本
将多个 FITS 文件处理为 H5 格式的训练数据
"""
import os
import glob
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import h5py
import multiprocessing as mp
from datetime import datetime


def load_fits_file(file_name):
    """读取 FITS 文件，返回二维数据 (time, freq)"""
    try:
        import fitsio
        data, h = fitsio.read(file_name, header=True)
    except Exception:
        with fits.open(file_name, memmap=False) as f:
            h = f[1].header
            data = f[1].data

    data = data["DATA"].reshape(h["NAXIS2"] * h["NSBLK"], h["NPOL"], h["NCHAN"])[:, :2, :]
    data = np.mean(data, axis=1).astype(np.float32)
    return data


def preprocess_slice(data2d):
    """将 (512, 512) 归一化到 0~1，并做 robust clip"""
    data = data2d.astype(np.float32, copy=True)
    # data += 1.0
    # data /= np.mean(data, axis=0)

    # vmin, vmax = np.percentile(data, [5, 95])
    vmin, vmax = np.percentile(data, [1, 99])
    data = np.clip(data, vmin, vmax)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data.astype(np.float32)


def processing_fits_to_slices(data, down_time_list):
    """
    输入 data: (T, F)
    输出：对每个 down_time 生成若干 (512, 512) slice
    """
    down_time_list = np.asarray(down_time_list, dtype=np.int32)
    T, F = data.shape
    if T < down_time_list.max() * 512:
        raise ValueError(f"时间维 T={T} 小于 down_time_list.max() * 512，无法下采样。")
    if F < 512 or (F % 512) != 0:
        raise ValueError(f"频率维 F={F} 不能整除 512（或小于512），无法 reshape 到 512。")

    # 频率维压到 512
    data_512 = np.mean(data.reshape(T, 512, F // 512), axis=2)  # (T, 512)

    out = []
    for down_time in down_time_list:
        down_time = int(down_time)
        if down_time <= 0:
            continue

        # 时间下采样：裁掉不能整除 down_time 的尾巴
        T2 = (T // down_time) * down_time
        if T2 < down_time:
            continue
        tmp = data_512[:T2].reshape(T2 // down_time, down_time, 512).mean(axis=1)

        # 切 512 行为一个 slice
        nrow = tmp.shape[0]
        nslice = nrow // 512
        if nslice <= 0:
            continue
        tmp = tmp[:nslice * 512].reshape(nslice, 512, 512)

        # 逐 slice 预处理
        for i in range(tmp.shape[0]):
            tmp[i] = preprocess_slice(tmp[i])

        out.append((down_time, tmp.astype(np.float32, copy=False)))

    return out


def _collect_fits_files(paths):
    """收集 FITS 文件路径，支持文件/目录/通配符，只选择文件名包含 M01 的文件"""
    if isinstance(paths, (str, os.PathLike)):
        paths = [str(paths)]
    paths = [str(p) for p in paths]

    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.fits", "*.fit", "*.FITS", "*.FIT"):
                files.extend(glob.glob(os.path.join(p, "**", ext), recursive=True))
        elif any(ch in p for ch in ["*", "?", "["]):
            files.extend(glob.glob(p, recursive=True))
        else:
            files.append(p)

    # 去重 + 排序 + 只保留文件名包含 M01 的文件
    return sorted({
        os.path.abspath(f) for f in files
        if os.path.isfile(f) and "M01" in os.path.basename(f)
    })


def _worker_process_one_file(args):
    """Worker: 处理单个 FITS 文件"""
    file_path, down_time_list = args
    try:
        data = load_fits_file(file_path)
        slices_list = processing_fits_to_slices(data, down_time_list)

        images_blocks, names_blocks, down_blocks, idx_blocks = [], [], [], []

        for down_time, imgs in slices_list:
            k = imgs.shape[0]
            if k == 0:
                continue
            images_blocks.append(imgs)
            names_blocks.append(np.array([file_path] * k, dtype=object).reshape(k, 1))
            down_blocks.append(np.full((k, 1), int(down_time), dtype=np.int32))
            idx_blocks.append(np.arange(k, dtype=np.int32).reshape(k, 1))

        if not images_blocks:
            return (file_path, None, None, None, None, None)

        return (
            file_path,
            np.concatenate(images_blocks, axis=0).astype(np.float32, copy=False),
            np.vstack(names_blocks),
            np.vstack(down_blocks),
            np.vstack(idx_blocks),
            None,
        )

    except Exception as e:
        return (file_path, None, None, None, None, repr(e))


def _create_h5_datasets(h5f):
    """创建可增长的 HDF5 数据集"""
    h5f.create_dataset(
        "images",
        shape=(0, 512, 512),
        maxshape=(None, 512, 512),
        dtype=np.float32,
        chunks=(1, 512, 512),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    h5f.create_dataset(
        "names",
        shape=(0, 1),
        maxshape=(None, 1),
        dtype=h5py.string_dtype(encoding="utf-8"),
        chunks=(1024, 1),
    )
    for k in ["down_time", "slice_idx"]:
        h5f.create_dataset(
            k,
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.int32,
            chunks=(1024, 1),
        )


def _append_to_h5(h5f, images, names, down_time, slice_idx):
    """向 H5 追加一批数据"""
    n_add = images.shape[0]
    if n_add == 0:
        return

    cur = h5f["images"].shape[0]
    new = cur + n_add

    for key in ["images", "names", "down_time", "slice_idx"]:
        h5f[key].resize((new,) + h5f[key].shape[1:])

    h5f["images"][cur:new] = images
    h5f["names"][cur:new] = names
    h5f["down_time"][cur:new] = down_time
    h5f["slice_idx"][cur:new] = slice_idx


def build_h5_from_fits_paths(
    fits_paths,
    out_h5_path=None,
    down_time_list=(2, 4),
    num_workers=None,
):
    """
    将 FITS 文件处理为 H5 训练数据

    Args:
        fits_paths: FITS 文件路径/目录路径/通配符列表
        out_h5_path: 输出 H5 文件路径（默认自动生成）
        down_time_list: 时间下采样率列表
        num_workers: 并行进程数

    Returns:
        (h5文件路径, 下采样率列表, 进程数)
    """
    files = _collect_fits_files(fits_paths)
    if not files:
        raise FileNotFoundError("没有找到任何 FITS 文件。")

    down_time_list = np.array(list(down_time_list), dtype=np.int32)

    if num_workers is None:
        num_workers = max(1, min(mp.cpu_count(), 8))
    num_workers = int(num_workers)

    if out_h5_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_h5_path = os.path.abspath(f"down_data_{ts}.h5")
    else:
        out_h5_path = os.path.abspath(out_h5_path)

    # 多进程处理
    ctx = mp.get_context("spawn")
    tasks = [(fp, down_time_list) for fp in files]
    error_list = []

    with h5py.File(out_h5_path, "w") as h5f:
        _create_h5_datasets(h5f)

        with ctx.Pool(processes=num_workers) as pool:
            for fp, images, names, down_time, slice_idx, err in tqdm(
                pool.imap_unordered(_worker_process_one_file, tasks),
                total=len(tasks),
                desc="Processing FITS",
                ncols=100,
            ):
                if err is not None:
                    error_list.append((fp, err))
                    continue
                if images is None:
                    continue
                _append_to_h5(h5f, images, names, down_time, slice_idx)

        # 保存元数据
        h5f.attrs["down_time_list"] = down_time_list
        h5f.attrs["num_workers"] = num_workers
        h5f.attrs["num_input_files"] = len(files)
        h5f.attrs["num_errors"] = len(error_list)
        if error_list:
            h5f.attrs["error_examples"] = np.array(
                [f"{p} :: {e}" for p, e in error_list[:50]],
                dtype=h5py.string_dtype("utf-8"),
            )

    return out_h5_path, down_time_list, num_workers


if __name__ == "__main__":
    fits_paths = [
        "/data31/PT2022_0067/",
        # "/data31/PT2023_0100/FRB20220912A",
        # "/data31/PT2024_0148/FRB20220912A/20240920",
    ]
    h5_path, down_list, nproc = build_h5_from_fits_paths(
        fits_paths,
        out_h5_path="FRB20220912A_PrepData.h5",
        # down_time_list=(8, 16, 32, 64, 128, 256),
        down_time_list=(32,),
        num_workers=32,
    )
    print(f"Saved: {h5_path}")
    print(f"down_time_list: {down_list}")
    print(f"num_workers: {nproc}")
