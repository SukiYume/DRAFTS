import os, sys
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import seaborn as sns


def process_group(file_paths, start_mjd, down_time_rate=64):
    """读取一组 FITS，downsample 并横向拼接，再画图保存。"""
    downs = []
    for fp in file_paths:
        with fits.open(fp, memmap=True) as f:
            h    = f[1].header
            data = f[1].data['DATA'][:, :, :2]
            down = data.reshape(
                h['NAXIS2']*h['NSBLK']//down_time_rate, down_time_rate, 2, 512, h['NCHAN']//512
            ).mean(axis=(1, 2, 4))
            downs.append(down)

    concat      = np.vstack(downs)
    first, last = os.path.basename(file_paths[0]), os.path.basename(file_paths[-1])
    out_name    = f"{first[:-5]}_{last[:-5]}.jpg"
    out_file    = os.path.join(save_path, out_name)
    time_start  = get_fits_mjd(file_paths[0])
    time_start  = (time_start - start_mjd) * 24 * 3600

    plt.figure(figsize=(10, 2.5))
    plt.imshow(concat.T, cmap='mako', vmin=1, vmax=40, aspect='auto', origin='lower')
    plt.yticks(np.linspace(0, 512, 6), np.linspace(1000, 1500, 6).astype(int))
    plt.xticks(np.linspace(0, concat.shape[0], 6), np.round(np.linspace(0, concat.shape[0], 6) * down_time_rate * h['TBIN'] + time_start, 1))
    plt.ylabel('Frequency (MHz)')
    plt.xlabel('Time (s)')
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()


def get_fits_mjd(file_path):
    with fits.open(file_path) as f:
        header = f[0].header
    a, b, c = header['STT_IMJD'], header['STT_SMJD'], header['STT_OFFS']
    return a + ( b + c ) / 24 / 60 / 60


def get_minmax_scale(file_path):
    with fits.open(file_path) as f:
        h    = f[1].header
        data = f[1].data['DATA'][:, :, :2].reshape(h['NAXIS2']*h['NSBLK'], 2, h['NCHAN'])
    vmin, vmax = 2, np.mean(data)*1.2
    return vmin, vmax


if __name__ == "__main__":

    # 全局常量（保持原状或仅微调）
    beam       = int(sys.argv[1]) + 1 if len(sys.argv) > 1 else 0
    group_size = 5
    data_path  = '/data31/ZD2024_5/M60/20250707/'
    save_path  = './DataCheck_Plots/'
    os.makedirs(save_path, exist_ok=True)

    fits_files = sorted(
        os.path.join(data_path, fn)
        for fn in os.listdir(data_path)
        if fn.endswith('.fits') and f'M{beam:02d}' in fn
    )
    start_mjd  = get_fits_mjd(fits_files[0])
    vmin, vmax = get_minmax_scale(fits_files[0])

    # 按每 group_size 个文件分组
    groups = [
        fits_files[i:i+group_size]
        for i in range(0, len(fits_files), group_size)
        if len(fits_files[i:i+group_size]) == group_size
    ]

    # 并行处理每个组
    with Pool(8) as pool:
        # 使用 partial 来固定 start_mjd 参数
        process_func = partial(process_group, start_mjd=start_mjd)
        pool.map(process_func, groups)

