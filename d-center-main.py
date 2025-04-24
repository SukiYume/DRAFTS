import os, re, sys, cv2
import numpy as np
from astropy.io import fits
from numba import cuda, njit, prange

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
sns.set_color_codes()

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from centernet_utils import get_res
from centernet_model import centernet


### 读取fits文件，只保留两维数据
def load_fits_file(file_name, reverse_flag=False):

    try:
        import fitsio
        data, h  = fitsio.read(file_name, header=True)
    except:
        with fits.open(file_name) as f:
            h    = f[1].header
            data = f[1].data
    data         = data['DATA'].reshape(h['NAXIS2']*h['NSBLK'], h['NPOL'], h['NCHAN'])[:, :2, :]
    if reverse_flag: data = np.array(data[:, :, ::-1])

    return data


### 读取fits头文件，获取观测参数，并指定为全局变量
def get_obparams(file_name):

    global freq, freq_reso, time_reso, file_leng, down_freq_rate, down_time_rate
    with fits.open(file_name) as f:
        time_reso  = f[1].header['TBIN']
        freq_reso  = f[1].header['NCHAN']
        file_leng  = f[1].header['NAXIS2'] * f[1].header['NSBLK']
        freq       = f[1].data['DAT_FREQ'][0, :].astype(np.float64)
    down_freq_rate = int(freq_reso / 512)
    down_time_rate = int((49.152 * 4 / 1e6) / time_reso)


### 单线ddm
def d_dm_time_s(data, height, width):
    new_data                    = np.zeros((3, height, width))
    freq_index                  = np.append(
        np.arange(int(10  / 4096 * freq_reso // down_freq_rate), int( 650 / 4096 * freq_reso // down_freq_rate), 1),
        np.arange(int(820 / 4096 * freq_reso // down_freq_rate), int(4050 / 4096 * freq_reso // down_freq_rate), 1)
    )
    for DM in range(0, height, 1):
        dds                     = (4.15 * DM * (freq**-2 - freq.max()**-2) * 1e3 / time_reso / down_time_rate).astype(np.int64)
        time_series             = np.zeros(width)
        for i in range(0, len(freq_index), 1):
            i                   = freq_index[i]
            time_series        += data[dds[i]: dds[i] + width, i]
            if i == 256:
                new_data[1, DM] = time_series
        new_data[0, DM]         = time_series
        new_data[2, DM]         = time_series - new_data[1, DM]
    return new_data


### 多线ddm
@njit(parallel=True)
def d_dm_time_m(data, height, width):
    new_data                    = np.zeros((3, height, width))
    freq_index                  = np.append(
        np.arange(int(10  / 4096 * freq_reso // down_freq_rate), int( 650 / 4096 * freq_reso // down_freq_rate), 1),
        np.arange(int(820 / 4096 * freq_reso // down_freq_rate), int(4050 / 4096 * freq_reso // down_freq_rate), 1)
    )
    for DM in prange(0, height, 1):
        dds                     = (4.15 * DM * (freq ** -2 - freq.max() ** -2) * 1e3 / time_reso / down_time_rate).astype(np.int64)
        time_series             = np.zeros(width)
        for i in prange(0, len(freq_index), 1):
            i                   = freq_index[i]
            time_series        += data[dds[i]: dds[i] + width, i]
            if i == int(freq_reso // 2):
                new_data[1, DM] = time_series
        new_data[0, DM]         = time_series
        new_data[2, DM]         = time_series - new_data[1, DM]
    return new_data


### 显卡ddm
@cuda.jit
def de_disp(dm_time, data, freq, index):
    x, y                 = cuda.grid(2)
    if x < dm_time.shape[1] and y < dm_time.shape[2]:
        td_i, DM         = 0, x
        for i in index:
            td_i        += data[int(4.15 * DM * (freq[i]**-2 - freq[-1]**-2) * 1e3 / time_reso / down_time_rate + y), i]
            if i == 256: dm_time[1, x, y] = td_i
        dm_time[2, x, y] = td_i - dm_time[1, x, y]
        dm_time[0, x, y] = td_i


def d_dm_time_g(data, height, width):

    freq_gpu      = cuda.to_device(np.mean(freq.reshape(freq_reso // down_freq_rate, down_freq_rate), axis=1))
    index_gpu     = cuda.to_device(np.append(
        np.arange(int(10  / 4096 * freq_reso // down_freq_rate), int( 650 / 4096 * freq_reso // down_freq_rate), 1),
        np.arange(int(820 / 4096 * freq_reso // down_freq_rate), int(4050 / 4096 * freq_reso // down_freq_rate), 1)
    )) # cuda.to_device(np.arange(0, int(freq_reso // down_freq_rate), 1))
    dm_time_gpu, data_gpu = cuda.to_device(np.zeros((3, height, width)).astype(np.float32)), cuda.to_device(data)

    nthreads = (8, 128)
    nblocks  = (height // nthreads[0] + 1, width // nthreads[1] + 1)
    de_disp[nblocks, nthreads](dm_time_gpu, data_gpu, freq_gpu, index_gpu)
    dm_time  = dm_time_gpu.copy_to_host()

    return dm_time


def preprocess_img(img):

    img  = (img - np.min(img)) / (np.max(img) - np.min(img))
    img  = (img - np.mean(img)) / np.std(img)
    img  = cv2.resize(img, (512, 512))

    img  = np.clip(img, *np.percentile(img, (0.1, 99.9)))
    img  = (img - np.min(img)) / (np.max(img) - np.min(img))
    img  = plt.get_cmap('mako')(img)
    img  = img[..., :3]

    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]

    return img


def postprocess_img(img):

    img  = np.array(img).transpose(1, 2, 0)
    img *= [0.229, 0.224, 0.225]
    img += [0.485, 0.456, 0.406]
    img  = (img * 255).astype(np.uint8)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


if __name__ == '__main__':

    DM_range      = 2048
    block_size    = 8192
    det_prob      = 0.3

    ## 载入模型
    base_model    = 'resnet18'
    model         = centernet(model_name=base_model).to(device)
    model.load_state_dict(torch.load('./cent_resnet18.pth', map_location=device, weights_only=True))
    model.eval()

    data_path     = './'
    save_path     = './'
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    file_list     = np.sort([i for i in os.listdir(data_path) if i.endswith('fits')])
    file_list     = np.append(file_list, file_list[-1])
    get_obparams(data_path + file_list[0])

    ### combine file number
    dds           = (4.15 * DM_range * (freq**-2 - freq.max()**-2) * 1e3 / time_reso).astype(np.int64)
    dds_file      = int(np.ceil(dds.max() / file_leng))
    block_file    = int(np.ceil(down_time_rate * block_size / file_leng))
    comb_file     = block_file + dds_file
    print(block_file, comb_file)

    ### loop
    for i in range(0, len(file_list), block_file):

        ### read data
        filename              = file_list[i].split('.fits')[0]
        print(filename)

        raw_data              = np.empty((0, 2, freq_reso))
        for j in range(comb_file):
            if i + j          < len(file_list):
                raw_data      = np.append(raw_data, load_fits_file(data_path + file_list[i + j]), axis=0)

        if raw_data.shape[0]  < comb_file * file_leng:
            raw_data          = np.append(raw_data, np.random.rand(comb_file * file_leng - raw_data.shape[0], 2, freq_reso) * np.std(raw_data) + np.mean(raw_data), axis=0)
        raw_data              = np.mean(raw_data.reshape(comb_file * file_leng // down_time_rate, down_time_rate, 2, freq_reso//down_freq_rate, down_freq_rate), axis=(1, 2, 4)).astype(np.float32)
        # raw_data              = raw_data / np.mean(raw_data, axis=0)

        print('done load file')
        ### time delay correct
        new_data              = d_dm_time_g(raw_data, height=DM_range, width=block_file*file_leng//down_time_rate)
        # del raw_data
        # cuda.current_context().memory_manager.deallocations.clear()
        print('done ddm')

        ### down_sampling and predict
        down_file_leng    = block_file * file_leng // down_time_rate
        data              = np.mean(new_data.reshape(3, DM_range // 2, 2, down_file_leng), axis=2).astype(np.float32)

        print(down_file_leng // block_size)
        for j in range(down_file_leng // block_size):
            slice         = data[:, :, j * block_size: (j + 1) * block_size]
            for k in range(3):
                img = preprocess_img(slice[k]).transpose([2, 0, 1])
                with torch.no_grad():
                    hm, wh, offset = model(torch.from_numpy(img).to(device).float().unsqueeze(0))
                top_conf, top_boxes = get_res(hm, wh, offset, confidence=det_prob)

                ## 画框并保存
                if top_boxes is not None:
                    img = postprocess_img(img) ## 转回能画图的img
                    for box in top_boxes:
                        left_x, left_y, right_x, right_y = box.astype(np.int64)
                        DM = (left_y + right_y) / 2 * (DM_range / 512)
                        dm_flag = True if DM > 20 else False
                        cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (0, 220, 0), 1)
                        print(top_conf, DM)

                    if dm_flag:
                        # data_slice = new_data[k, :, j * block_size: (j + 1) * block_size]
                        # np.save('{}{}-TS{:0>2d}-FS{}.npy'.format(save_path, filename, j, k), data_slice.astype(np.float32))

                        TOA        = ((left_x + right_x) / 2 * (block_size / 512) + j * block_size) * down_time_rate * time_reso
                        toa_samp   = np.int64(TOA / time_reso / down_time_rate)
                        start_samp = np.max([0, toa_samp - 512])
                        freq_down  = np.mean(freq.reshape(freq_reso // down_freq_rate, down_freq_rate), axis=1)

                        dds        = np.int64(4.15 * DM * (freq_down ** -2 - freq_down.max() ** -2) * 1e3 / time_reso / down_time_rate)
                        burst      = raw_data[start_samp: start_samp + dds.max() + 2048, :]
                        new_data   = np.zeros((2048, 512))
                        for q in range(512):
                            new_data[:, q] = burst[dds[q]: dds[q] + 2048, q]
                        new_data   = np.mean(new_data.reshape(512, 4, 512, 1), axis=(1, 3))
                        new_data   = new_data / np.mean(new_data, axis=0)
                        vmin, vmax = np.percentile(new_data, [5, 95])

                        plt.figure(figsize=(5, 4))
                        plt.imshow(new_data.T, aspect='auto', origin='lower', cmap='mako', vmin=vmin, vmax=vmax)
                        plt.yticks(np.linspace(0, 512, 6), np.round(np.linspace(freq.min(), freq.max(), 6)).astype(np.int64))
                        plt.xticks(np.linspace(0, 512, 5), np.round(np.linspace(0, 512, 5)*time_reso*down_time_rate*4*1e3, 2))
                        plt.xlabel('Time (ms)')
                        plt.ylabel('Frequency (MHz)')
                        plt.savefig('{}{}-TS{:0>2d}-FS{}-Burst.jpg'.format(save_path, filename, j, k), dpi=300, bbox_inches='tight')
                        plt.show()

                        plt.figure(figsize=(5, 4))
                        plt.imshow(img, origin='lower')
                        plt.savefig('{}{}-TS{:0>2d}-FS{}.jpg'.format(save_path, filename, j, k), dpi=300, bbox_inches='tight')
                        plt.close()
        del new_data
