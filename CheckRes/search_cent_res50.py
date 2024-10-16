import os, re, sys, cv2, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    down_time_rate = int((49.152 * 16 / 1e6) / time_reso)


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

    det_prob   = 0.5
    base_model = 'resnet50'
    root_path  = './RawData/Data/'
    save_path  = './RawData/Result/ObjectDetection/{}/'.format(base_model)
    if not os.path.exists(save_path): os.makedirs(save_path)

    time_list  = []
    ## 载入模型
    model = centernet(model_name=base_model).to(device)
    model.load_state_dict(torch.load('cent_{}.pth'.format(base_model), map_location=device))
    model.eval()

    for frb in ['FRB20121102', 'FRB20201124', 'FRB20180301']:
        file_list = [i for i in os.listdir(root_path) if i.endswith('fits') and frb in i]
        file_list.sort()

        get_obparams(root_path + file_list[0])

        for i in range(len(file_list)):
            print(file_list[i])
            start = time.time()
            data = load_fits_file(root_path + file_list[i])
            data = np.vstack([data, data[::-1, :]])
            data = np.mean(data.reshape(data.shape[0] // down_time_rate, down_time_rate, freq_reso //   down_freq_rate, down_freq_rate), axis=(1, 3)).astype(np.float32)

            dm_time = d_dm_time_g(data, height=1024, width=file_leng // down_time_rate)

            time_length = dm_time.shape[2]
            time_slice = 4
            time_slice_length = time_length // time_slice
            ## 文件切成4片
            for j in range(time_slice):
                slice = dm_time[:, :, time_slice_length * j: time_slice_length * (j + 1)]
                ## 循环三个频率切片
                for k in range(3):
                    a = np.mean(slice[k].reshape(512, 2, time_slice_length), axis=1)
                    img = preprocess_img(a).transpose([2, 0, 1])
                    with torch.no_grad():
                        hm, wh, offset = model(torch.from_numpy(img).to(device).float().unsqueeze(0))
                    top_conf, top_boxes = get_res(hm, wh, offset, confidence=det_prob)

                    ## 画框并保存
                    if top_boxes is not None:
                        img = postprocess_img(img)
                        for box_index in range(len(top_boxes)):
                            left_x, left_y, right_x, right_y = top_boxes[box_index].astype(np.int64)
                            with open('./RawData/Result/result_{}.txt'.format(base_model), 'a+', encoding='utf-8') as f:
                                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                                    file_list[i], j, k, left_x, left_y, right_x, right_y, top_conf[box_index]
                                ))
                            cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (0, 220, 0), 1)

                            plt.figure()
                            plt.imshow(img, origin='lower')
                            plt.savefig(save_path + '{}_{}_{}.png'.format(
                                file_list[i].replace('.fits', ''), j, k
                            ), dpi=300, bbox_inches='tight')
                            plt.close()

            time_list.append(time.time()-start)
            print(np.mean(time_list))