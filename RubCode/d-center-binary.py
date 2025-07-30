import os, sys, cv2
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns
from numba import cuda
from dataclasses import dataclass

import torch
from centernet_utils import get_res
from centernet_model import centernet
from binary_model import BinaryNet

# 设置matplotlib和seaborn样式
plt.style.use('default')
sns.set_color_codes()

# 判断是否有GPU
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### 设置全局变量
@dataclass
class ProcessConfig:
    """存储算法和运行时超参数"""
    DM_range: int       = 4096
    DM_SCALE: float     = 1    # 原为 1
    DM_OFFSET: float    = 0    # 原为 0
    DM_THRESHOLD: float = 50
    block_size: int     = 8192
    det_prob: float     = 0.3
    gpu_num: int        = 5      # 用于多进程切片


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

    return np.mean(data, axis=1)


### 读取2bit fits文件，只保留两维数据
def load_2bit_fits_file(file_name):

    with fits.open(file_name) as f:
        h        = f[1].header
        try:
            data = f[1].data['DATA']
        except:
            # 如果缺失DATA字段，生成一个随机数组
            data = np.random.randint(0, 3, (h['NAXIS2'], h['NSBLK']//4, h['NPOL'], h['NCHAN'], 1), dtype=np.uint8)
    data         = np.unpackbits(data.reshape(h['NAXIS2'], -1), axis=1).reshape(h['NAXIS2'], -1, 2)
    data         = data[..., 0] << 1 | data[..., 1]
    # 调整为二维（时间, 频率）
    data         = data.reshape(h['NAXIS2'] * h['NSBLK'], h['NCHAN'])

    return data


### 读取fits头文件，获取观测参数，并指定为全局变量
def get_obparams(file_name):

    global freq, freq_reso, time_reso, file_leng, down_freq_rate, down_time_rate, nbits
    with fits.open(file_name) as f:
        time_reso  = f[1].header['TBIN']
        freq_reso  = f[1].header['NCHAN']
        file_leng  = f[1].header['NAXIS2'] * f[1].header['NSBLK']
        obs_freq   = f[0].header['OBSFREQ']
        obs_bw     = f[0].header['OBSBW']
        nbits      = f[1].header['NBITS']
        freq       = obs_freq - obs_bw / 2 + np.arange(freq_reso) * obs_bw / freq_reso
    down_freq_rate = int(freq_reso / 512)
    down_time_rate = int((49.152 * 4 / 1e6) / time_reso)


### 显卡 dedispersion kernel，已传入必要常量，预计算频率倒数平方
@cuda.jit
def de_disp(dm_time, data, freq_inv2, freq_max_inv2, index, time_reso, down_time_rate, dm_scale, dm_offset):

    x, y                 = cuda.grid(2)
    if x < dm_time.shape[1] and y < dm_time.shape[2]:
        td_i             = 0.0
        DM               = x * dm_scale + dm_offset
        # 预先计算常量因子，减少循环中重复计算
        factor           = 4.15 * DM * 1e3 / (time_reso * down_time_rate)
        for idx in range(index.shape[0]):
            i            = index[idx]
            # 计算时延（向下取整）
            shift        = int(factor * (freq_inv2[i] - freq_max_inv2) + y)
            # 检查索引是否越界
            if shift < data.shape[0]:
                td_i    += data[shift, i]
            # 若特定索引位置（例如256）存储中间结果
            if i == 256:
                dm_time[1, x, y] = td_i
        dm_time[2, x, y] = td_i - dm_time[1, x, y]
        dm_time[0, x, y] = td_i


### GPU去散射接口函数（对传入数据预处理计算预先转换到GPU，复用预计算结果）
def d_dm_time_g(data, height, width, process_config):

    # 计算下采样后频率均值及倒数平方
    freq_down     = np.mean(freq.reshape(freq_reso // down_freq_rate, down_freq_rate), axis=1).astype(np.float32)
    # 计算频率倒数平方
    freq_inv2     = 1.0 / (freq_down * freq_down)
    freq_max_inv2 = freq_inv2[-1]
    freq_inv2_gpu = cuda.to_device(freq_inv2)

    # 对index数组进行优化：若观测参数不变，可以考虑预先构造并全局复用
    index_array   = np.concatenate([
        np.arange(int( 10 / 4096 * freq_reso // down_freq_rate), int( 650 / 4096 * freq_reso // down_freq_rate)),
        np.arange(int(820 / 4096 * freq_reso // down_freq_rate), int(4050 / 4096 * freq_reso // down_freq_rate))
    ]).astype(np.int32)
    index_gpu     = cuda.to_device(index_array)
    dm_time_gpu   = cuda.to_device(np.zeros((3, height, width), dtype=np.float32))
    data_gpu      = cuda.to_device(data)

    nthreads      = (8, 128)
    nblocks       = ((height + nthreads[0] - 1) // nthreads[0], (width + nthreads[1] - 1) // nthreads[1])
    de_disp[nblocks, nthreads](
        dm_time_gpu, data_gpu, freq_inv2_gpu, freq_max_inv2, index_gpu, time_reso, down_time_rate,
        process_config.DM_SCALE, process_config.DM_OFFSET
    )
    dm_time       = dm_time_gpu.copy_to_host()

    return dm_time


### 图像预处理和后处理
def preprocess_img(img):

    # 归一化
    img_norm     = (img - np.min(img)) / np.ptp(img)
    # 标准化
    img_norm     = (img_norm - np.mean(img_norm)) / np.std(img_norm)
    # 调整尺寸
    img_resized  = cv2.resize(img_norm, (512, 512))
    # 裁剪极值
    lower, upper = np.percentile(img_resized, (0.1, 99.9))
    img_clipped  = np.clip(img_resized, lower, upper)
    img_norm2    = (img_clipped - np.min(img_clipped)) / np.ptp(img_clipped)
    # 使用预设 colormap（返回RGBA数组，取前三个通道）
    img_colormap = plt.get_cmap('mako')(img_norm2)[..., :3]
    # 标准化至ImageNet均值与标准差（可考虑提前合并）
    img_final    = (img_colormap - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    return img_final.transpose(2, 0, 1) # 保持通道在第一维


def postprocess_img(img):

    # 假设img维度为 (C, H, W)，转换为 (H, W, C)
    img = np.array(img).transpose(1, 2, 0)
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img = np.clip(img, 0, 1)

    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def preprocess_data(data, exp_cut=5):

    data       = data.copy() + 1
    data       = data / np.mean(data, axis=0)
    vmin, vmax = np.nanpercentile(data, [exp_cut, 100-exp_cut])
    data       = np.clip(data, vmin, vmax)
    data       = (data - data.min()) / np.ptp(data)

    return data


### 对每一路径数据进行处理（在此处对文件的读写操作及数据拼接做优化）
def process_path(data_path, beam, save_path, model, class_model, process_config, section=None):

    block_size    = process_config.block_size
    # 使用更高效的方式获取文件列表
    file_list = sorted([
        i for i in os.listdir(data_path)
        if i.endswith('.fits')
            and f"M{beam:0>2d}" in i
            and '_N_' not in i and '_W_' not in i
    ])
    if len(file_list) == 0: return

    # 如果指定了section参数，则只处理该section的文件
    if section is not None and process_config.gpu_num is not None:
        chunk_size = len(file_list) // process_config.gpu_num
        file_list  = file_list[
            section * chunk_size: (section + 1) * chunk_size
            if section < process_config.gpu_num - 1 else len(file_list)
        ]
        print(f"[INFO] Section {section} 处理 {len(file_list)} 个文件")

    # 创建保存路径，补充最后一个文件
    os.makedirs(save_path, exist_ok=True)
    file_list.append(file_list[-1])
    get_obparams(os.path.join(data_path, file_list[0]))

    # 算联合文件数量
    dds                   = (4.15 * process_config.DM_range * (freq**-2 - freq.max()**-2) * 1e3 / time_reso).astype(np.int64)
    dds_file              = int(np.ceil(dds.max() / file_leng))
    block_file            = int(np.ceil(down_time_rate * block_size / file_leng))
    comb_file             = block_file + dds_file
    print(block_file, comb_file)

    # 按块循环处理文件
    for i in range(0, len(file_list), block_file):
        filename          = os.path.splitext(file_list[i])[0]
        print(filename)

        # 使用列表先收集文件数据
        raw_data_list     = []
        for j in range(comb_file):
            if i + j < len(file_list):
                file_path = os.path.join(data_path, file_list[i+j])
                # 根据nbits选择合适的加载函数
                if nbits == 2:
                    raw_data_list.append(load_2bit_fits_file(file_path))
                else:
                    raw_data_list.append(load_fits_file(file_path))
        raw_data          = np.concatenate(raw_data_list, axis=0)

        # 如果文件数据不足，则通过pad补充
        expected_samples  = comb_file * file_leng
        if raw_data.shape[0] < expected_samples:
            padding       = np.random.rand(expected_samples - raw_data.shape[0], freq_reso) * np.std(raw_data) + np.mean(raw_data)
            raw_data      = np.concatenate([raw_data, padding], axis=0)

        new_shape         = (comb_file * file_leng // down_time_rate, down_time_rate, freq_reso//down_freq_rate, down_freq_rate)
        raw_data          = np.mean(raw_data.reshape(new_shape), axis=(1, 3)).astype(np.float32)
        raw_data          = raw_data / (np.mean(raw_data, axis=0) + 1e-8)

        # 调用GPU消色散
        print('done load file')
        new_data          = d_dm_time_g(
            raw_data,
            height         = process_config.DM_range,
            width          = block_file * file_leng // down_time_rate,
            process_config = process_config
        )
        print('done ddm')

        # 将DM通道数据按行降采样
        down_file_leng    = block_file * file_leng // down_time_rate
        data_reshaped     = np.mean(new_data.reshape(3, process_config.DM_range // 2, 2, down_file_leng), axis=2).astype(np.float32)

        # 循环处理每个块
        num_slices        = down_file_leng // block_size
        print(num_slices)
        for j in range(num_slices):
            slice_data    = data_reshaped[:, :, j * block_size: (j + 1) * block_size]
            # 批量构造图片，减少循环调用模型的开销
            img_batch     = []
            batch_indices = []
            for k in range(3):
                for d in range(process_config.DM_range // 2048):
                    img   = preprocess_img(slice_data[k, d*1024:(d+1)*1024])
                    img_batch.append(img)
                    batch_indices.append((k, d))
            img_tensor    = torch.from_numpy(np.stack(img_batch, axis=0)).to(device).float()
            with torch.no_grad():
                hm, wh, offset = model(img_tensor)

            # 获取检测结果
            for idx, (k, d) in enumerate(batch_indices):
                top_conf, top_boxes         = get_res(hm[idx:idx+1], wh[idx:idx+1], offset[idx:idx+1], confidence=process_config.det_prob)
                if top_boxes is not None:
                    img_display             = postprocess_img(img_tensor[idx].cpu().numpy())
                    for box in top_boxes:
                        left_x, left_y, right_x, right_y = box.astype(np.int64)
                        DM_value            = (left_y + right_y) / 2 * (process_config.DM_range / (process_config.DM_range // 2048) / 512) + (d * 1024 * (process_config.DM_range // 2048))
                        DM_value            = DM_value * process_config.DM_SCALE + process_config.DM_OFFSET
                        dm_flag             = DM_value > process_config.DM_THRESHOLD
                        cv2.rectangle(img_display, (left_x, left_y), (right_x, right_y), (0, 220, 0), 1)
                        print(top_conf, DM_value)

                        # 如果DM值大于100，则画图检测
                        if dm_flag:
                            TOA             = ((left_x + right_x) / 2 * (block_size / 512) + j * block_size) * down_time_rate * time_reso
                            toa_samp        = int(TOA / time_reso / down_time_rate)
                            start_samp      = max(0, toa_samp - 512)
                            freq_down       = np.mean(freq.reshape(freq_reso // down_freq_rate, down_freq_rate), axis=1)
                            dds_vals        = (4.15 * DM_value * (freq_down ** -2 - freq_down.max() ** -2) * 1e3 / (time_reso * down_time_rate)).astype(np.int64)
                            burst           = raw_data[start_samp: start_samp + dds_vals.max() + 2048, :]
                            new_burst       = np.empty((2048, 512), dtype=np.float32)
                            for q in range(512):
                                new_burst[:, q] = burst[dds_vals[q]: dds_vals[q] + 2048, q]
                            new_burst       = np.mean(new_burst.reshape(512, 4, 512, 1), axis=(1, 3))
                            new_burst       = preprocess_data(new_burst)
                            with torch.no_grad():
                                inputs      = torch.from_numpy(new_burst[np.newaxis, np.newaxis, :, :]).to(device).float()
                                predict_res = class_model(inputs)
                                predict_res = torch.softmax(predict_res, dim=1)[0, 1].item()

                            # 绘图
                            if predict_res > 0.5:
                                print('burst!')
                                data_slice = new_data[k, :, j * block_size: (j + 1) * block_size]
                                np.save('{}{}-TS{:0>2d}-FS{}.npy'.format(save_path, filename, j, k), data_slice.astype(np.float32))

                                plt.figure(figsize=(7.5, 3))
                                plt.subplot(121)
                                plt.imshow(img_display, origin='lower')
                                plt.xticks(np.linspace(0, 512, 5), np.round(np.linspace(0, block_size, 5) * time_reso * down_time_rate * 1e3, 1))
                                plt.yticks(np.linspace(0, 500, 5), ((np.linspace(0, 2000, 5) + 2048 * d) * process_config.DM_SCALE + process_config.DM_OFFSET).astype(np.int64))
                                plt.xlabel('Time (ms)')
                                plt.ylabel('DM (pc cm-3)')
                                plt.subplot(122)
                                plt.imshow(new_burst.T, origin='lower', cmap='mako')
                                plt.yticks(np.linspace(0, 512, 6), np.round(np.linspace(freq.min(), freq.max(), 6)).astype(np.int64))
                                plt.xticks(np.linspace(0, 512, 5), np.round(np.linspace(0, 512, 5) * time_reso * down_time_rate * 4 * 1e3, 2))
                                plt.xlabel('Time (ms)')
                                plt.ylabel('Frequency (MHz)')
                                plt.savefig(f'{save_path}{filename}-TS{j:0>2d}-FS{k}-DM{DM_value:.1f}.jpg', dpi=300,  bbox_inches='tight')
                                plt.close()
        del new_data


if __name__ == '__main__':

    process_config = ProcessConfig(
        DM_range        = 2048,
        DM_SCALE        = 1, # 1 / 4
        DM_OFFSET       = 0, # - 128
        DM_THRESHOLD    = 50,
        block_size      = 8192,
        det_prob        = 0.3,
        gpu_num         = 5
    )

    ## 载入模型
    base_model      = 'resnet18'
    model           = centernet(model_name=base_model).to(device)
    model.load_state_dict(
        torch.load('./best_model_resnet18.pth', map_location=device, weights_only=True)
    )
    model.eval()

    class_model     = BinaryNet(base_model, num_classes=2).to(device)
    class_model.load_state_dict(
        torch.load('./best_model_resnet18_fix_n4.pth', map_location=device, weights_only=True)
    )
    class_model.eval()

    ## 开始处理
    if len(sys.argv) > 1:
        section         = int(sys.argv[1])
        print(f"[INFO] 启动 Section {section}")

        process_path(
            data_path      = '/data31/ZD2024_5/FRB20240114A/20250606/',
            beam           = 1,
            save_path      = '/home/ykzhang/low.iops.files/FRB20240114A/CentData/20250606/',
            model          = model,
            class_model    = class_model,
            process_config = process_config,
            section        = section,
        )
    else:
        process_path(
            data_path      = '/data31/ZD2024_5/FRB20240114A/20250606/',
            beam           = 1,
            save_path      = '/home/ykzhang/low.iops.files/FRB20240114A/CentData/20250606/',
            model          = model,
            class_model    = class_model,
            process_config = process_config,
        )
