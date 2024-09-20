import os, re, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
from astropy.io import fits

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('default')
sns.set_color_codes()

import torch, torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from binary_model import SPPResNet, BinaryNet


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


def preprocess_data(data, exp_cut=5):

    data         = data.copy()
    data         = data + 1
    w, h         = data.shape
    data        /= np.mean(data, axis=0)
    vmin, vmax   = np.nanpercentile(data, [exp_cut, 100-exp_cut])
    data         = np.clip(data, vmin, vmax)
    data         = (data - data.min()) / (data.max() - data.min())

    return data

def plot_burst(data, filename, block):

    fig          = plt.figure(figsize=(5, 5))
    gs           = gridspec.GridSpec(4, 1)

    w, h         = data.shape
    profile      = np.mean(data, axis=1)
    time_start   = (block * block_size) * time_reso
    peak_time    = time_start + np.argmax(profile) * time_reso

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(gs[0, 0])
    plt.plot(profile, color='royalblue', alpha=0.8, lw=1)
    plt.xlim(0, w)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(gs[1:, 0])
    plt.imshow(data.T, origin='lower', cmap='mako', aspect='auto')
    plt.yticks(np.linspace(0, h, 6), np.int64(np.linspace(freq.min(), freq.max(), 6)))
    plt.xticks(np.linspace(0, w, 6), np.round(time_start + np.arange(6) * time_reso * block_size / 5, 2))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.savefig('{}{}-{:0>4d}-{}.jpg'.format(save_path, filename, block, peak_time), format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

    return None


if __name__ == '__main__':

    ### path config
    # down_sampling_rate        = 8
    prob                      = 0.5
    block_size                = 512
    base_model                = 'resnet50'
    # section                   = int(sys.argv[1])

    time_list                 = []
    for frb in ['FRB20121102', 'FRB20201124', 'FRB20180301']:

        if frb == 'FRB20121102':
            DM                    = 565
        elif frb == 'FRB20201124':
            DM                    = 413
        elif frb == 'FRB20180301':
            DM                    = 515

        model_path                = 'class_{}.pth'.format(base_model)
        date_path                 = './RawData/Data/'
        save_path                 = './RawData/Result/BinaryClassification/{}/'.format(base_model)
        if not os.path.exists(save_path): os.makedirs(save_path)

        file_list                 = np.sort([i for i in os.listdir(date_path) if i.endswith('fits') and i.startswith(frb)])
        # section_size              = int(len(file_list) / 20) + 1
        # file_list                 = np.sort(file_list[section*section_size: (section+1)*section_size])

        ### file params read
        with fits.open(date_path + file_list[0]) as f:
            down_sampling_rate    = int(16 / (f[1].header['TBIN'] / (49.152 / 1e6)))
            time_reso             = f[1].header['TBIN'] * down_sampling_rate
            freq_reso             = f[1].header['NCHAN']
            file_leng             = f[1].header['NAXIS2'] * f[1].header['NSBLK']  // down_sampling_rate
            freq                  = f[1].data['DAT_FREQ'][0, :].astype(np.float64)

        reverse_flag              = False
        if freq[0] > freq[-1]:
            reverse_flag          = True
            freq                  = np.array(freq[::-1])

        ### time delay correct
        dds                       = (4.15 * DM * (freq**-2 - freq.max()**-2) * 1e3 / time_reso).astype(np.int64)
        if file_leng % 512:
            redundancy            = ((file_leng // 512) + 1) * 512 - file_leng
        else:
            redundancy            = 0

        ### model config
        model                     = BinaryNet(base_model, num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        ### read data
        for i in range(len(file_list)):
            start = time.time()
            raw_data              = load_fits_file(date_path + file_list[i], reverse_flag)
            fits_number           = i + 1 # section_size * section + i + 1
            filename              = file_list[i].split('.fits')[0]
            print(filename)

            data                  = np.mean(raw_data.reshape(raw_data.shape[0] // down_sampling_rate, down_sampling_rate, freq_reso), axis=1)

            new_data              = np.zeros(data.shape)
            for j in range(freq_reso):
                new_data[:, j]    = np.append(data[dds[j]: , j], data[:dds[j], j])
            data                  = np.mean(new_data.reshape(data.shape[0]//512, 512, 512, freq_reso//512), axis=3)

            ### predict
            for j in range(data.shape[0]):
                data[j, :, :]     = preprocess_data(data[j, :, :])
            inputs                = torch.from_numpy(data[:, np.newaxis, :, :]).float().to(device)
            predict_res           = model(inputs)
            # print(time.time()-start)
            time_list.append(time.time()-start)
            print(np.mean(time_list))

            ### plot
            with torch.no_grad():
                predict_res       = predict_res.softmax(dim=1)[:, 1].cpu().numpy()
            blocks                = np.where(predict_res >= prob)[0]
            for block in blocks:
                plotres           = plot_burst(data[block], filename, block)
                # np.save('{}{}-{:0>4d}.npy'.format(save_path, filename, block), data[block])

