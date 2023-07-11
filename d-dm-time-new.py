import os, re, sys
import numpy as np
import pandas as pd
from astropy.io import fits
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('default')
sns.set_color_codes()

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

def preprocess_data(data, exposure_cut=20):
    data         = data.copy()
    data         = data + 1
    w, h         = data.shape
    data        /= np.mean(data, axis=0)
    flatten_data = np.sort(data.flatten())
    vmin, vmax   = flatten_data[int(h * w / exposure_cut)], flatten_data[int(h * w / exposure_cut * (exposure_cut - 1))]
    data         = np.clip(data, vmin, vmax)
    data         = (data - data.min()) / (data.max() - data.min())
    return data

def plot_burst(data, filename, block):
    fig          = plt.figure(figsize=(5, 5))
    gs           = gridspec.GridSpec(4, 1)

    w, h         = data.shape
    profile      = np.mean(data, axis=1)
    time_start   = ((fits_number - 1) * file_leng + block * block_size) * time_reso
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
    down_sampling_rate        = 8
    section_size              = 100
    DM                        = 100
    prob                      = 0.6
    block_size                = 512

    frb, date                 = '', ''
    model_path                = ''
    date_path                 = ''
    save_path                 = ''
    if not os.path.exists(save_path): os.makedirs(save_path)

    section                   = int(sys.argv[1])
    file_list                 = np.sort(os.listdir(date_path))
    file_list                 = np.sort(file_list[section*section_size: (section+1)*section_size])

    ### file params read
    with fits.open(date_path + file_list[0]) as f:
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

    comb_leng                 = int(dds.max() / file_leng) + 1
    comb_file_leng            = (file_leng + redundancy + dds.max()) * down_sampling_rate
    down_file_leng            = file_leng + redundancy

    ### model config
    model                     = tf.keras.models.load_model(model_path)

    ### read data
    for i in range(len(file_list)):
        raw_data              = load_fits_file(date_path + file_list[i], reverse_flag)
        fits_number           = section_size * section + i + 1
        filename              = file_list[i].split('.fits')[0]
        print(filename)

        for j in range(comb_leng):
            if i + j + 1      < len(file_list):
                raw_data      = np.append(raw_data, load_fits_file(date_path + file_list[i+j+1], reverse_flag), axis=0)
        if raw_data.shape[0]  < comb_file_leng:
            raw_data          = np.append(raw_data, np.ones((comb_file_leng-raw_data.shape[0], 2, freq_reso)), axis=0)

        raw_data              = raw_data[:comb_file_leng, :, :]
        data                  = np.mean(raw_data.reshape(raw_data.shape[0] // down_sampling_rate, down_sampling_rate, 2, freq_reso), axis=(1, 2)).astype(np.float32)

        new_data              = np.zeros((down_file_leng, freq_reso))
        for j in range(freq_reso):
            new_data[:, j]    = data[dds[j]: dds[j]+down_file_leng, j]
        data                  = np.mean(new_data.reshape(down_file_leng//512, 512, 512, freq_reso//512), axis=3)

        ### predict
        for j in range(data.shape[0]):
            data[j, :, :]     = preprocess_data(data[j, :, :])
        predict_res           = model.predict(data)

        ### plot
        blocks                = np.where(predict_res[:, 1]>=prob)[0]
        for block in blocks:
            plotres           = plot_burst(data[block], filename, block)
            np.save('{}{}-{:0>4d}.npy'.format(save_path, filename, block), data[block])
