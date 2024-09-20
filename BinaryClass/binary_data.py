import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class BurstDataset(Dataset):

    def __init__(self, data, val=False):
        self.image = data.file_name.values
        self.label = data.label.values
        self.val   = val
        ## ToTensor时，会将二维数组变为三维数组，且其它的transforms需要PIL格式的数据
        self.trans = transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        X, y       = self.random_comb_data(idx) # load_single_data
        X          = torch.from_numpy(X[None, :, :].astype(np.float32))
        y          = torch.tensor(y, dtype=torch.long) # for one class - torch.tensor([y], dtype=torch.float32)
        if not self.val:
            X      = self.trans(X)
        return X, y

    def preprocess_data(self, data, mean_norm=True, exp_cut=5):
        data       = data.copy()
        if np.random.rand() > 0.5:
            data   = self.add_noise(data)
        data      += 1
        if mean_norm:
            data  /= np.mean(data, axis=0)
        vmin, vmax = np.nanpercentile(data, [exp_cut, 100-exp_cut])
        data       = np.clip(data, vmin, vmax)
        data       = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def load_single_data(self, idx):
        data, label = np.load(self.image[idx]), self.label[idx]
        if np.abs(data.max() - data.min() - 1) < 1:
            mean_norm, exp_cut = False, np.random.rand() * 15
        else:
            mean_norm, exp_cut = True, np.random.rand() * 20
        data = self.preprocess_data(data, mean_norm=mean_norm, exp_cut=exp_cut)
        return data, label

    def random_comb_data(self, idx):
        comb_num = np.random.randint(1, 6)
        comb_idx = np.append([idx], np.random.choice(len(self.image), comb_num - 1, replace=False))
        if comb_num == 4 and np.random.rand() > 0.5:
            comb_data = np.zeros((1024, 1024))
            for i in range(comb_num):
                data, _ = self.load_single_data(comb_idx[i])
                row, col = i // 2, i % 2
                comb_data[512*row: 512*(row+1), 512*col: 512*(col+1)] = data
            comb_data = np.mean(comb_data.reshape(512, 2, 512, 2), axis=(1, 3))
        else:
            if np.random.rand() > 0.5:
                comb_data = np.zeros((512, 512*comb_num))
                for i in range(comb_num):
                    data, _ = self.load_single_data(comb_idx[i])
                    comb_data[:, i*512:(i+1)*512] = data
                comb_data = np.mean(comb_data.reshape(512, 512, comb_num), axis=2)
            else:
                comb_data = np.zeros((512*comb_num, 512))
                for i in range(comb_num):
                    data, _ = self.load_single_data(comb_idx[i])
                    comb_data[i*512:(i+1)*512, :] = data
                comb_data = np.mean(comb_data.reshape(512, comb_num, 512), axis=1)
        labels = 1 if np.any(np.array([self.label[j] for j in comb_idx]) == 1) else 0
        return comb_data, labels

    def add_noise(self, data):
        data_max, data_min   = data.max(), data.min()
        # 加0
        if np.random.rand() > 0.3:
            insert_start = np.random.randint(0, 512)
            insert_lengt = np.random.randint(10, 200)
            data         = np.insert(data, insert_start, np.zeros((insert_lengt, 512)), axis=1)
            data         = resize(data, (512, 512))
        # 加直线
        if np.random.rand() > 0.8:
            insert_num   = np.random.randint(1, 5)
            for _ in range(insert_num):
                insert_start = np.random.randint(0, 512)
                insert_slope = np.random.rand() * 5 + 0.5
                for j in range(insert_start):
                    data[insert_start - j: insert_start + np.random.randint(1, 10) - j, int(511 - j / insert_slope)] = np.random.rand() * data_max / 2 + np.random.rand() + data_max / 100
        # 加抛物线
        if np.random.rand() > 0.4:
            insert_num   = np.random.randint(1, 8)
            DM = np.random.rand() * 1000 + 50
            freq = np.linspace(1000, 1500, 512)
            delay = (4.15 * DM * (freq**-2 - freq.max()**-2) * 1e3 / (49.152 * 1e-6) / 8).astype(np.int64)
            for _ in range(insert_num):
                insert_start = np.random.randint(0, np.max(delay))
                insert_lengt = np.random.randint(2, 20)
                single_delay = - delay[::-1] + insert_start
                for j in range(512):
                    if (single_delay[j] < 0) or (single_delay[j] > 511): continue
                    data[single_delay[j]:single_delay[j]+insert_lengt, 511-j] = np.random.rand() * data_max / 2 + np.random.rand() + data_max / 100
        # 加横条
        if np.random.rand() > 0.4:
            insert_num   = np.random.randint(1, 5)
            for _ in range(insert_num):
                insert_perio = np.random.randint(1, 10)
                insert_start = np.random.randint(0, 512)
                insert_lengt = np.random.randint(1, 10)
                for j in range(insert_perio):
                    insert_x_start = np.random.randint(512//insert_perio*j, 512//insert_perio*(j+1))
                    insert_x_legnt = np.random.randint(512//insert_perio//5, 512//insert_perio//3*2)
                    noise_block    = (np.min([512 - insert_x_start, insert_x_legnt]), np.min([512 - insert_start, insert_lengt]))
                    data[insert_x_start: insert_x_start+insert_x_legnt, insert_start:insert_start+insert_lengt] = np.random.rand(*noise_block) * 0.5 + data_max
        # 加窄带斜线 New
        if np.random.rand() > 0.6:
            insert_num   = np.random.randint(1, 2)
            for _ in range(insert_num):
                insert_start = np.random.randint(0, 512)
                insert_lengt = np.random.randint(10, 100)
                insert_slope = np.random.rand() * 5 + 0.5
                insert_choice = np.random.randint(1, 512)
                for j in range(insert_choice):
                    insert_x_start = np.random.randint(1, 512)
                    insert_x_legnt = np.random.randint(1, insert_lengt)
                    for k in range(insert_x_legnt):
                        if (insert_x_start + k > 511) or (int(insert_start + k / insert_slope) > 511): continue
                        data[insert_x_start + k, int(insert_start + k / insert_slope)] = np.random.rand() * data_max * 2
        # 加渐变横条
        if np.random.rand() > 0.7:
            insert_num   = np.random.randint(1, 5)
            for _ in range(insert_num):
                insert_start = np.random.randint(0, 512)
                insert_lengt = np.random.randint(1, 10)
                f = interp1d([0, 511], [(np.random.rand()*5+0.5)*data_max, (np.random.rand()*5+0.5)*data_min])
                y = f(np.arange(0, 512))
                if np.random.rand() < 0.5:
                    y = y[::-1]
                data[:, insert_start: insert_start+insert_lengt] = data[:, insert_start: insert_start+insert_lengt] * y[:, np.newaxis]
        # 加窄带散点
        if np.random.rand() > 0.2:
            insert_num   = np.random.randint(10, 200)
            insert_start = np.random.randint(0, 512)
            insert_lengt = np.random.randint(1, 10)
            for _ in range(insert_num):
                insert_x_start = np.random.randint(0, 512)
                data[insert_x_start, np.min([511, insert_start+np.random.randint(0, insert_lengt)])] = (np.random.rand() * 10 + 1) * data_max
        # 加散点
        if np.random.rand() > 0.3:
            insert_num   = np.random.randint(10, 200)
            for _ in range(insert_num):
                data[np.random.randint(0, 512), np.random.randint(0, 512)] = (np.random.rand() * 10 + 1) * data_max
        # 加高斯噪声
        if np.random.rand() > 0.4:
            data = data * np.abs(np.random.normal(loc=1, scale=0.2, size=(512, 512)))
        return data


def get_train_val(root_path='./Data/'):

    positive_path = root_path + 'True/'
    positive_x    = [positive_path + i for i in os.listdir(positive_path) if i.endswith('npy')]

    negative_path = root_path + 'False/'
    negative_x    = [negative_path + i for i in os.listdir(negative_path) if i.endswith('npy')]

    x             = np.append(positive_x, negative_x)
    y             = np.append([1]*len(positive_x), [0]*len(negative_x))

    print(len(positive_x), len(negative_x))

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, shuffle=True)
    train_data = pd.DataFrame({'file_name': train_x, 'label': train_y})
    val_data   = pd.DataFrame({'file_name': val_x, 'label': val_y})

    return train_data, val_data