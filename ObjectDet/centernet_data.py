import os, cv2, torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
# import albumentations as A


input_size  = 512
model_scale = 4
data_path   = './Data/'


if False:
    def draw_gaussian(heatmap, center, w_radius, h_radius, k=1):

        theta    = np.arctan2(h_radius, w_radius)
        h, w     = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x  = np.sqrt((w / 6)**2 + (h / 6)**2) / 2
        gaussian = gaussian2D((h, w), sigma_x, theta)

        x, y            = int(center[0]), int(center[1])
        height, width   = heatmap.shape[0: 2]
        left, right     = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom     = min(y, h_radius), min(height - y, h_radius + 1)
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
             np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap


    def gaussian2D(shape, sx, theta):

        ## 短轴是长轴的一半
        sy = sx / 2
        a  =   np.cos(theta)**2  / (2 * sx**2) + np.sin(theta)**2  / (2 * sy**2)
        b  = - np.sin(theta * 2) / (4 * sx**2) + np.sin(theta * 2) / (4 * sy**2)
        c  =   np.sin(theta)**2  / (2 * sx**2) + np.cos(theta)**2  / (2 * sy**2)

        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m: m + 1, -n: n + 1]
        h = np.exp(- (a * x**2 + 2 * b * x * y + c * y**2))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h


if False:
    def draw_gaussian(heatmap, center, w_radius, h_radius, k=1):

        radius   = np.min([w_radius, h_radius])
        diameter = 2 * radius + 1
        gaussian = gaussian2D((diameter, diameter), sigma=diameter/6)

        x, y            = int(center[0]), int(center[1])
        height, width   = heatmap.shape[0: 2]
        left, right     = min(x, radius), min(width - x, radius + 1)
        top, bottom     = min(y, radius), min(height - y, radius + 1)
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap


    def gaussian2D(shape, sigma=1):

        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m: m + 1, -n: n + 1]
        h    = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h


if True:
    # 中心点变热力图
    def draw_gaussian(heatmap, center, w_radius, h_radius):

        sigma    = np.clip(w_radius * h_radius // 2000, 2, 4)
        tmp_size = sigma * 6
        mu_x     = int(center[0] + 0.5)
        mu_y     = int(center[1] + 0.5)
        w, h     = heatmap.shape[0], heatmap.shape[1]
        ul       = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br       = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
            return heatmap

        size  = 2 * tmp_size + 1
        x     = np.arange(0, size, 1, np.float32)
        y     = x[:, np.newaxis]
        x0    = y0 = size // 2
        g     = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        g_x   = max(0, -ul[0]), min(br[0], h) - ul[0]
        g_y   = max(0, -ul[1]), min(br[1], w) - ul[1]
        img_x = max(0, ul[0]), min(br[0], h)
        img_y = max(0, ul[1]), min(br[1], w)
        heatmap[img_y[0]: img_y[1], img_x[0]: img_x[1]] = np.maximum(heatmap[img_y[0]: img_y[1], img_x[0]: img_x[1]], g[g_y[0]: g_y[1], g_x[0]:     g_x[1]])

        return heatmap


def make_data(target):

    output_shape = input_size // model_scale
    hm       = np.zeros([1, output_shape, output_shape])
    wh       = np.zeros([2, output_shape, output_shape])
    reg      = np.zeros([2, output_shape, output_shape])
    reg_mask = np.zeros([1, output_shape, output_shape])

    ## 如果没有目标，那么target应该是小于0的
    if len(target) == 1 and target[0, 0] < 0:
        return np.vstack((hm, wh, reg, reg_mask))

    ## 如果有目标，计算高斯，w和h在开始已经乘2
    for i in range(len(target)):

        x, y, w, h  = target[i]
        x, y, w, h  = x / model_scale, y / model_scale, w / model_scale, h / model_scale

        ct          = np.array([x, y], dtype=np.float32)
        ct_int      = ct.astype(np.int32)
        hm[:, :, :] = draw_gaussian(hm[0], ct_int, int(w), int(h))

        wh[:, ct_int[1], ct_int[0]]       =  1 * w, 1 * h
        reg[:, ct_int[1], ct_int[0]]      =  ct - ct_int
        reg_mask[:, ct_int[1], ct_int[0]] = 1

    return np.vstack((hm, wh, reg, reg_mask))


class Normalize(object):

    def __init__(self):

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

    def __call__(self, image):

        image  = image.astype(np.float32)
        image -= self.mean
        image /= self.std

        return image


class BurstDataset(torch.utils.data.Dataset):

    def __init__(self, img_id, labels, val=False, transform=None):

        self.img_id = img_id
        self.labels = labels
        if transform:
            self.transform = transform
        self.normalize = Normalize()
        self.val = val

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):

        img, target = self.load_img(idx) # 读取数据
        img = self.normalize(img)
        img = img.transpose([2, 0, 1]) # RGB转到第一个通道
        targ_gt = make_data(target) # 构建目标数据

        return img, targ_gt

    def load_img(self, idx):

        if self.val:
            img, target = self.load_raw_data(idx)
            img, target = self.down_samp(img, target)
        else:
            img, target = self.random_comb(idx)
        img = self.render_color(img)

        return img, target

    def load_img_old(self, idx):

        img, target      = self.load_raw_data(idx)
        if (len(target) == 1 and target[0, 0] == -1) or np.random.rand() < 0.5 or self.val:
            img, target = self.down_samp(img, target)
        else:
            img, target = self.random_clip(img, target)
        img             = self.render_color(img)

        return img, target

    def load_raw_data(self, idx):

        frb_path = self.img_id[idx].split('__')[0] + '.npy'
        freq_sli = int(self.img_id[idx].split('__')[1].split('.npy')[0])

        ## 读取数据和box，x时间，y色散
        img  = np.load(os.path.join(data_path, frb_path))[freq_sli]
        img      = (img - np.min(img)) / (np.max(img) - np.min(img))
        img      = (img - np.mean(img)) / np.std(img)
        target   = self.labels[self.labels['image_id']==self.img_id[idx]]
        target   = target.loc[:, ['x', 'y', 'w', 'h']].values

        return img, target

    def down_samp(self, img, target):

        img                  = np.mean(img.reshape(512, 2, 2048, 4), axis=(1, 3))
        img                  = cv2.resize(img, (input_size, input_size))
        ## 如果没有目标，那么target就会是-1/16或者-1/2
        target[:, [0, 2]]   /= 16
        target[:, [1, 3]]   /= 2

        return img, target

    def random_clip(self, img, target):

        ## 随机裁剪，计算图像中所有目标框的坐标最小与最大值
        time_min, time_max   = np.min(target[:, 0] - target[:, 2] / 2), np.max(target[:, 0] + target[:,     2] / 2)
        dm_min, dm_max       = np.min(target[:, 1] - target[:, 3] / 2), np.max(target[:, 1] + target[:,     3] / 2)
        ## 随机选取图像边缘到目标之间的区间
        time_start, time_end = np.random.randint(0, int(time_min)), np.random.randint(int(time_max), 8192)
        dm_start, dm_end     = np.random.randint(0, int(dm_min)), np.random.randint(int(dm_max), 1024)
        img                  = img[dm_start: dm_end, time_start: time_end]
        ## 由于坐标0点变化导致目标框的转变
        target[:, 0]        -= time_start
        target[:, 1]        -= dm_start
        ## 根据剩余数据的长宽计算下采样率
        down_time_rate       = int(np.ceil((time_end - time_start) / 2048))
        down_dm_rate         = int(np.ceil((dm_end - dm_start) / 512))
        ## 下采样
        img                  = img[:img.shape[0] - img.shape[0] % down_dm_rate, :img.shape[1] - img.shape   [1] % down_time_rate]
        img                  = np.mean(img.reshape(
            img.shape[0] // down_dm_rate, down_dm_rate,
            img.shape[1] // down_time_rate, down_time_rate
        ), axis=(1, 3))
        ## 只需要考虑下采样率
        target[:, [0, 2]]   /= (down_time_rate * (img.shape[1] / input_size))
        target[:, [1, 3]]   /= (down_dm_rate   * (img.shape[0] / input_size))
        img                  = cv2.resize(img, (input_size, input_size))
        return img, target

    def random_comb(self, idx):

        comb_num = np.random.randint(1, 6)
        comb_idx = np.append([idx], np.random.choice(len(self.img_id), comb_num - 1, replace=False))

        img_list, target_list = [], []
        for i in range(comb_num):
            img, target = self.load_raw_data(comb_idx[i])
            if (len(target) == 1 and target[0, 0] == -1) or np.random.rand() < 0.5:
                img, target = self.down_samp(img, target)
            else:
                img, target = self.random_clip(img, target)
            img_list.append(img)
            target_list.append(target)

        if comb_num == 4 and np.random.rand() > 0.5:
            comb_data                 = np.zeros((1024, 1024))
            targ_data                 = []
            for i in range(comb_num):
                row, col              = i // 2, i % 2
                comb_data[512*row: 512*(row+1), 512*col: 512*(col+1)] = img_list[i]
                target_i              = target_list[i]
                if len(target_i) == 1 and target_i[0, 0] < 0: continue
                target_i[:, 0]       += 512 * col
                target_i[:, 1]       += 512 * row
                targ_data.append(target_i)
            comb_data                 = np.mean(comb_data.reshape(512, 2, 512, 2), axis=(1, 3))
            if len(targ_data) == 0: targ_data = [np.array([-1, -1, -1, -1], dtype=np.float32)]
            targ_data                 = np.vstack(targ_data)
            targ_data[:, [0, 2]]     /= 2
            targ_data[:, [1, 3]]     /= 2
        else:
            if np.random.rand() > 0.5:
                comb_data             = np.zeros((512, 512*comb_num))
                targ_data             = []
                for i in range(comb_num):
                    comb_data[:, i*512: (i+1)*512] = img_list[i]
                    target_i          = target_list[i]
                    if len(target_i) == 1 and target_i[0, 0] < 0: continue
                    target_i[:, 0]   += 512 * i
                    targ_data.append(target_i)
                comb_data             = np.mean(comb_data.reshape(512, 512, comb_num), axis=2)
                if len(targ_data) == 0: targ_data = [np.array([-1, -1, -1, -1], dtype=np.float32)]
                targ_data             = np.vstack(targ_data)
                targ_data[:, [0, 2]] /= comb_num
            else:
                comb_data = np.zeros((512*comb_num, 512))
                targ_data             = []
                for i in range(comb_num):
                    comb_data[i*512: (i+1)*512, :] = img_list[i]
                    target_i          = target_list[i]
                    if len(target_i) == 1 and target_i[0, 0] < 0: continue
                    target_i[:, 1]   += 512 * i
                    targ_data.append(target_i)
                comb_data = np.mean(comb_data.reshape(512, comb_num, 512), axis=1)
                if len(targ_data) == 0: targ_data = [np.array([-1, -1, -1, -1], dtype=np.float32)]
                targ_data             = np.vstack(targ_data)
                targ_data[:, [1, 3]] /= comb_num

        return comb_data, targ_data

    def render_color(self, img):

        img = np.clip(img, *np.percentile(img, (0.1, 99.9)))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = plt.get_cmap('mako')(img)
        img = img[..., :3]
        # img = (img * 255).astype(np.uint8)

        return img