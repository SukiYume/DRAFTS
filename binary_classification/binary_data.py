"""Binary burst classifier 的 H5 数据集 + 数据增强。

数据来源：``True_*.h5`` / ``False_*.h5``，每个文件含 ``images / labels / idx / names``，可选
``sources``。``label == 1`` 表示真 burst（正样本），``label == 0`` 表示假信号（负样本）。

训练集分层切分 + 过采样：
  - 按 label 做分层 ``train_test_split``，避免类别失衡
  - ``True_LPT.h5`` 中匹配 ``^YYYYMMDD-(02|05|10)`` 且 source≠"True_LPT_Old" 的样本视为「新 LPT」，
    重复 6 倍；``False_*.h5`` 负样本重复 2 倍；其余 1 倍
  - 验证集不重复

数据增强（仅训练）：
  - 50% 概率走 max-mixup（两图逐像素取 max，label 取或）
  - 50% 概率走「随机拼图」（2~5 张图 2×2 / 1×n / n×1 平均下采样回 512²，label 取或）
  - 然后 ``preprocess_data``（含一定概率合成假信号 ``add_noise``）
  - 最后 RandomRotation / VerticalFlip / HorizontalFlip
"""

import os
import re

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# 1. 数据文件清单 + 目录读取
# ---------------------------------------------------------------------------

POSITIVE_H5_FILES = ("True_FRB.h5", "True_LPT.h5")
NEGATIVE_H5_FILES = ("False_FRB.h5", "False_LPT.h5", "False_FRB20240114A.h5")


def _decode_array(values):
    """h5 里读出来的字符串字段可能是 bytes，统一解码为 Python str。"""
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="ignore"))
        else:
            out.append(str(value))
    return np.asarray(out)


def _read_h5_records(h5_path):
    """逐样本读 1 个 h5，输出 dataframe 行的字典列表；同时按文件名决定基础 ``repeat``。"""
    with h5py.File(h5_path, "r") as f:
        names = _decode_array(f["names"][...])
        labels = f["labels"][...].astype(np.int64)
        idxs = f["idx"][...].astype(np.int64)
        sources = _decode_array(f["sources"][...]) if "sources" in f else np.array([""] * len(names))

    base_name = os.path.basename(h5_path)
    records = []
    for i, name in enumerate(names):
        # True_LPT 里以 YYYYMMDD-(02|05|10) 开头、source 非 Old 的为「新 LPT」，强过采样
        if base_name == "True_LPT.h5":
            is_new_lpt = re.match(r"^\d{8}-(02|05|10)", name) and sources[i] != "True_LPT_Old"
            repeat = 6 if is_new_lpt else 1
        elif base_name in NEGATIVE_H5_FILES:
            repeat = 2
        else:
            repeat = 1

        records.append({
            "h5_file": h5_path,
            "local_idx": int(idxs[i]),
            "name": name,
            "label": int(labels[i]),
            "source": sources[i],
            "repeat": int(repeat),
            "sample_key": f"{base_name}:{int(idxs[i])}",       # 用来去重
        })
    return records


def build_catalog(root_path="./Data/"):
    """扫描 ``root_path`` 下声明过的 h5 文件，合并成单张 catalog dataframe（去重后）。"""
    root_path = os.path.normpath(root_path)
    records = []
    for filename in POSITIVE_H5_FILES + NEGATIVE_H5_FILES:
        h5_path = os.path.join(root_path, filename)
        if not os.path.exists(h5_path):
            print(f"[Data] Skip missing file: {h5_path}")
            continue
        records.extend(_read_h5_records(h5_path))

    if not records:
        raise FileNotFoundError(f"No binary classification H5 files found in: {root_path}")

    catalog = pd.DataFrame(records).drop_duplicates(subset=["sample_key"]).reset_index(drop=True)
    pos = int((catalog["label"] == 1).sum())
    neg = int((catalog["label"] == 0).sum())
    print(f"[Data] Unique samples: {len(catalog)} | Positive: {pos}, Negative: {neg}")
    return catalog


# ---------------------------------------------------------------------------
# 2. 分层切分 + 训练集过采样
# ---------------------------------------------------------------------------

def _repeat_train_samples(train_df, seed):
    """按 ``repeat`` 列做行级过采样：repeat=k 的行复制 k 份；并打印 before / after label 分布。"""
    before = train_df["label"].value_counts().to_dict()
    parts = []
    for repeat, group in train_df.groupby("repeat", sort=True):
        parts.extend([group] * int(repeat))
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    after = out["label"].value_counts().to_dict()

    print(f"[Data] Train label count before repeat: {before}")
    print(f"[Data] Train label count after  repeat: {after}")
    print("[Data] Repeat schedule: new True_LPT ×6, negatives ×2, others ×1; validation not repeated")
    return out


def get_train_val(root_path="./Data/", train_ratio=0.8, seed=42):
    """从 catalog 按 label 分层切分；切分先于过采样完成，避免 train/val 泄漏。"""
    catalog = build_catalog(root_path)
    train_df, val_df = train_test_split(
        catalog,
        train_size=train_ratio,
        shuffle=True,
        stratify=catalog["label"],
        random_state=seed,
    )
    train_df = _repeat_train_samples(train_df.reset_index(drop=True), seed)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"[Data] Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    return train_df, val_df


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------

class BurstDataset(Dataset):
    """H5-backed 二分类数据集。

    Dataframe 必须包含 ``h5_file``、``local_idx``、``label`` 列（``build_catalog`` 默认满足）。

    训练模式（``val=False``）：
      - 50% mixup（两图逐像素 max，label 取或）/ 50% 随机拼图（rows×cols mean-pool）
      - 然后 ``preprocess_data``（含一定概率合成假信号 ``add_noise``）
      - 最后 tensor 级的 rotation / flip 增强

    验证模式（``val=True``）：单图加载，不做任何随机增强。
    """

    def __init__(self, data, val=False):
        self.cat = data.reset_index(drop=True)
        self.val = bool(val)
        self.h5_files = {}                                  # worker-local h5 句柄缓存
        self.trans = transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    # ---- 主入口 ------------------------------------------------------------

    def __len__(self):
        return len(self.cat)

    def __getitem__(self, idx):
        if self.val:
            x, y = self._load_single(idx)
        else:
            # 训练：一半 mixup、一半「随机拼图」
            x, y = self._mixup(idx) if np.random.rand() > 0.5 else self._random_comb(idx)
        x = torch.from_numpy(x[None, :, :].astype(np.float32))      # [1, H, W]
        if not self.val:
            x = self.trans(x)
        return x, torch.tensor(y, dtype=torch.long)

    # ---- 读盘 + 预处理 -----------------------------------------------------

    def _load_single(self, row_id):
        """单帧加载：从 h5 取原始图 → ``preprocess_data`` 归一到 [0, 1]。"""
        row = self.cat.loc[int(row_id)]
        h5_path = row["h5_file"]
        if h5_path not in self.h5_files:
            self.h5_files[h5_path] = h5py.File(h5_path, "r")
        x = self.h5_files[h5_path]["images"][int(row["local_idx"])].copy()
        return self.preprocess_data(x), int(row["label"])

    def preprocess_data(self, data):
        """归一化前的预处理：

        1. （仅训练时一定概率）``add_noise`` 合成假信号
        2. 减去频谱基线（除以每列均值）
        3. 按对称分位截断 + 0/1 归一
        """
        data = data.copy()
        # 验证：固定 5% 分位截断；训练：[0, 20] 之间随机
        exp_cut = 5 if self.val else np.random.rand() * 20

        if not self.val and np.random.rand() > 0.5:
            data = self.add_noise(data)

        data = data + 1.0
        data /= np.mean(data, axis=0)                           # 行（=不同时间）方向去基线
        vmin, vmax = np.nanpercentile(data, [exp_cut, 100 - exp_cut])
        data = np.clip(data, vmin, vmax)
        return (data - data.min()) / (data.max() - data.min() + 1e-8)

    # ---- 训练时的样本合成：mixup + 随机拼图 --------------------------------

    def _mixup(self, idx):
        """两图逐像素 max（保留更亮的特征），label = label1 | label2。"""
        if len(self.cat) < 2:
            return self._load_single(idx)
        x1, y1 = self._load_single(idx)
        other_idx = np.random.choice(np.delete(np.arange(len(self.cat)), idx))
        x2, y2 = self._load_single(other_idx)
        return np.max([x1, x2], axis=0), y1 | y2

    def _random_comb(self, idx):
        """2~5 张图按 2×2 / 1×n / n×1 拼接后 mean-pool 回 512²。label = 任一 == 1 即正。"""
        comb_num = min(np.random.randint(1, 6), len(self.cat))
        if comb_num == 1:
            return self._load_single(idx)

        other = np.random.choice(np.delete(np.arange(len(self.cat)), idx), comb_num - 1, replace=False)
        indices = np.append([idx], other)
        imgs, labels = [], []
        for i in indices:
            x, y = self._load_single(int(i))
            imgs.append(x)
            labels.append(y)

        if comb_num == 4 and np.random.rand() > 0.5:
            rows, cols = 2, 2
        elif np.random.rand() > 0.5:
            rows, cols = 1, comb_num
        else:
            rows, cols = comb_num, 1

        canvas = np.zeros((512 * rows, 512 * cols), dtype=np.float32)
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            canvas[512 * r: 512 * (r + 1), 512 * c: 512 * (c + 1)] = imgs[i]
        img = canvas.reshape(512, rows, 512, cols).mean(axis=(1, 3))
        label = 1 if any(l == 1 for l in labels) else 0
        return img, label

    # ---- 假信号合成（仅训练；用于让分类器学会拒绝 RFI / 周期脉冲 / 色散条纹等）

    def add_noise(self, data):
        """以一定概率链式应用多种合成「假信号」，模拟 RFI / 周期脉冲 / 色散条纹等噪声模式。"""
        dmax, dmin = data.max(), data.min()
        ops = [
            self._noise_vertical_band_resize,
            self._noise_oblique_streaks,
            self._noise_dispersion_traces,
            self._noise_periodic_blocks,
            self._noise_dense_slope,
            self._noise_column_gradient,
            self._noise_horizontal_pixels,
            self._noise_scatter_pixels,
            self._noise_multiplicative_gauss,
        ]
        for op in ops:
            data = op(data, dmax, dmin)
        return data

    # 每个 _noise_* 都按论文/经验给定一个触发概率；触发后改写 data 并返回。返回值即下一步输入。

    @staticmethod
    def _noise_vertical_band_resize(data, dmax, dmin):
        """70% 概率：在随机列插入纵向零带后再 resize 回 512²，模拟时间方向的间断。"""
        if np.random.rand() <= 0.3:
            return data
        start = np.random.randint(0, 512)
        length = np.random.randint(10, 200)
        data = np.insert(data, start, np.zeros((length, 512)), axis=1)
        return resize(data, (512, 512))

    @staticmethod
    def _noise_oblique_streaks(data, dmax, dmin):
        """20% 概率：从某一行向左下/右下方画若干斜线（RFI 残留）。"""
        if np.random.rand() <= 0.8:
            return data
        for _ in range(np.random.randint(1, 5)):
            start = np.random.randint(0, 512)
            slope = np.random.rand() * 5 + 0.5
            for j in range(start):
                data[
                    start - j: start + np.random.randint(1, 10) - j,
                    int(511 - j / slope),
                ] = np.random.rand() * dmax / 2 + np.random.rand() + dmax / 100
        return data

    @staticmethod
    def _noise_dispersion_traces(data, dmax, dmin):
        """60% 概率：按 DM = 50~1050 的色散公式合成弯曲轨迹（最像真信号的'陷阱'）。"""
        if np.random.rand() <= 0.4:
            return data
        dm = np.random.rand() * 1000 + 50
        freq = np.linspace(1000, 1500, 512)
        delay = (4.15 * dm * (freq ** -2 - freq.max() ** -2) * 1e3 / (49.152 * 1e-6) / 8).astype(np.int64)
        for _ in range(np.random.randint(1, 8)):
            insert_start = np.random.randint(0, int(np.max(delay)))
            length = np.random.randint(2, 20)
            single_delay = -delay[::-1] + insert_start
            for j in range(512):
                if 0 <= single_delay[j] <= 511:
                    data[single_delay[j]: single_delay[j] + length, 511 - j] = (
                        np.random.rand() * dmax / 2 + np.random.rand() + dmax / 100
                    )
        return data

    @staticmethod
    def _noise_periodic_blocks(data, dmax, dmin):
        """60% 概率：在一段时间内按 1~9 个等距块写入亮斑（周期脉冲）。"""
        if np.random.rand() <= 0.4:
            return data
        for _ in range(np.random.randint(1, 5)):
            period = np.random.randint(1, 10)
            insert_start = np.random.randint(0, 512)
            length = np.random.randint(1, 10)
            for j in range(period):
                xs = np.random.randint(512 // period * j, 512 // period * (j + 1))
                xl = np.random.randint(512 // period // 5, 512 // period // 3 * 2)
                block = (
                    np.min([512 - xs, xl]),
                    np.min([512 - insert_start, length]),
                )
                data[xs: xs + xl, insert_start: insert_start + length] = np.random.rand(*block) * 0.5 + dmax
        return data

    @staticmethod
    def _noise_dense_slope(data, dmax, dmin):
        """40% 概率：在一个梯形区域内大量画斜线，模拟连续色散条纹的密集情况。"""
        if np.random.rand() <= 0.6:
            return data
        for _ in range(np.random.randint(1, 2)):
            start = np.random.randint(0, 512)
            length = np.random.randint(10, 100)
            slope = np.random.rand() * 5 + 0.5
            for _ in range(np.random.randint(1, 512)):
                xs = np.random.randint(1, 512)
                xl = np.random.randint(1, length)
                for k in range(xl):
                    if xs + k > 511 or int(start + k / slope) > 511:
                        continue
                    data[xs + k, int(start + k / slope)] = np.random.rand() * dmax * 2
        return data

    @staticmethod
    def _noise_column_gradient(data, dmax, dmin):
        """30% 概率：选若干列乘上一段连续渐变（模拟某通道增益漂移）。"""
        if np.random.rand() <= 0.7:
            return data
        for _ in range(np.random.randint(1, 5)):
            start = np.random.randint(0, 512)
            length = np.random.randint(1, 10)
            f = interp1d(
                [0, 511],
                [(np.random.rand() * 5 + 0.5) * dmax, (np.random.rand() * 5 + 0.5) * dmin],
            )
            y = f(np.arange(0, 512))
            if np.random.rand() < 0.5:
                y = y[::-1]
            data[:, start: start + length] = data[:, start: start + length] * y[:, np.newaxis]
        return data

    @staticmethod
    def _noise_horizontal_pixels(data, dmax, dmin):
        """80% 概率：在若干行随机位置写入孤立亮像素，模拟尖刺 RFI。"""
        if np.random.rand() <= 0.2:
            return data
        num = np.random.randint(10, 200)
        start = np.random.randint(0, 512)
        length = np.random.randint(1, 10)
        for _ in range(num):
            xs = np.random.randint(0, 512)
            data[xs, min(511, start + np.random.randint(0, length))] = (np.random.rand() * 10 + 1) * dmax
        return data

    @staticmethod
    def _noise_scatter_pixels(data, dmax, dmin):
        """70% 概率：撒一把孤立亮点（pepper-style 强噪声）。"""
        if np.random.rand() <= 0.3:
            return data
        for _ in range(np.random.randint(10, 200)):
            data[np.random.randint(0, 512), np.random.randint(0, 512)] = (np.random.rand() * 10 + 1) * dmax
        return data

    @staticmethod
    def _noise_multiplicative_gauss(data, dmax, dmin):
        """60% 概率：整张图按 N(1, 0.2) 做乘法噪声，整体亮度抖动。"""
        if np.random.rand() <= 0.4:
            return data
        return data * np.abs(np.random.normal(loc=1, scale=0.2, size=(512, 512)))

    # ---- 资源清理 ----------------------------------------------------------

    def close(self):
        for h5f in self.h5_files.values():
            try:
                h5f.close()
            except Exception:
                pass
        self.h5_files.clear()

    def __del__(self):
        self.close()
