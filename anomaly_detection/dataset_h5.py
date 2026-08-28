"""
HDF5 Dataset for Radio Telescope Anomaly Detection

HDF5 格式：
    images    : (N, 512, 512) float32, 已归一化到 [0, 1]
    names     : (N, 1) string, 原始 FITS 文件路径
    down_time : (N, 1) int, 时间下采样倍率
    slice_idx : (N, 1) int, 每个文件每个倍率下的 slice 序号
"""

import os
from typing import Optional, Callable, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

import h5py
import numpy as np

# =============================================================================
# Data Augmentation
# =============================================================================

class DataAugmentation:
    """时频谱图数据增强 (numpy HxW)"""

    def __init__(
        self,
        flip_time: bool = True,
        flip_freq: bool = False,
        noise_std: float = 0.02,
        time_shift: float = 0.1,
        freq_mask_prob: float = 0.03,
    ):
        self.flip_time = flip_time
        self.flip_freq = flip_freq
        self.noise_std = noise_std
        self.time_shift = time_shift
        self.freq_mask_prob = freq_mask_prob

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # Time flip (horizontal)
        if self.flip_time and np.random.rand() > 0.5:
            data = np.flip(data, axis=1).copy()

        # Frequency flip (vertical) - 默认关闭，频率有物理意义
        if self.flip_freq and np.random.rand() > 0.5:
            data = np.flip(data, axis=0).copy()

        # Gaussian noise
        if self.noise_std > 0:
            data = data + np.random.randn(*data.shape).astype(np.float32) * self.noise_std

        # Time shift (circular)
        if self.time_shift > 0:
            max_shift = int(self.time_shift * data.shape[1])
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                if shift != 0:
                    data = np.roll(data, shift, axis=1)

        # Frequency masking (random rows)
        if self.freq_mask_prob > 0:
            mask = np.random.rand(data.shape[0]) < self.freq_mask_prob
            if mask.any():
                data = data.copy()
                data[mask, :] = 0.0

        return np.clip(data, 0.0, 1.0).astype(np.float32)


# =============================================================================
# HDF5 Dataset
# =============================================================================

class H5RadioDataset(Dataset):
    """
    从 HDF5 文件读取预处理后的射电数据

    Args:
        h5_path: HDF5 文件路径
        indices: 样本索引 (None = 使用全部)
        transform: 数据增强
        return_meta: 是否返回元信息
    """

    def __init__(
        self,
        h5_path: str,
        indices: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
    ):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")

        self.h5_path = h5_path
        self.transform = transform
        self.return_meta = return_meta
        self._h5: Optional[h5py.File] = None

        # 获取数据集大小
        with h5py.File(h5_path, "r") as f:
            n_total = f["images"].shape[0]

        self.indices = np.arange(n_total, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

        if self.indices.size == 0:
            raise ValueError("No samples in dataset")

        print(f"[Dataset] {self.indices.size} samples from {os.path.basename(h5_path)}")

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, idx: int):
        h5 = self._get_h5()
        i = int(self.indices[idx])

        # Load image
        data = h5["images"][i].astype(np.float32)

        # Apply transform
        if self.transform is not None:
            data = self.transform(data)
        else:
            data = np.clip(data, 0.0, 1.0).astype(np.float32)

        # (H, W) -> (1, H, W)
        tensor = torch.from_numpy(data).unsqueeze(0)

        if not self.return_meta:
            return tensor

        # Load metadata
        name_raw = h5["names"][i][0]
        name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)

        meta = {
            "name": name,
            "down_time": int(h5["down_time"][i][0]),
            "slice_idx": int(h5["slice_idx"][i][0]),
        }
        return tensor, meta


# =============================================================================
# Pair Dataset for Temporal Prediction
# =============================================================================

class H5PairDataset(Dataset):
    """
    时间预测数据集：返回相邻两帧 (x_{t-1}, x_t)

    - 按 (fits_dir, down_time) 分组
    - 组内按 (filename, slice_idx) 排序（时间连续）
    - 相邻 slice 组成一对

    用于训练 f(x_{t-1}) = x_t 的预测模型
    """

    def __init__(
        self,
        h5_path: str,
        down_time_list: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")

        self.h5_path = h5_path
        self.transform = transform
        self._h5: Optional[h5py.File] = None

        # 读取元数据
        with h5py.File(h5_path, "r") as f:
            n_total = f["images"].shape[0]
            names_raw = f["names"][:].squeeze()  # (N,)
            down_time = f["down_time"][:].squeeze().astype(np.int32)  # (N,)
            slice_idx = f["slice_idx"][:].squeeze().astype(np.int32)  # (N,)

        # 解码 names
        names = []
        for name in names_raw:
            if isinstance(name, bytes):
                names.append(name.decode("utf-8"))
            else:
                names.append(str(name))
        names = np.array(names)

        # 提取 fits_dir 和 filename
        fits_dirs = np.array([os.path.dirname(n) for n in names])
        filenames = np.array([os.path.basename(n) for n in names])

        # 按 down_time 过滤
        if down_time_list is not None:
            down_time_list = np.array(down_time_list, dtype=np.int32)
            mask = np.isin(down_time, down_time_list)
            valid_idx = np.nonzero(mask)[0]
        else:
            valid_idx = np.arange(n_total)

        # 按 (fits_dir, down_time, filename, slice_idx) 排序
        fits_dirs = fits_dirs[valid_idx]
        filenames = filenames[valid_idx]
        down_time = down_time[valid_idx]
        slice_idx = slice_idx[valid_idx]

        # lexsort 最后一个是主排序键（需要反着写）
        order = np.lexsort((slice_idx, filenames, down_time, fits_dirs))

        # 重排
        fits_dirs = fits_dirs[order]
        down_time = down_time[order]
        sorted_idx = valid_idx[order]

        # 构造相邻对：同一 (fits_dir, down_time) 内的相邻样本
        self.pairs = []
        n = len(sorted_idx)
        for i in range(n - 1):
            # 只有同一组内的才配对
            if fits_dirs[i] == fits_dirs[i + 1] and down_time[i] == down_time[i + 1]:
                self.pairs.append((sorted_idx[i], sorted_idx[i + 1]))

        print(f"[H5PairDataset] {len(self.pairs)} pairs from {n_total} samples")

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        h5 = self._get_h5()
        i_prev, i_curr = self.pairs[idx]

        # 加载两帧
        x_prev = h5["images"][i_prev].astype(np.float32)
        x_curr = h5["images"][i_curr].astype(np.float32)

        # 数据增强（两帧使用相同的随机变换）
        if self.transform is not None:
            # 保存随机状态
            state = np.random.get_state()
            x_prev = self.transform(x_prev)
            np.random.set_state(state)
            x_curr = self.transform(x_curr)
        else:
            x_prev = np.clip(x_prev, 0.0, 1.0).astype(np.float32)
            x_curr = np.clip(x_curr, 0.0, 1.0).astype(np.float32)

        # (H, W) -> (1, H, W)
        x_prev = torch.from_numpy(x_prev).unsqueeze(0)
        x_curr = torch.from_numpy(x_curr).unsqueeze(0)

        return x_prev, x_curr


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_h5_dataloaders(
    h5_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    down_time_list: Optional[List[int]] = None,
    augment: bool = True,
    seed: int = 42,
    return_meta: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建 train/val DataLoader

    Args:
        h5_path: HDF5 文件路径
        batch_size: 批次大小
        num_workers: 数据加载进程数
        val_ratio: 验证集比例 (0 = 不创建验证集)
        down_time_list: 只使用指定的下采样率 (None = 全部)
        augment: 是否对训练集做数据增强
        seed: 随机种子
        return_meta: 是否返回元信息

    Returns:
        (train_loader, val_loader)
    """
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    # Filter by down_time
    with h5py.File(h5_path, "r") as f:
        n_total = f["images"].shape[0]
        if down_time_list is None:
            indices = np.arange(n_total, dtype=np.int64)
        else:
            down = np.asarray(f["down_time"][:]).reshape(-1)
            indices = np.where(np.isin(down, down_time_list))[0].astype(np.int64)
            print(f"[Filter] down_time {down_time_list}: {indices.size}/{n_total} samples")

    if indices.size == 0:
        raise ValueError("No samples after filtering")

    # Train/Val split
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_val = int(indices.size * val_ratio)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    print(f"[Split] Train: {train_indices.size}, Val: {n_val}")

    # Create datasets
    train_dataset = H5RadioDataset(
        h5_path,
        indices=train_indices,
        transform=DataAugmentation() if augment else None,
        return_meta=return_meta,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if n_val > 0:
        val_dataset = H5RadioDataset(
            h5_path,
            indices=val_indices,
            transform=None,
            return_meta=return_meta,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test augmentation
    aug = DataAugmentation()
    test = np.random.rand(512, 512).astype(np.float32)
    out = aug(test)
    print(f"Augmentation: {test.shape} -> {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]")
