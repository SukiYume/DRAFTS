# fmt: off

"""HDF5 datasets for radio-telescope anomaly detection.

Expected datasets:
    images: (N, 512, 512) float32 in [0, 1], ordered as (time, frequency)
    names: (N, 1) UTF-8 source FITS paths
    down_time: (N, 1) integer time-downsampling factors
    slice_idx: (N, 1) integer slice indices within each source file
"""

import os
from collections.abc import Callable
from contextlib import suppress

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DataAugmentation:
    """Apply synchronized-safe augmentation to a (time, frequency) array."""

    def __init__(
        self,
        flip_time: bool = True,
        flip_freq: bool = False,
        noise_std: float = 0.02,
        time_shift: float = 0.1,
        freq_mask_prob: float = 0.03,
    ):
        self.flip_time      = flip_time
        self.flip_freq      = flip_freq
        self.noise_std      = noise_std
        self.time_shift     = time_shift
        self.freq_mask_prob = freq_mask_prob

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D time-frequency array, got {data.shape}")

        if self.flip_time and np.random.random() > 0.5:
            data = np.flip(data, axis=0).copy()

        if self.flip_freq and np.random.random() > 0.5:
            data = np.flip(data, axis=1).copy()

        if self.noise_std > 0:
            noise = np.random.randn(*data.shape).astype(np.float32)
            data  = data + noise * self.noise_std

        if self.time_shift > 0:
            max_shift = int(self.time_shift * data.shape[0])
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                if shift:
                    data = np.roll(data, shift, axis=0)

        if self.freq_mask_prob > 0:
            mask = np.random.random(data.shape[1]) < self.freq_mask_prob
            if mask.any():
                data          = data.copy()
                data[:, mask] = 0.0

        return np.clip(data, 0.0, 1.0).astype(np.float32, copy=False)


class H5RadioDataset(Dataset):
    """Read individual normalized time-frequency slices from HDF5."""

    def __init__(
        self,
        h5_path: str,
        indices: np.ndarray | None = None,
        transform: Callable | None = None,
        return_meta: bool = False,
    ):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")

        self.h5_path               = h5_path
        self.transform             = transform
        self.return_meta           = return_meta
        self._h5: h5py.File | None = None

        required = {"images", "names", "down_time", "slice_idx"}
        with h5py.File(h5_path, "r") as h5:
            missing = required.difference(h5.keys())
            if missing:
                raise ValueError(f"Missing HDF5 datasets: {sorted(missing)}")
            n_total = h5["images"].shape[0]
            if h5["images"].shape[1:] != (512, 512):
                raise ValueError(
                    f"Expected images shaped (N, 512, 512), got {h5['images'].shape}"
                )
            if any(h5[key].shape[0] != n_total for key in required):
                raise ValueError("HDF5 datasets have inconsistent sample counts")

        if indices is None:
            self.indices = np.arange(n_total, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)

        if self.indices.size == 0:
            raise ValueError("No samples in dataset")
        if self.indices.min() < 0 or self.indices.max() >= n_total:
            raise IndexError("Dataset indices are outside the HDF5 sample range")

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __del__(self):
        if self._h5 is not None:
            with suppress(Exception):
                self._h5.close()

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, idx: int):
        h5         = self._get_h5()
        source_idx = int(self.indices[idx])
        data       = h5["images"][source_idx].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)
        else:
            data = np.clip(data, 0.0, 1.0).astype(np.float32, copy=False)

        tensor = torch.from_numpy(data).unsqueeze(0)
        if not self.return_meta:
            return tensor

        name_raw = np.asarray(h5["names"][source_idx]).reshape(-1)[0]
        name = (
            name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
        )
        return tensor, {
            "name": name,
            "down_time": int(np.asarray(h5["down_time"][source_idx]).reshape(-1)[0]),
            "slice_idx": int(np.asarray(h5["slice_idx"][source_idx]).reshape(-1)[0]),
        }


class H5PairDataset(Dataset):
    """Return consecutive slices ``(x[t-1], x[t])`` from one FITS file.

    Pairs must have the same full source path and downsampling factor, and their
    slice indices must differ by exactly one. This prevents accidental pairing
    across files or across gaps in an incomplete HDF5 dataset.
    """

    def __init__(
        self,
        h5_path: str,
        down_time_list: list[int] | None = None,
        transform: Callable | None = None,
    ):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")

        self.h5_path               = h5_path
        self.transform             = transform
        self._h5: h5py.File | None = None

        required = {"images", "names", "down_time", "slice_idx"}
        with h5py.File(h5_path, "r") as h5:
            missing = required.difference(h5.keys())
            if missing:
                raise ValueError(f"Missing HDF5 datasets: {sorted(missing)}")

            n_total = h5["images"].shape[0]
            if h5["images"].shape[1:] != (512, 512):
                raise ValueError(
                    f"Expected images shaped (N, 512, 512), got {h5['images'].shape}"
                )
            names_raw = np.asarray(h5["names"][:]).reshape(-1)
            down_time = np.asarray(h5["down_time"][:], dtype=np.int32).reshape(-1)
            slice_idx = np.asarray(h5["slice_idx"][:], dtype=np.int32).reshape(-1)

        if not (names_raw.size == down_time.size == slice_idx.size == n_total):
            raise ValueError("HDF5 datasets have inconsistent sample counts")

        names = np.array(
            [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in names_raw
            ]
        )

        if down_time_list is None:
            valid_idx = np.arange(n_total, dtype=np.int64)
        else:
            valid_idx = np.flatnonzero(np.isin(down_time, down_time_list))

        order = np.lexsort(
            (slice_idx[valid_idx], down_time[valid_idx], names[valid_idx])
        )
        sorted_idx                        = valid_idx[order]
        self.pairs: list[tuple[int, int]] = []

        for previous, current in zip(sorted_idx[:-1], sorted_idx[1:], strict=True):
            if (
                names[previous] == names[current]
                and down_time[previous] == down_time[current]
                and slice_idx[current] == slice_idx[previous] + 1
            ):
                self.pairs.append((int(previous), int(current)))

        if not self.pairs:
            raise ValueError("No consecutive slice pairs after filtering")
        self.source_names = np.asarray([names[current] for _, current in self.pairs])

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __del__(self):
        if self._h5 is not None:
            with suppress(Exception):
                self._h5.close()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        h5                = self._get_h5()
        previous, current = self.pairs[idx]
        x_previous        = h5["images"][previous].astype(np.float32)
        x_current         = h5["images"][current].astype(np.float32)

        if self.transform is not None:
            random_state = np.random.get_state()
            x_previous   = self.transform(x_previous)
            np.random.set_state(random_state)
            x_current = self.transform(x_current)
        else:
            x_previous = np.clip(x_previous, 0.0, 1.0).astype(np.float32, copy=False)
            x_current  = np.clip(x_current, 0.0, 1.0).astype(np.float32, copy=False)

        return (
            torch.from_numpy(x_previous).unsqueeze(0),
            torch.from_numpy(x_current).unsqueeze(0),
        )

# fmt: on
