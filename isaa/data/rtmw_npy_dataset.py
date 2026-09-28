"""Fast dataset for fixed-window RTMW features written as NumPy files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RTMWNpyDataset(Dataset):
    """Read ``data.npy``, ``labels.npy`` and optional ``frame_mask.npy``.

    ``data.npy`` is stored as ``N x 3 x T x 32 x M`` and already contains
    relative x/y plus score.  NumPy memmap keeps the training process from
    copying the complete dataset during startup.
    """

    def __init__(self, directory: str | Path, *, use_frame_mask: bool = False,
                 max_samples: int = 0) -> None:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"预处理目录不存在: {directory}")
        data_path = directory / "data.npy"
        labels_path = directory / "labels.npy"
        if not data_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"预处理目录必须包含 data.npy 和 labels.npy: {directory}")
        self.data = np.load(data_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if self.data.ndim != 5 or self.data.shape[1] != 3 or self.data.shape[3] != 32:
            raise ValueError(f"data.npy 必须是 N x 3 x T x 32 x M，当前为 {self.data.shape}")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.data.shape[0]:
            raise ValueError("labels.npy 必须是一维且样本数与 data.npy 一致")
        self.frame_mask = None
        mask_path = directory / "frame_mask.npy"
        if use_frame_mask and mask_path.is_file():
            self.frame_mask = np.load(mask_path, mmap_mode="r")
            if self.frame_mask.shape != (self.data.shape[0], self.data.shape[2]):
                raise ValueError("frame_mask.npy 形状必须为 N x T")
        self.limit = min(int(max_samples), len(self.labels)) if max_samples else len(self.labels)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        x = torch.from_numpy(np.array(self.data[index], dtype=np.float32, copy=True))
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        if self.frame_mask is None:
            mask = torch.ones(x.shape[1], dtype=torch.bool)
        else:
            mask = torch.from_numpy(np.asarray(self.frame_mask[index], dtype=np.bool_))
        return x, label, mask
