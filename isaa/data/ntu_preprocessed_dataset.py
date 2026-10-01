"""Loader for preprocessed official NTU60 skeleton splits.

The converter in HumanActionParticipationModeling writes ``train.npz`` and
``test.npz`` with ``x`` shaped ``B x T x 25 x M x C``.  This loader converts
that representation to the model's ``C x T x V x M`` layout and never reads
raw RTMW archives or performs a random crop at training time.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from isaa.data.temporal_augmentation import apply_temporal_augmentation


class NTUPreprocessedDataset(Dataset):
    """Read one fixed, already-preprocessed NTU60 split."""

    def __init__(self, path: str | Path, *, max_samples: int = 0, expected_nodes: int = 25,
                 window_size: int | None = None, augment: bool = False,
                 augmentation_config: dict | None = None) -> None:
        self.path = Path(path)
        frame_mask = None
        if self.path.is_dir():
            # Official CTR-GCN preprocessing layout:
            # train_data.npy/val_data.npy and train_label.pkl/val_label.pkl.
            raise ValueError("目录方式需传入 train_data.npy 或 val_data.npy 文件路径")
            if not data_path.is_file():
                raise FileNotFoundError(f"官方 NTU 预处理文件不存在: {data_path}")
            data = np.load(data_path, mmap_mode="r")
            with label_path.open("rb") as handle:
                payload = pickle.load(handle)
            labels = np.asarray(payload[1] if isinstance(payload, (tuple, list)) and len(payload) == 2 else payload)
        elif self.path.suffix == ".npy":
            if not self.path.is_file():
                raise FileNotFoundError(f"官方 NTU 预处理文件不存在: {self.path}")
            data = np.load(self.path, mmap_mode="r")
            label_path = self.path.with_name(self.path.name.replace("_data.npy", "_label.pkl"))
            with label_path.open("rb") as handle:
                payload = pickle.load(handle)
            labels = np.asarray(payload[1] if isinstance(payload, (tuple, list)) and len(payload) == 2 else payload)
        else:
            if not self.path.is_file():
                raise FileNotFoundError(f"官方 NTU 预处理文件不存在: {self.path}")
            with np.load(self.path, allow_pickle=False) as archive:
                if "x" not in archive or "y" not in archive:
                    raise ValueError(f"{self.path} 必须包含 x 和 y")
                data = np.asarray(archive["x"])
                labels = np.asarray(archive["y"])
                frame_mask = np.asarray(archive["valid_frame_mask"]) if "valid_frame_mask" in archive else None
        self.ctvm_layout = bool(data.ndim == 5 and data.shape[1] == 3 and data.shape[3] == expected_nodes)
        if self.ctvm_layout:
            pass
        elif data.ndim != 5 or data.shape[2] != expected_nodes:
            raise ValueError(f"官方 NTU 数据必须是 BxCxTx{expected_nodes}xM 或 BxTx{expected_nodes}xMxC，当前为 {data.shape}")
        if labels.ndim != 1 or labels.shape[0] != data.shape[0]:
            raise ValueError("官方 NTU y 必须是一维且与 x 样本数一致")
        if not self.ctvm_layout and data.shape[-1] < 3:
            raise ValueError("官方 NTU 预处理至少需要 3 个坐标通道")
        if frame_mask is not None and frame_mask.shape != (data.shape[0], data.shape[1]):
            raise ValueError("valid_frame_mask 必须是 B x T")
        self.data = data.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.frame_mask = frame_mask.astype(bool, copy=False) if frame_mask is not None else None
        self.limit = min(int(max_samples), len(self.labels)) if max_samples else len(self.labels)
        self.source_window_size = int(self.data.shape[2] if self.ctvm_layout else self.data.shape[1])
        self.window_size = int(window_size or self.source_window_size)
        if self.window_size < 1:
            raise ValueError("window_size 必须为正数")
        self.temporal_indices = (
            np.linspace(0, self.source_window_size - 1, self.window_size).round().astype(np.int64)
            if self.window_size != self.source_window_size else None
        )
        self.augment = bool(augment)
        self.augmentation_config = dict(augmentation_config or {})

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        # Official CTR-GCN uses the first three coordinate channels.  If the
        # converter retained tracking score as a fourth channel, it is ignored.
        if self.ctvm_layout:
            sample = np.asarray(self.data[index, :3], dtype=np.float32)  # C x T x V x M
            if self.temporal_indices is not None:
                sample = sample[:, self.temporal_indices]
        else:
            sample = np.asarray(self.data[index, ..., :3], dtype=np.float32)
            if self.temporal_indices is not None:
                sample = sample[self.temporal_indices]
            sample = np.transpose(sample, (3, 0, 1, 2))  # C x T x V x M
        x = torch.from_numpy(np.ascontiguousarray(sample))
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        if self.frame_mask is None or self.temporal_indices is not None:
            mask = torch.ones(x.shape[1], dtype=torch.bool)
        else:
            mask = torch.from_numpy(np.asarray(self.frame_mask[index], dtype=np.bool_))
        if self.augment:
            x, mask = apply_temporal_augmentation(
                x, mask,
                crop_min_ratio=float(self.augmentation_config.get("crop_min_ratio", 0.875)),
                max_shift=int(self.augmentation_config.get("max_shift", 4)),
                jitter_probability=float(self.augmentation_config.get("jitter_probability", 0.2)),
            )
        return x, label, mask
