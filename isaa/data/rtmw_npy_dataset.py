"""Fast dataset for fixed-window RTMW features written as NumPy files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from isaa.data.temporal_augmentation import apply_coordinate_noise, apply_temporal_augmentation


class RTMWNpyDataset(Dataset):
    """Read ``data.npy``, ``labels.npy`` and optional ``frame_mask.npy``.

    ``data.npy`` is stored as ``N x C x T x V x M``. Three-channel files
    use the original features; five/eight-channel files also carry precomputed
    torso-relative x/y in channels 3:5. Eight-channel files add the corresponding
    cross-hand distance and unit direction at channels 5:8. NumPy memmap avoids copying the full
    dataset during startup.
    """

    def __init__(self, directory: str | Path, *, use_frame_mask: bool = False,
                 max_samples: int = 0, augment: bool = False,
                 augmentation_config: dict | None = None) -> None:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"预处理目录不存在: {directory}")
        data_path = directory / "data.npy"
        labels_path = directory / "labels.npy"
        if not data_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"预处理目录必须包含 data.npy 和 labels.npy: {directory}")
        self.data = np.load(data_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if self.data.ndim != 5 or self.data.shape[1] not in (3, 5, 8) or self.data.shape[3] not in (25, 32, 133):
            raise ValueError(f"data.npy 必须是 N x (3/5/8) x T x (25/32/133) x M，当前为 {self.data.shape}")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.data.shape[0]:
            raise ValueError("labels.npy 必须是一维且样本数与 data.npy 一致")
        self.frame_mask = None
        mask_path = directory / "frame_mask.npy"
        if use_frame_mask and mask_path.is_file():
            self.frame_mask = np.load(mask_path, mmap_mode="r")
            if self.frame_mask.shape != (self.data.shape[0], self.data.shape[2]):
                raise ValueError("frame_mask.npy 形状必须为 N x T")
        self.limit = min(int(max_samples), len(self.labels)) if max_samples else len(self.labels)
        self.augment = bool(augment)
        self.augmentation_config = dict(augmentation_config or {})

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        x = torch.from_numpy(np.array(self.data[index], dtype=np.float32, copy=True))
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        if self.frame_mask is None:
            mask = torch.ones(x.shape[1], dtype=torch.bool)
        else:
            mask = torch.from_numpy(np.asarray(self.frame_mask[index], dtype=np.bool_))
        if self.augment:
            if bool(self.augmentation_config.get("temporal_enabled", True)):
                x, mask = apply_temporal_augmentation(
                    x, mask,
                    crop_min_ratio=float(self.augmentation_config.get("crop_min_ratio", 0.875)),
                    max_shift=int(self.augmentation_config.get("max_shift", 4)),
                    jitter_probability=float(self.augmentation_config.get("jitter_probability", 0.2)),
                )
            if x.shape[0] in (5, 8):
                relative = x[3:5].clone()
                original_xy = x[:2].clone()
                raw, mask = apply_coordinate_noise(
                    x[:3], mask,
                    std=float(self.augmentation_config.get("coordinate_jitter_std", 0.0)),
                )
                relative += raw[:2] - original_xy
                if x.shape[0] == 8:
                    # Recompute distance and direction after temporal
                    # resampling/noise so channels 5:8 stay consistent.
                    x = torch.cat((raw, relative, x[5:8]), dim=0)
                    torso_ids = torch.tensor((5, 6, 11, 12), dtype=torch.long)
                    torso_xy = relative.index_select(2, torso_ids)  # [2,T,4,M]
                    torso_valid = raw[2].index_select(1, torso_ids) > 0  # [T,4,M]
                    torso_weight = torso_valid.to(torso_xy.dtype).unsqueeze(0)
                    shoulder_count = torso_weight[:, :, :2].sum(2).clamp_min(1)
                    hip_count = torso_weight[:, :, 2:].sum(2).clamp_min(1)
                    shoulder = (torso_xy[:, :, :2] * torso_weight[:, :, :2]).sum(2) / shoulder_count
                    hip = (torso_xy[:, :, 2:] * torso_weight[:, :, 2:]).sum(2) / hip_count
                    torso_scale = torch.linalg.vector_norm(shoulder - hip, dim=0).clamp_min(1e-3)
                    left = relative[:, :, 91:112]
                    right = relative[:, :, 112:133]
                    pair_valid = (raw[2, :, 91:112] > 0) & (raw[2, :, 112:133] > 0)
                    distance = torch.linalg.vector_norm(left - right, dim=0) / torso_scale[:, None, :]
                    distance = distance * pair_valid.to(distance.dtype)
                    vector = relative[:, :, 112:133] - relative[:, :, 91:112]
                    direction = vector / torch.linalg.vector_norm(
                        vector, dim=0, keepdim=True
                    ).clamp_min(1e-6)
                    direction = direction * pair_valid.to(direction.dtype).unsqueeze(0)
                    x[5].zero_()
                    x[6:8].zero_()
                    x[5, :, 91:112] = distance
                    x[5, :, 112:133] = distance
                    x[6:8, :, 91:112] = direction
                    x[6:8, :, 112:133] = -direction
                else:
                    x = torch.cat((raw, relative), dim=0)
            else:
                x, mask = apply_coordinate_noise(
                    x, mask,
                    std=float(self.augmentation_config.get("coordinate_jitter_std", 0.0)),
                )
        return x, label, mask
