"""Streaming dataset for per-video RTMW skeletons stored in a ZIP archive.

The expected archive contains one ``*_rgb.npz`` file per NTU RGB+D 120
video.  Each sample stores RTMW / COCO-WholeBody keypoints as
``T x M x 133 x 2`` plus a separate ``T x M x 133`` score array.  Samples
are read on demand so the multi-gigabyte archive never needs to be unpacked
or loaded into memory in full.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from isaa.data.transforms import apply_skeleton_augmentation, build_skeleton_feature_channels
from isaa.data.skeleton_dataset import (
    _align_joint_count,
    _scale_range_from_config,
    _temporal_crop_or_pad_sample_with_mask,
)


NTU120_XSUB_TRAIN_SUBJECTS = frozenset(
    {
        1,
        2,
        4,
        5,
        8,
        9,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        25,
        27,
        28,
        31,
        34,
        35,
        38,
        45,
        46,
        47,
        49,
        50,
        52,
        53,
        54,
        55,
        57,
        58,
        59,
        70,
        74,
        78,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        89,
        91,
        92,
        93,
        94,
        95,
        97,
        98,
        100,
        103,
    }
)
SUPPORTED_RTMW_SPLIT_PROTOCOLS = frozenset({"xsub120", "xset120"})
SUPPORTED_RTMW_SCORE_NORMALIZATIONS = frozenset({"auto", "clip", "sigmoid"})

_SAMPLE_NAME_PATTERN = re.compile(
    r"(?:^|/)S(?P<setup>\d{3})C\d{3}P(?P<subject>\d{3})R\d{3}A(?P<action>\d{3})_rgb\.npz$",
    flags=re.IGNORECASE,
)


class RTMWSampleInfo(NamedTuple):
    """Minimal archive metadata needed for splitting and labeling."""

    member_name: str
    setup: int
    subject: int
    action: int


def _canonical_split_protocol(value: object) -> str:
    protocol = str(value).strip().lower()
    aliases = {"xsub": "xsub120", "xset": "xset120"}
    protocol = aliases.get(protocol, protocol)
    if protocol not in SUPPORTED_RTMW_SPLIT_PROTOCOLS:
        raise ValueError(
            "RTMW ZIP split_protocol 只支持 xsub120/xset120 "
            f"（也可简写 xsub/xset），当前为 {value!r}"
        )
    return protocol


@lru_cache(maxsize=4)
def _scan_archive_cached(
    archive_path: str,
    archive_size: int,
    archive_mtime_ns: int,
) -> tuple[RTMWSampleInfo, ...]:
    """Read the ZIP central directory once and parse NTU sample identifiers."""
    del archive_size, archive_mtime_ns  # These arguments invalidate the cache after replacement.
    samples: list[RTMWSampleInfo] = []
    unmatched_npz: list[str] = []
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                normalized_name = info.filename.replace("\\", "/")
                if not normalized_name.lower().endswith(".npz"):
                    continue
                match = _SAMPLE_NAME_PATTERN.search(normalized_name)
                if match is None:
                    unmatched_npz.append(info.filename)
                    continue
                samples.append(
                    RTMWSampleInfo(
                        member_name=info.filename,
                        setup=int(match.group("setup")),
                        subject=int(match.group("subject")),
                        action=int(match.group("action")),
                    )
                )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"RTMW 数据文件不是有效 ZIP: {archive_path}") from exc

    if unmatched_npz:
        examples = ", ".join(unmatched_npz[:3])
        raise ValueError(
            "RTMW ZIP 中存在无法按 NTU SxxxCxxxPxxxRxxxAxxx 命名解析的 npz："
            f"{examples}"
        )
    if not samples:
        raise ValueError(f"RTMW ZIP 中没有找到 *_rgb.npz 样本: {archive_path}")
    return tuple(samples)


def scan_rtmw_archive(archive_path: str | Path) -> tuple[RTMWSampleInfo, ...]:
    """Return parsed sample metadata, cached by path/size/mtime."""
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(f"RTMW ZIP 不存在或不是文件: {path}")
    stat = path.stat()
    return _scan_archive_cached(str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))


def _belongs_to_train_split(sample: RTMWSampleInfo, protocol: str) -> bool:
    if protocol == "xsub120":
        return sample.subject in NTU120_XSUB_TRAIN_SUBJECTS
    return sample.setup % 2 == 0


def _normalize_scores(scores: np.ndarray, normalization: str) -> np.ndarray:
    """Convert RTMW scores to the model's [0, 1] confidence channel."""
    normalization = str(normalization).strip().lower()
    if normalization not in SUPPORTED_RTMW_SCORE_NORMALIZATIONS:
        raise ValueError(
            "score_normalization 必须是 auto/clip/sigmoid，"
            f"当前为 {normalization!r}"
        )
    scores = np.nan_to_num(scores, nan=0.0, posinf=30.0, neginf=-30.0).astype(np.float32, copy=False)
    use_sigmoid = normalization == "sigmoid" or (
        normalization == "auto" and scores.size > 0 and float(scores.max()) > 1.0
    )
    if use_sigmoid:
        clipped = np.clip(scores, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)
    return np.clip(scores, 0.0, 1.0).astype(np.float32, copy=False)


class RTMWZipDataset(Dataset):
    """Stream RTMW-133 samples from an NTU120 archive.

    The returned skeleton has shape ``5 x T x 133 x M`` with channels
    ``x, y, dx, dy, score``.  Labels are converted from NTU's one-based
    action ids to zero-based class indices.
    """

    def __init__(
        self,
        archive_path: str | Path,
        *,
        split: str,
        split_protocol: str,
        window_size: int,
        num_joints: int,
        num_classes: int,
        layout: str,
        max_persons: int = 2,
        score_normalization: str = "auto",
        augment: bool = False,
        augmentation_config: dict[str, Any] | None = None,
        deterministic_temporal_crop: str = "center",
        max_samples: int = 0,
    ) -> None:
        path = Path(archive_path)
        if not path.is_file():
            raise FileNotFoundError(f"RTMW ZIP 不存在或不是文件: {path}")
        split = str(split).strip().lower()
        if split not in {"train", "val"}:
            raise ValueError(f"RTMW ZIP split 只支持 train/val，当前为 {split!r}")
        protocol = _canonical_split_protocol(split_protocol)
        window_size = int(window_size)
        num_joints = int(num_joints)
        num_classes = int(num_classes)
        max_persons = int(max_persons)
        max_samples = int(max_samples)
        if min(window_size, num_joints, num_classes, max_persons) <= 0:
            raise ValueError("window_size/num_joints/num_classes/max_persons 必须是正整数")
        if max_samples < 0:
            raise ValueError(f"max_samples 不能为负数，当前为 {max_samples}")
        deterministic_temporal_crop = str(deterministic_temporal_crop).strip().lower()
        if deterministic_temporal_crop not in {"start", "center", "end"}:
            raise ValueError("deterministic_temporal_crop 只支持 start/center/end")

        all_samples = scan_rtmw_archive(path)
        want_train = split == "train"
        selected = tuple(
            sample for sample in all_samples if _belongs_to_train_split(sample, protocol) == want_train
        )
        if max_samples > 0:
            selected = selected[:max_samples]
        if not selected:
            raise ValueError(f"RTMW ZIP 按 {protocol}/{split} 划分后没有样本")
        label_min = min(sample.action for sample in selected) - 1
        label_max = max(sample.action for sample in selected) - 1
        if label_min < 0 or label_max >= num_classes:
            raise ValueError(
                f"RTMW 标签范围 [{label_min}, {label_max}] 与 num_classes={num_classes} 不匹配"
            )

        self.archive_path = path.resolve()
        self.members = tuple(sample.member_name for sample in selected)
        self.labels = np.fromiter((sample.action - 1 for sample in selected), dtype=np.int64)
        self.split = split
        self.split_protocol = protocol
        self.window_size = window_size
        self.num_joints = num_joints
        self.layout = str(layout)
        self.max_persons = max_persons
        self.score_normalization = str(score_normalization).strip().lower()
        if self.score_normalization not in SUPPORTED_RTMW_SCORE_NORMALIZATIONS:
            raise ValueError(
                "score_normalization 必须是 auto/clip/sigmoid，"
                f"当前为 {score_normalization!r}"
            )
        self.augment = bool(augment)
        self.augmentation_config = augmentation_config or {}
        self.deterministic_temporal_crop = deterministic_temporal_crop
        self._archive: zipfile.ZipFile | None = None
        self._archive_pid: int | None = None

    def __len__(self) -> int:
        return len(self.members)

    def __getstate__(self) -> dict[str, Any]:
        """Do not pickle an open file handle into DataLoader workers."""
        state = self.__dict__.copy()
        state["_archive"] = None
        state["_archive_pid"] = None
        return state

    def __del__(self) -> None:
        archive = getattr(self, "_archive", None)
        if archive is not None:
            try:
                archive.close()
            except Exception:
                pass

    def _get_archive(self) -> zipfile.ZipFile:
        current_pid = os.getpid()
        if self._archive is not None and self._archive_pid != current_pid:
            self._archive.close()
            self._archive = None
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path, mode="r")
            self._archive_pid = current_pid
        return self._archive

    def _read_raw_sample(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        member_name = self.members[index]
        try:
            payload = self._get_archive().read(member_name)
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                missing = [key for key in ("keypoints", "scores") if key not in data.files]
                if missing:
                    raise ValueError(f"缺少字段 {missing}")
                keypoints = np.asarray(data["keypoints"], dtype=np.float32)
                scores = np.asarray(data["scores"], dtype=np.float32)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"读取 RTMW 样本失败: {member_name}: {exc}") from exc

        if keypoints.ndim != 4 or keypoints.shape[-1] != 2:
            raise ValueError(
                f"RTMW keypoints 必须是 T x M x N x 2，{member_name} 当前为 {keypoints.shape}"
            )
        if scores.shape != keypoints.shape[:-1]:
            raise ValueError(
                f"RTMW scores 必须匹配 keypoints 的 T x M x N，{member_name} 当前为 {scores.shape}"
            )
        if keypoints.shape[0] <= 0:
            raise ValueError(f"RTMW 样本至少需要 1 帧: {member_name}")
        if keypoints.shape[2] != self.num_joints:
            raise ValueError(
                f"RTMW 布局要求 {self.num_joints} 个关节，{member_name} 当前为 {keypoints.shape[2]}"
            )

        coordinate_valid = np.isfinite(keypoints).all(axis=-1)
        keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)
        scores = _normalize_scores(scores, self.score_normalization)
        scores = np.where(coordinate_valid, scores, 0.0).astype(np.float32, copy=False)
        keypoints = np.where(coordinate_valid[..., None], keypoints, 0.0).astype(np.float32, copy=False)

        # Source: T x M x N x 2 and T x M x N.
        # Model raw input: 3 x T x N x M (x, y, score).
        coords = np.transpose(keypoints, (3, 0, 2, 1))
        confidence = np.transpose(scores, (0, 2, 1))[None, ...]
        raw = torch.from_numpy(np.concatenate((coords, confidence), axis=0))
        valid_frame_mask = torch.from_numpy(coordinate_valid.any(axis=(1, 2)))

        persons = raw.size(3)
        if persons > self.max_persons:
            raw = raw[:, :, :, : self.max_persons]
        elif persons < self.max_persons:
            pad = raw.new_zeros(raw.size(0), raw.size(1), raw.size(2), self.max_persons - persons)
            raw = torch.cat((raw, pad), dim=3)
        return raw.contiguous(), valid_frame_mask.bool().contiguous()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw, file_mask = self._read_raw_sample(index)
        raw = _align_joint_count(raw, expected_num_joints=self.num_joints)
        random_crop = self.augment and bool(self.augmentation_config.get("random_temporal_crop", False))
        raw, valid_frame_mask = _temporal_crop_or_pad_sample_with_mask(
            raw,
            file_mask,
            self.window_size,
            random_start=random_crop,
            deterministic_crop=self.deterministic_temporal_crop,
        )
        if self.augment:
            raw = apply_skeleton_augmentation(
                raw,
                coordinate_dims=2,
                rotation_degrees=float(self.augmentation_config.get("rotation_degrees", 0.0)),
                scale_range=_scale_range_from_config(
                    self.augmentation_config.get("scale_range", (1.0, 1.0))
                ),
                jitter_std=float(self.augmentation_config.get("coordinate_jitter_std", 0.0)),
            )

        point_valid = raw[2:3] > 0
        raw[:2] = raw[:2] * point_valid
        features = build_skeleton_feature_channels(
            raw,
            layout=self.layout,
            coordinate_dims=2,
            score_index=2,
        )
        # Invalid RTMW points are NaN in the archive.  Keep their coordinates
        # and the motion touching them at zero after torso normalization.
        features[:2] = features[:2] * point_valid
        motion_valid = point_valid.clone()
        motion_valid[:, 1:] = point_valid[:, 1:] & point_valid[:, :-1]
        features[2:4] = features[2:4] * motion_valid
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return features.contiguous(), label, valid_frame_mask.contiguous()
