"""Official SkeleT preprocessing on RTMW ZIP or PYSKL annotation pickle.

The ZIP adapter shares only raw NPZ reading with the existing loader; it
does not reuse ISAA's crop/pad, torso-relative coordinates or augmentation.
PreNormalize2D, GenSkeFeat and UniformSample follow upstream fb2b2b23.
"""
from pathlib import Path
import pickle
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .rtmw_zip_dataset import (
    RTMWZipDataset, scan_rtmw_archive,
    NTU60_XSUB_TRAIN_SUBJECTS, NTU120_XSUB_TRAIN_SUBJECTS,
)
from isaa.models.skelet.model import EXPRESSIVE_INDICES

SPLITS = ("xsub60", "xview60", "xsub120", "xset120")
STREAMS = ("j", "b", "jm", "bm")


def bone_parents():
    # Upstream JointToBone('expressive-keypoints'), corrected reachable branch.
    parents = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14,
               19, 19, 15, 22, 22, 16]
    for root, wrist in ((23, 9), (44, 10)):
        parents.append(wrist)
        for finger in range(5):
            start = root + 1 + 4 * finger
            parents.extend((root, start, start + 1, start + 2))
    return np.asarray(parents)


def preprocess_keypoints(raw, stream="j", image_shape=(1080, 1920), threshold=0.01):
    """C,T,V,M -> normalized M,T,65,C, before temporal sampling.

    Scores in b/jm/bm are differences, matching upstream's expressive format;
    they are not silently replaced by mean confidence.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 4 or raw.shape[0] != 3 or raw.shape[2] not in (65, 133):
        raise ValueError("Expected raw 3,T,65-or-133,M")
    if stream not in STREAMS:
        raise ValueError(f"Unknown skeleton stream: {stream}")
    if raw.shape[2] == 133:
        raw = raw[:, :, EXPRESSIVE_INDICES, :]
    x = raw.transpose(3, 1, 2, 0).copy()
    h, w = image_shape
    if h <= 0 or w <= 0 or not np.isfinite(x).all():
        raise ValueError("Image dimensions must be positive and keypoints finite")
    x[..., 0] = (x[..., 0] - w / 2) / (w / 2)
    x[..., 1] = (x[..., 1] - h / 2) / (h / 2)
    x[..., :2][x[..., 2] <= threshold] = 0
    if stream in ("b", "bm"):
        x = x - x[:, :, bone_parents(), :]
    if stream in ("jm", "bm"):
        motion = np.zeros_like(x)
        motion[:, :-1] = np.diff(x, axis=1)
        x = motion
    return x


def uniform_indices(num_frames, clip_len, *, training, num_clips=1, seed=289800):
    """PYSKL UniformSample with p_interval=1, without resetting global test RNG."""
    if min(num_frames, clip_len, num_clips) < 1:
        raise ValueError("Frame count, clip length and clip count must be positive")
    rng = np.random if training else np.random.RandomState(seed)
    clips = []
    for i in range(num_clips):
        # Preserve the upstream RNG draws for the degenerate full-video crop.
        rng.rand()
        rng.randint(1)
        if num_frames < clip_len:
            start = rng.randint(num_frames) if training else (
                i if num_frames < num_clips else i * num_frames // num_clips)
            inds = np.arange(start, start + clip_len)
        elif num_frames < 2 * clip_len:
            selected = rng.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[selected] = 1
            inds = np.arange(clip_len) + np.cumsum(offset)[:-1]
        else:
            boundaries = np.arange(clip_len + 1) * num_frames // clip_len
            inds = boundaries[:-1] + rng.randint(np.diff(boundaries))
        clips.append(inds % num_frames)
    return np.stack(clips)


def format_clips(raw, *, stream, image_shape, clip_len, training, num_clips):
    x = preprocess_keypoints(raw, stream, image_shape)
    inds = uniform_indices(x.shape[1], clip_len, training=training, num_clips=num_clips)
    # K,C,T,V,M; single training/validation clip returns C,T,V,M.
    clips = np.stack([x[:, index].transpose(3, 1, 2, 0) for index in inds])
    return torch.from_numpy(np.ascontiguousarray(clips[0] if num_clips == 1 else clips))


class SkeleTRTMWDataset(RTMWZipDataset):
    def __init__(self, archive_path, *, protocol="xsub60", training=True,
                 stream="j", clip_len=100, num_person=2, num_clips=1,
                 image_shape=(1080, 1920), max_samples=0):
        if protocol not in SPLITS or stream not in STREAMS:
            raise ValueError("Invalid SkeleT split or stream")
        self.archive_path = Path(archive_path).resolve()
        samples = scan_rtmw_archive(self.archive_path)
        selected = []
        for sample in samples:
            if protocol.endswith("60") and (sample.action > 60 or sample.setup > 17):
                continue
            if protocol == "xview60":
                camera = int(re.search(r"S\d{3}C(\d{3})P", sample.member_name).group(1))
                train = camera in (2, 3)
            elif protocol == "xset120":
                train = sample.setup % 2 == 0
            else:
                subjects = NTU60_XSUB_TRAIN_SUBJECTS if protocol == "xsub60" else NTU120_XSUB_TRAIN_SUBJECTS
                train = sample.subject in subjects
            if train == training:
                selected.append(sample)
        if max_samples:
            selected = selected[:max_samples]
        if not selected:
            raise ValueError(f"Empty SkeleT dataset: {protocol}, training={training}")
        self.members = tuple(s.member_name for s in selected)
        self.labels = np.asarray([s.action - 1 for s in selected], dtype=np.int64)
        self.num_joints, self.max_persons = 133, num_person
        self.score_normalization = "auto"
        self._archive, self._archive_pid = None, None
        self.training, self.stream = training, stream
        self.clip_len, self.num_clips, self.image_shape = clip_len, num_clips, image_shape

    def __getitem__(self, index):
        raw, _ = self._read_raw_sample(index)
        clips = format_clips(raw.numpy(), stream=self.stream, image_shape=self.image_shape,
                             clip_len=self.clip_len, training=self.training, num_clips=self.num_clips)
        return clips, torch.tensor(int(self.labels[index]))


class SkeleTPoseDataset(Dataset):
    """Read the official PYSKL {split, annotations} pickle when available."""
    def __init__(self, ann_file, *, protocol="xsub60", training=True, stream="j",
                 clip_len=100, num_person=2, num_clips=1,
                 image_shape=(1080, 1920), max_samples=0):
        with Path(ann_file).open("rb") as f:
            data = pickle.load(f)
        split_name = {"xsub60": "xsub", "xview60": "xview", "xsub120": "xsub", "xset120": "xset"}[protocol]
        split_name += "_train" if training else "_val"
        names = set(data["split"][split_name])
        self.annotations = [a for a in data["annotations"] if a["frame_dir"] in names]
        if max_samples:
            self.annotations = self.annotations[:max_samples]
        if not self.annotations:
            raise ValueError(f"Empty annotation split: {split_name}")
        self.training, self.stream, self.clip_len = training, stream, clip_len
        self.num_person, self.num_clips, self.image_shape = num_person, num_clips, image_shape

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        a = self.annotations[index]
        xy = np.asarray(a["keypoint"], dtype=np.float32)
        scores = np.asarray(a["keypoint_score"], dtype=np.float32)
        valid = np.isfinite(xy).all(axis=-1) & np.isfinite(scores)
        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.where(valid, scores, 0)
        x = np.concatenate((xy, scores[..., None]), axis=-1)
        # Pad after preprocessing, just as FormatGCNInput does upstream.
        raw = x.transpose(3, 1, 2, 0)
        clips = format_clips(raw, stream=self.stream, image_shape=a.get("img_shape", self.image_shape),
                             clip_len=self.clip_len, training=self.training, num_clips=self.num_clips)
        if clips.shape[-1] < self.num_person:
            clips = torch.cat((clips, clips.new_zeros(*clips.shape[:-1], self.num_person - clips.shape[-1])), -1)
        return clips[..., :self.num_person].contiguous(), torch.tensor(int(a["label"]))
