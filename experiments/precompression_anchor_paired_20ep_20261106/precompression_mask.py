"""Deterministic precompression masking for four body anchor joints."""
from __future__ import annotations

import hashlib

import torch
from torch.utils.data import Dataset

from isaa.data.rtmw_zip_dataset import _confidence_weighted_temporal_compress
from isaa.data.skeleton_dataset import _temporal_crop_or_pad_sample_with_mask

ANCHORS = (5, 6, 11, 12)


def member_seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class PairedPrecompression(Dataset):
    def __init__(self, dataset, members, probability, seed):
        self.dataset, self.members = dataset, members
        self.probability, self.seed = probability, seed
        index = {name: i for i, name in enumerate(dataset.members)}
        if len(index) != len(dataset.members) or len(set(members)) != len(members):
            raise ValueError("Duplicate members")
        self.indices = [index[name] for name in members]

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        raw, frame = self.dataset._read_raw_sample(self.indices[i])
        rng = torch.Generator().manual_seed(member_seed(self.members[i], self.seed))
        anchor = raw[:, :, list(ANCHORS), :]
        hidden = (torch.rand(anchor[2].shape, generator=rng) < self.probability)
        hidden &= (anchor[2] > 0) & frame[:, None, None]
        corrupt = raw.clone()
        corrupt[:, :, list(ANCHORS), :] = anchor.masked_fill(hidden[None], 0)
        outputs = []
        for value in (raw, corrupt):
            value, mask = _confidence_weighted_temporal_compress(value, frame, self.dataset.window_size)
            value, mask = _temporal_crop_or_pad_sample_with_mask(
                value, mask, self.dataset.window_size, random_start=False,
                deterministic_crop=self.dataset.deterministic_temporal_crop)
            value[:2] *= value[2:3] > 0
            value = value.masked_fill(~mask[None, :, None, None], 0)
            outputs.append(value.contiguous())
        return outputs[0], outputs[1], int(self.dataset.labels[self.indices[i]]), mask, int(hidden.sum()), raw.shape[1]
