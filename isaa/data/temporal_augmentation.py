"""Conservative temporal augmentation for fixed-length skeleton windows."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _resample(x: torch.Tensor, mask: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample C x T x V x M at fractional temporal indices."""
    length = x.shape[1]
    indices = indices.to(device=x.device, dtype=x.dtype).clamp(0, length - 1)
    left = indices.floor().long()
    right = (left + 1).clamp_max(length - 1)
    weight = (indices - left.to(indices.dtype)).view(1, -1, 1, 1)
    sampled = x[:, left] * (1.0 - weight) + x[:, right] * weight
    mask_sampled = mask[left] | mask[right]
    return sampled, mask_sampled


def apply_temporal_augmentation(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    crop_min_ratio: float = 0.875,
    max_shift: int = 4,
    jitter_probability: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one mild temporal transform to a fixed-length skeleton window.

    Coordinates, scores, nodes, and persons remain aligned.  The transform
    changes only the temporal sampling and is intended for training samples.
    """
    if x.ndim != 4 or mask.ndim != 1 or x.shape[1] != mask.shape[0]:
        raise ValueError("Expected x=[C,T,V,M] and mask=[T]")
    target_t = int(x.shape[1])
    draw = float(torch.rand((), device=x.device))

    if draw < 0.5:
        crop_len = int(torch.randint(
            max(2, math.ceil(target_t * crop_min_ratio)), target_t + 1, (1,),
        ).item())
        start = int(torch.randint(0, target_t - crop_len + 1, (1,)).item())
        crop = x[:, start:start + crop_len]
        crop_mask = mask[start:start + crop_len]
        resized = F.interpolate(
            crop.reshape(1, crop.shape[0] * crop.shape[2] * crop.shape[3], crop_len),
            size=target_t, mode="linear", align_corners=False,
        ).reshape_as(x)
        resized_mask = F.interpolate(
            crop_mask.float().reshape(1, 1, crop_len), size=target_t, mode="nearest",
        )[0, 0].bool()
        return resized, resized_mask

    if draw < 0.8:
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        indices = torch.arange(target_t, device=x.device) - shift
        indices = indices.clamp(0, target_t - 1)
        return x.index_select(1, indices), mask.index_select(0, indices)

    if draw < 0.8 + jitter_probability:
        base = torch.arange(target_t, device=x.device, dtype=x.dtype)
        offsets = torch.randint(-1, 2, (target_t,), device=x.device).to(x.dtype)
        indices = (base + offsets).clamp(0, target_t - 1)
        indices = torch.cummax(indices, dim=0).values
        return _resample(x, mask, indices)

    return x, mask


__all__ = ["apply_temporal_augmentation"]
