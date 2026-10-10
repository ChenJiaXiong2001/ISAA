"""Mask adapters and validation shared by reusable GCN backbones."""
import torch
from torch.nn import functional as F


def validate_stack(channels, strides, temporal_branches=1):
    if len(channels) != len(strides) or not channels:
        raise ValueError("channels and strides must have equal non-zero lengths")
    if any(type(c) is not int or c < 1 or c % temporal_branches for c in channels):
        raise ValueError(f"channels must be positive multiples of {temporal_branches}")
    if any(type(s) is not int or s not in (1, 2) for s in strides):
        raise ValueError("strides must contain only 1 or 2")


def prepare_masked_input(x, mask, channels, nodes):
    if x.ndim != 4 or x.shape[1] != channels or x.shape[3] != nodes:
        raise ValueError(f"Expected B x {channels} x T x {nodes}")
    if mask.shape != (x.shape[0], 1, x.shape[2], nodes):
        raise ValueError("mask must have shape B x 1 x T x V")
    mask = mask.bool() & torch.isfinite(x).all(1, keepdim=True)
    return x.masked_fill(~mask, 0), mask


def downsample_mask(mask, stride):
    if stride == 1:
        return mask
    return F.max_pool2d(mask.float(), (stride, 1), (stride, 1), ceil_mode=True).bool()
