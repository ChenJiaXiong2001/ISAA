"""Native PyTorch replacements for the two MMCV factories used upstream."""
from torch import nn


def build_norm_layer(config, channels):
    config = dict(config)
    kind = config.pop("type")
    requires_grad = config.pop("requires_grad", True)
    if kind not in {"BN", "BN2d"}:
        raise ValueError(f"Unsupported SkeleT normalization: {kind}")
    layer = nn.BatchNorm2d(channels, **config)
    for parameter in layer.parameters():
        parameter.requires_grad_(requires_grad)
    return "bn", layer


def build_activation_layer(config):
    config = dict(config)
    kind = config.pop("type")
    if kind != "ReLU":
        raise ValueError(f"Unsupported SkeleT activation: {kind}")
    return nn.ReLU(**config)
