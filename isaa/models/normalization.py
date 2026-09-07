"""HAPM 模型共享的归一化工具。

这里放与具体卷积核无关、但多个模块都会用到的 masked BatchNorm 逻辑。
训练骨架序列时同一个 batch 内常有 padding 帧，如果直接进入 BatchNorm，
补齐帧会污染均值和方差。这里的实现会在传入 `valid_frame_mask` 时只统计有效帧。
"""

from __future__ import annotations

import torch
from torch import nn


MASKED_STATS_MAX_ELEMENTS = 8_000_000


def _mask_has_empty_sample(mask: torch.Tensor) -> bool:
    """检查是否存在完全没有有效帧的样本。

    只在 CPU mask 上执行布尔判断，避免 CUDA 同步带来的训练性能损耗。
    """
    if mask.device.type != "cpu":
        return False
    return not bool(mask.any(dim=1).all())


def _mask_is_all_true(mask: torch.Tensor) -> bool:
    """判断 mask 是否全为 True。

    只在 CPU mask 上走普通 BatchNorm 快路径；GPU 上不主动同步检查。
    """
    if mask.device.type != "cpu":
        return False
    return bool(mask.all())


def _running_var_from_count(var: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """把当前 batch 方差转换成可写入 running_var 的无偏估计。"""
    correction = torch.where(count > 1.0, count / (count - 1.0).clamp_min(1.0), torch.ones_like(count))
    return var.detach() * correction


def _masked_stats_chunk_size(batch_size: int, elements_per_sample: int) -> int:
    """按临时张量上限估计 masked 统计的 batch 分块大小。"""
    if batch_size <= 1 or elements_per_sample <= 0:
        return max(1, batch_size)
    return max(1, min(batch_size, MASKED_STATS_MAX_ELEMENTS // elements_per_sample))


def _sum_chunks(parts: list[torch.Tensor]) -> torch.Tensor:
    """合并多个 batch 分块上的统计结果。"""
    if len(parts) == 1:
        return parts[0]
    return torch.stack(parts, dim=0).sum(dim=0)


def _masked_moments_2d(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 B x C x T x N 张量的按通道 masked 均值和方差。"""
    b, c, t, n = x.shape
    mask = mask.to(dtype=x.dtype).view(b, 1, t, 1)
    count = (mask.sum() * n).clamp_min(1.0)
    chunk_size = _masked_stats_chunk_size(b, c * t * n)
    sums: list[torch.Tensor] = []
    square_sums: list[torch.Tensor] = []
    for start in range(0, b, chunk_size):
        end = min(start + chunk_size, b)
        chunk_x = x[start:end]
        chunk_mask = mask[start:end]
        sums.append((chunk_x * chunk_mask).sum(dim=(0, 2, 3)))
        square_sums.append((chunk_x.square() * chunk_mask).sum(dim=(0, 2, 3)))
    mean = _sum_chunks(sums) / count
    second_moment = _sum_chunks(square_sums) / count
    var = (second_moment - mean.square()).clamp_min(0.0)
    return mean, var, count


def _masked_moments_1d(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 B x C x T 张量的按通道 masked 均值和方差。"""
    b, c, t = x.shape
    mask = mask.to(dtype=x.dtype).view(b, 1, t)
    count = mask.sum().clamp_min(1.0)
    chunk_size = _masked_stats_chunk_size(b, c * t)
    sums: list[torch.Tensor] = []
    square_sums: list[torch.Tensor] = []
    for start in range(0, b, chunk_size):
        end = min(start + chunk_size, b)
        chunk_x = x[start:end]
        chunk_mask = mask[start:end]
        sums.append((chunk_x * chunk_mask).sum(dim=(0, 2)))
        square_sums.append((chunk_x.square() * chunk_mask).sum(dim=(0, 2)))
    mean = _sum_chunks(sums) / count
    second_moment = _sum_chunks(square_sums) / count
    var = (second_moment - mean.square()).clamp_min(0.0)
    return mean, var, count


class MaskedBatchNorm2d(nn.Module):
    """支持时间 padding mask 的二维 BatchNorm。

    输入:
        B x C x T x N。

    行为:
        没有 mask 或 mask 全 True 时使用普通 BatchNorm2d；
        存在 padding 时只用有效帧统计每个通道的 mean/var。
        training 模式会同步更新 running mean/var，eval 模式在传入 mask 时也使用 masked stats。
    """

    def __init__(self, channels: int) -> None:
        """初始化内部 BatchNorm2d。"""
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """执行带可选时间 mask 的归一化。"""
        if valid_frame_mask is None:
            return self.bn(x)

        b, c, t, n = x.shape
        mask = valid_frame_mask.to(device=x.device).bool()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(b, -1)
        if mask.shape != (b, t):
            raise ValueError(f"valid_frame_mask 形状必须是 B x T，当前为 {tuple(mask.shape)}，输入为 {tuple(x.shape)}")
        if _mask_has_empty_sample(mask):
            raise ValueError("valid_frame_mask 每个样本至少需要 1 帧有效")
        if _mask_is_all_true(mask):
            return self.bn(x)
        mean, var, count = _masked_moments_2d(x, mask)
        if self.training and self.bn.track_running_stats:
            with torch.no_grad():
                self.bn.num_batches_tracked.add_(1)
                if self.bn.momentum is None:
                    momentum = 1.0 / float(self.bn.num_batches_tracked.item())
                else:
                    momentum = float(self.bn.momentum)
                running_var = _running_var_from_count(var, count)
                self.bn.running_mean.mul_(1.0 - momentum).add_(mean.detach() * momentum)
                self.bn.running_var.mul_(1.0 - momentum).add_(running_var * momentum)
        out = (x - mean.view(1, c, 1, 1)) * torch.rsqrt(var.view(1, c, 1, 1) + self.bn.eps)
        if self.bn.affine:
            out.mul_(self.bn.weight.view(1, c, 1, 1))
            out.add_(self.bn.bias.view(1, c, 1, 1))
        return out


class SkeletonInputNorm(nn.Module):
    """骨架输入批归一化。

    形状约定:
        输入为 B x C x T x N。

    做法:
        先把关节维 N 和通道维 C 合并成 N*C 个 BatchNorm1d 通道，
        贴近 ST-GCN 系列对骨架输入的处理方式。传入 mask 时只用有效帧统计。
    """

    def __init__(self, channels: int, num_joints: int) -> None:
        """初始化输入归一化层。"""
        super().__init__()
        self.channels = channels
        self.num_joints = num_joints
        self.bn = nn.BatchNorm1d(channels * num_joints)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """执行输入归一化，并在结束时恢复 B x C x T x N 形状。"""
        input_shape = tuple(x.shape)
        b, c, t, n = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, n * c, t)
        if valid_frame_mask is None:
            x = self.bn(x)
        else:
            mask = valid_frame_mask.to(device=x.device).bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(b, -1)
            if mask.shape != (b, t):
                raise ValueError(f"valid_frame_mask 形状必须是 B x T，当前为 {tuple(mask.shape)}，输入为 {input_shape}")
            if _mask_has_empty_sample(mask):
                raise ValueError("valid_frame_mask 每个样本至少需要 1 帧有效")
            if _mask_is_all_true(mask):
                x = self.bn(x)
                return x.view(b, n, c, t).permute(0, 2, 3, 1).contiguous()
            mean, var, count = _masked_moments_1d(x, mask)
            if self.training and self.bn.track_running_stats:
                with torch.no_grad():
                    self.bn.num_batches_tracked.add_(1)
                    if self.bn.momentum is None:
                        momentum = 1.0 / float(self.bn.num_batches_tracked.item())
                    else:
                        momentum = float(self.bn.momentum)
                    running_var = _running_var_from_count(var, count)
                    self.bn.running_mean.mul_(1.0 - momentum).add_(mean.detach() * momentum)
                    self.bn.running_var.mul_(1.0 - momentum).add_(running_var * momentum)
            x = (x - mean.view(1, -1, 1)) * torch.rsqrt(var.view(1, -1, 1) + self.bn.eps)
            if self.bn.affine:
                x.mul_(self.bn.weight.view(1, -1, 1))
                x.add_(self.bn.bias.view(1, -1, 1))
        return x.view(b, n, c, t).permute(0, 2, 3, 1).contiguous()
