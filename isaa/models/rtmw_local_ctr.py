"""RTMW experiment: fixed local graphs and CTR coordination on 32 real joints."""

from __future__ import annotations

import torch
from torch import nn

from isaa.graph.adjacency import build_joint_spatial_partitions, normalize_adjacency_partitions
from isaa.graph.regions import build_region_partition
from isaa.models.graph_convs.ctr_channel import ChannelWiseTopologyGraphConv
from isaa.models.normalization import _masked_stats_chunk_size


class PointBatchNorm(nn.Module):
    """Exclude missing nodes/frames/people; use running statistics in evaluation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.bn(x).masked_fill(~mask, 0)
        stats = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        weight = mask.to(stats.dtype)
        count = weight.sum()
        chunk_size = _masked_stats_chunk_size(stats.size(0), stats[0].numel())
        sums, squares = [], []
        for start in range(0, stats.size(0), chunk_size):
            chunk = stats[start:start + chunk_size].masked_fill(~mask[start:start + chunk_size], 0)
            sums.append(chunk.sum(dim=(0, 2, 3)))
            squares.append(chunk.square().sum(dim=(0, 2, 3)))
        mean = torch.stack(sums).sum(0) / count.clamp_min(1)
        var = (torch.stack(squares).sum(0) / count.clamp_min(1) - mean.square()).clamp_min(0)
        with torch.no_grad():
            # Empty branches leave running statistics untouched, including on CUDA.
            update = (count > 0).to(mean.dtype) * self.bn.momentum
            self.bn.num_batches_tracked.add_((count > 0).long())
            unbiased = var * count / (count - 1).clamp_min(1)
            self.bn.running_mean.lerp_(mean.detach(), update)
            self.bn.running_var.lerp_(unbiased.detach(), update)
        out = (stats - mean[None, :, None, None]) * torch.rsqrt(var[None, :, None, None] + self.bn.eps)
        out = out * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]
        return out.to(x.dtype).masked_fill(~mask, 0)


class FixedSkeletonConv(nn.Module):
    """Self/inward/outward projections with frozen RTMW connectivity."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency.clone())
        self.projection = nn.Conv2d(in_channels, 3 * out_channels, 1, bias=False)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, n = x.shape
        projected = self.projection(x).reshape(b, 3, self.out_channels, t, n)
        return torch.einsum("bpctv,puv->bctu", projected, self.adjacency)


class MainNodeCTR(ChannelWiseTopologyGraphConv):
    """CTR-GCN spatial unit with unrestricted, channel-wise dynamic topology."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__(in_channels, out_channels, adjacency, diagonal_fast_path=False)
        # Parent buffers use source/target order. This implementation uses target/source.
        self.base_topology.copy_(adjacency)
        self.topology_mask.fill_(1)
        self.static_topology = nn.Parameter(adjacency.clone())
        self.proj_norm = PointBatchNorm(out_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(x.dtype)
        count = weight.sum(dim=2).clamp_min(1)
        query = (self.theta(x) * weight).sum(dim=2) / count
        key = (self.phi(x) * weight).sum(dim=2) / count
        relation = torch.tanh(query.unsqueeze(-1) - key.unsqueeze(-2))
        topology = self.static_topology[None, None] + self.topology_alpha * self.relation_proj(relation)
        feature = self.feature_proj(x).masked_fill(~mask, 0)
        out = torch.einsum("bcuv,bctv->bctu", topology, feature)
        return self.proj_act(self.proj_norm(out, mask)).masked_fill(~mask, 0)


class LocalCoordinationBlock(nn.Module):
    """CTR-GCN style spatial branches followed by multi-scale temporal fusion."""

    def __init__(self, in_channels, out_channels, joint_graph, main_graph, centers, owners, dilation=1):
        super().__init__()
        self.register_buffer("centers", centers.clone())
        self.register_buffer("owners", owners.clone())
        # CTR-GCN applies channel-wise topology refinement to every node and
        # keeps the three spatial partitions as separate branches.
        self.spatial_ctr = nn.ModuleList([
            MainNodeCTR(in_channels, out_channels, graph)
            for graph in joint_graph
        ])
        self.temporal = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, (3, 1),
                      padding=(dilation, 0), dilation=(dilation, 1), bias=False),
            nn.Conv2d(out_channels, out_channels, (5, 1),
                      padding=(2 * dilation, 0), dilation=(dilation, 1), bias=False),
            nn.Conv2d(out_channels, out_channels, (1, 1), bias=False),
            nn.Conv2d(out_channels, out_channels, (3, 1),
                      padding=(dilation, 0), dilation=(dilation, 1), bias=False),
        ])
        self.temporal_norm = PointBatchNorm(out_channels)
        self.residual = (nn.Identity() if in_channels == out_channels
                         else nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.act = nn.ReLU()

    def forward(self, x, mask, *, fine_enabled=True):
        residual = self.residual(x).masked_fill(~mask, 0)
        spatial = sum(branch(x, mask) for branch in self.spatial_ctr)
        spatial = self.act(spatial).masked_fill(~mask, 0)
        temporal = sum(branch(spatial) for branch in self.temporal)
        temporal = self.temporal_norm(temporal, mask)
        return self.act(temporal + residual).masked_fill(~mask, 0)


class RTMWLocalCTR(nn.Module):
    """CTR-GCN style full-node RTMW-133 classifier with mask-aware input handling."""

    DEFAULT_CHANNELS = (64, 64, 64, 96, 128, 128, 128, 192, 256, 256)

    def __init__(self, num_classes: int = 120, *, channels=None) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        channels = tuple(self.DEFAULT_CHANNELS if channels is None else channels)
        if not channels or any(c < 1 for c in channels):
            raise ValueError("channels must be a nonempty sequence of positive integers")
        partition = build_region_partition("rtmw_133", 133)
        centers = torch.tensor(partition.center_joint_indices, dtype=torch.long)
        owners = torch.tensor(partition.joint_to_region, dtype=torch.long)
        self.register_buffer("main_joint_indices", centers)
        self.register_buffer("joint_to_main", owners)
        # Full-node CTR is the default path, matching the original CTR-GCN
        # topology refinement granularity.
        self.register_buffer("fine_stage", torch.tensor(True))
        self._fine_enabled = True
        self.register_load_state_dict_post_hook(self._restore_stage)
        graph = build_joint_spatial_partitions(133, partition, "rtmw_133", scope="full")
        main_graph = normalize_adjacency_partitions(graph.index_select(1, centers).index_select(2, centers))
        self.register_buffer("joint_graph", graph)
        self.register_buffer("main_graph", main_graph)
        self.blocks = nn.ModuleList()
        in_channels = 3
        for index, out_channels in enumerate(channels):
            self.blocks.append(LocalCoordinationBlock(
                in_channels, out_channels, graph, main_graph, centers, owners,
                dilation=1 if index < 7 else (2 if index < 9 else 4),
            ))
            in_channels = out_channels
        # Keep the trained head across the coarse/fine boundary.
        self.classifier = nn.Linear(channels[-1], num_classes)

    @property
    def fine_enabled(self) -> bool:
        return self._fine_enabled

    def _restore_stage(self, module, incompatible_keys) -> None:
        self._fine_enabled = bool(self.fine_stage.item())

    def set_fine_enabled(self, enabled: bool) -> None:
        self._fine_enabled = bool(enabled)
        self.fine_stage.fill_(self._fine_enabled)

    def forward(self, x, valid_frame_mask=None, *, return_node_features=False):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.size(1) != 3 or x.size(3) != 133:
            raise ValueError("Expected B x 3 x T x 133 [relative x, relative y, score], optionally x M")
        b, c, t, n, m = x.shape
        if min(b, t, m) <= 0:
            raise ValueError("Batch, frame and person dimensions must be nonempty")
        mask = (x[:, 2:3] > 0) & torch.isfinite(x).all(dim=1, keepdim=True)
        if valid_frame_mask is not None:
            if valid_frame_mask.shape != (b, t):
                raise ValueError("valid_frame_mask must have shape B x T")
            mask = mask & valid_frame_mask.to(device=x.device, dtype=torch.bool)[:, None, :, None, None]
        x = x.permute(0, 4, 1, 2, 3).reshape(b * m, c, t, n)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(b * m, 1, t, n)
        x = x.masked_fill(~mask, 0)
        for block in self.blocks:
            x = block(x, mask, fine_enabled=True)
        nodes = x.size(-1)
        features = x.reshape(b, m, x.size(1), t, nodes)
        point_mask = mask.reshape(b, m, 1, t, nodes)
        count = point_mask.sum(dim=(1, 3, 4)).clamp_min(1)
        pooled = features.sum(dim=(1, 3, 4)) / count
        logits = self.classifier(pooled)
        if return_node_features:
            return {"logits": logits, "node_features": features, "node_mask": point_mask,
                    "node_indices": torch.arange(133, device=x.device)}
        return logits
