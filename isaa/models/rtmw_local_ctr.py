"""32-joint CTR-GCN with low-width RTMW-133 spatiotemporal region fusion."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.graph.adjacency import build_joint_spatial_partitions, normalize_adjacency_partitions
from isaa.graph.regions import build_region_partition
from isaa.models.normalization import _masked_stats_chunk_size


class PointBatchNorm(nn.Module):
    """Ignore missing observations; support shared or per-channel validity masks."""

    def __init__(self, channels: int, *, native: bool = False) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.native = bool(native)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.native:
            return self.bn(x).masked_fill(~mask, 0)
        if not self.training:
            return self.bn(x).masked_fill(~mask, 0)
        stats = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        count = mask.to(stats.dtype).sum(dim=(0, 2, 3))
        chunk_size = _masked_stats_chunk_size(stats.size(0), stats[0].numel())
        sums, squares = [], []
        for start in range(0, stats.size(0), chunk_size):
            chunk = stats[start:start + chunk_size].masked_fill(~mask[start:start + chunk_size], 0)
            sums.append(chunk.sum(dim=(0, 2, 3)))
            squares.append(chunk.square().sum(dim=(0, 2, 3)))
        mean = torch.stack(sums).sum(0) / count.clamp_min(1)
        var = (torch.stack(squares).sum(0) / count.clamp_min(1) - mean.square()).clamp_min(0)
        with torch.no_grad():
            update = (count > 0).to(mean.dtype) * self.bn.momentum
            self.bn.num_batches_tracked.add_((count.sum() > 0).long())
            unbiased = var * count / (count - 1).clamp_min(1)
            self.bn.running_mean.lerp_(mean.detach(), update)
            self.bn.running_var.lerp_(unbiased.detach(), update)
        out = (stats - mean[None, :, None, None]) * torch.rsqrt(var[None, :, None, None] + self.bn.eps)
        out = out * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]
        return out.to(x.dtype).masked_fill(~mask, 0)


def _downsample_mask(mask: torch.Tensor, stride: int) -> torch.Tensor:
    if stride == 1:
        return mask
    # Retain a bin with any valid observation, including the last partial bin.
    return F.max_pool2d(mask.float(), (stride, 1), (stride, 1), ceil_mode=True).bool()


class FixedSkeletonConv(nn.Module):
    """Self/inward/outward projections with frozen RTMW connectivity."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency.clone())
        self.projection = nn.Conv2d(in_channels, 3 * out_channels, 1, bias=False)
        self.out_channels = out_channels
        self.register_buffer("edge_partitions", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_targets", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_sources", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_weights", torch.empty(0), persistent=False)
        self._refresh_edges()
        self.register_load_state_dict_post_hook(self._restore_edges)

    def _refresh_edges(self):
        partitions, targets, sources = self.adjacency.nonzero(as_tuple=True)
        self.edge_partitions, self.edge_targets, self.edge_sources = partitions, targets, sources
        self.edge_weights = self.adjacency[partitions, targets, sources]

    def _restore_edges(self, module, incompatible_keys):
        self._refresh_edges()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, n = x.shape
        projected = self.projection(x).reshape(b, 3, self.out_channels, t, n)
        # Preserve the normalized target/source weights, without multiplying
        # all 3 * N * N positions of the mostly empty adjacency tensor.
        messages = projected.permute(0, 2, 3, 1, 4).flatten(3)
        messages = messages.index_select(3, self.edge_partitions * n + self.edge_sources)
        messages = messages * self.edge_weights.to(projected.dtype)[None, None, None, :]
        out = projected.new_zeros(b, self.out_channels, t, n)
        return out.index_add_(3, self.edge_targets, messages)


class FaceTokenCompression(nn.Module):
    """Masked raw-input means: 65 unchanged joints followed by six face tokens."""

    FACE_GROUPS = (tuple(range(23, 34)), tuple(range(34, 45)), tuple(range(45, 56)),
                   tuple(range(56, 67)), tuple(range(67, 79)), tuple(range(79, 91)))

    def __init__(self, adjacency, owners):
        super().__init__()
        nonface = torch.tensor(list(range(23)) + list(range(91, 133)), dtype=torch.long)
        mapping = torch.empty(133, dtype=torch.long)
        mapping[nonface] = torch.arange(65)
        members = torch.zeros(6, 12, dtype=torch.long)
        valid = torch.zeros(6, 12, dtype=torch.bool)
        for index, group in enumerate(self.FACE_GROUPS):
            members[index, :len(group)] = torch.tensor(group)
            valid[index, :len(group)] = True
            mapping[list(group)] = 65 + index
        self.register_buffer("nonface_indices", nonface)
        self.register_buffer("face_indices", members)
        self.register_buffer("face_members", valid)
        self.register_buffer("original_to_token", mapping)
        self.register_buffer("token_owners", torch.cat((owners[nonface], owners[members[:, 0]])))
        # Contract original edges, deduplicate them, and normalize again. Edges
        # internal to a token become its single self-loop, not directional loops.
        compressed = adjacency.new_zeros(3, 71, 71)
        compressed[0] = torch.eye(71, dtype=adjacency.dtype)
        for part in (1, 2):
            targets, sources = (adjacency[part] != 0).nonzero(as_tuple=True)
            targets, sources = mapping[targets], mapping[sources]
            keep = targets != sources
            compressed[part, targets[keep], sources[keep]] = 1
        self.register_buffer("adjacency", normalize_adjacency_partitions(compressed))

    def forward(self, x, mask):
        b, c, t, _ = x.shape
        indices = self.face_indices.flatten()
        face = x.index_select(-1, indices).reshape(b, c, t, 6, 12)
        valid = mask.index_select(-1, indices).reshape(b, 1, t, 6, 12)
        valid = valid & self.face_members[None, None, None]
        count = valid.sum(-1)
        face = face.masked_fill(~valid, 0).sum(-1) / count.clamp_min(1)
        nonface_mask = mask.index_select(-1, self.nonface_indices)
        nonface = x.index_select(-1, self.nonface_indices).masked_fill(~nonface_mask, 0)
        return torch.cat((nonface, face), dim=-1), torch.cat((nonface_mask, count > 0), dim=-1)


class AuxiliarySkeleton(nn.Module):
    """Low-width fixed graph layers plus depthwise temporal detail modeling."""

    def __init__(self, adjacency: torch.Tensor, channels: int, *, nonface_count: int | None = None) -> None:
        super().__init__()
        self.nonface_count = nonface_count
        self.layers = nn.ModuleList([
            FixedSkeletonConv(3, channels, adjacency),
            FixedSkeletonConv(channels, channels, adjacency),
        ])
        self.norms = nn.ModuleList([PointBatchNorm(channels) for _ in self.layers])
        self.temporal = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0),
                                  groups=channels, bias=False)
        self.temporal_mix = nn.Conv2d(channels, channels, 1, bias=False)
        self.temporal_norm = PointBatchNorm(channels)
        if nonface_count is not None:
            self.face_temporal = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0),
                                           groups=channels, bias=False)
            self.face_mix = nn.Conv2d(channels, channels, 1, bias=False)
            self.face_norm = PointBatchNorm(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = F.relu(norm(layer(x), mask)).masked_fill(~mask, 0)
        if self.nonface_count is None:
            temporal = self.temporal_mix(self.temporal(x).masked_fill(~mask, 0))
            return F.relu(x + self.temporal_norm(temporal, mask)).masked_fill(~mask, 0)
        nonface, face = x[..., :self.nonface_count], x[..., self.nonface_count:]
        nonmask, face_mask = mask[..., :self.nonface_count], mask[..., self.nonface_count:]
        nonface = self.temporal_mix(self.temporal(nonface).masked_fill(~nonmask, 0))
        nonface = self.temporal_norm(nonface, nonmask)
        face = self.face_mix(self.face_temporal(face).masked_fill(~face_mask, 0))
        face = self.face_norm(face, face_mask)
        return F.relu(x + torch.cat((nonface, face), dim=-1)).masked_fill(~mask, 0)


class RegionalDetailPool(nn.Module):
    """Per-frame regional mean plus masked learned attention; never mix regions."""

    def __init__(self, channels, owners, num_regions):
        super().__init__()
        # Padded region membership is fixed metadata; a single gather and
        # softmax process all regions without a Python loop in forward.
        members = [torch.where(owners == index)[0] for index in range(num_regions)]
        width = max(indices.numel() for indices in members)
        indices = owners.new_zeros(num_regions, width)
        member_mask = torch.zeros(num_regions, width, dtype=torch.bool, device=owners.device)
        for index, joints in enumerate(members):
            indices[index, :joints.numel()] = joints
            member_mask[index, :joints.numel()] = True
        self.register_buffer("member_indices", indices)
        self.register_buffer("member_mask", member_mask)
        self.score = nn.Conv2d(channels, 1, 1, bias=False)
        self.num_regions = num_regions

    def forward(self, x, mask):
        b, c, t, _ = x.shape
        indices = self.member_indices.flatten()
        values = x.index_select(-1, indices).reshape(b, c, t, self.num_regions, -1)
        valid = mask.index_select(-1, indices).reshape(b, 1, t, self.num_regions, -1)
        valid = valid & self.member_mask[None, None, None]
        values = values.masked_fill(~valid, 0)
        count = valid.sum(-1)
        region_mask = count > 0
        mean = values.sum(-1) / count.clamp_min(1)
        logits = self.score(x).index_select(-1, indices).reshape(b, 1, t, self.num_regions, -1)
        # Softmax in FP32; empty regions receive finite logits before softmax,
        # then all weights are zeroed, avoiding NaNs and invalid gradients.
        logits = logits.float().masked_fill(~valid, float("-inf"))
        logits = torch.where(region_mask.unsqueeze(-1), logits, torch.zeros_like(logits))
        attention = logits.softmax(-1).masked_fill(~valid, 0).to(x.dtype)
        weighted = (values * attention).sum(-1)
        return torch.cat((mean, weighted), dim=1), region_mask


class MainNodeCTR(nn.Module):
    """One CTR relation branch; normalization and residual belong to the GCN unit."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        relation_channels = 8 if in_channels in (3, 9) else max(1, in_channels // 8)
        self.theta = nn.Conv2d(in_channels, relation_channels, 1)
        self.phi = nn.Conv2d(in_channels, relation_channels, 1)
        self.feature_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.relation_proj = nn.Conv2d(relation_channels, out_channels, 1)

    def forward(self, x, mask, adjacency, alpha):
        weight = mask.to(x.dtype)
        count = weight.sum(dim=2).clamp_min(1)
        query = (self.theta(x) * weight).sum(dim=2) / count
        key = (self.phi(x) * weight).sum(dim=2) / count
        relation = torch.tanh(query.unsqueeze(-1) - key.unsqueeze(-2))
        topology = adjacency[None, None] + alpha * self.relation_proj(relation)
        feature = self.feature_proj(x).masked_fill(~mask, 0)
        return torch.einsum("bcuv,bctv->bctu", topology, feature).masked_fill(~mask, 0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, native_bn=False):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1),
                              stride=(stride, 1), padding=(dilation * (kernel_size - 1) // 2, 0),
                              dilation=(dilation, 1))
        self.norm = PointBatchNorm(out_channels, native=native_bn)

    def forward(self, x, mask):
        return self.norm(self.conv(x.masked_fill(~mask, 0)), _downsample_mask(mask, self.stride))


class CTRGraphConv(nn.Module):
    """Three CTR branches, a shared alpha, post-sum BN, and graph residual."""

    def __init__(self, in_channels, out_channels, adjacency, native_bn=False):
        super().__init__()
        self.static_topology = nn.Parameter(adjacency.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.branches = nn.ModuleList([MainNodeCTR(in_channels, out_channels) for _ in adjacency])
        self.norm = PointBatchNorm(out_channels, native=native_bn)
        self.residual = (TemporalConv(in_channels, out_channels, native_bn=native_bn)
                         if in_channels != out_channels else None)

    def forward(self, x, mask):
        out = sum(branch(x, mask, self.static_topology[i], self.alpha)
                  for i, branch in enumerate(self.branches))
        residual = x if self.residual is None else self.residual(x, mask)
        return F.relu(self.norm(out, mask) + residual).masked_fill(~mask, 0)


class MultiScaleTemporalConv(nn.Module):
    """CTR-GCN's two dilated, pooling, and pointwise branches, concatenated."""

    def __init__(self, channels: int, stride: int = 1, native_bn=False) -> None:
        super().__init__()
        if channels % 4:
            raise ValueError("Temporal output channels must be divisible by four")
        self.stride = stride
        width = channels // 4
        self.projections = nn.ModuleList([
            TemporalConv(channels, width, native_bn=native_bn) for _ in range(3)
        ])
        self.dilated = nn.ModuleList([
            TemporalConv(width, width, kernel_size=5, stride=stride, dilation=dilation,
                         native_bn=native_bn)
            for dilation in (1, 2)
        ])
        self.pool = nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0))
        self.pool_norm = PointBatchNorm(width, native=native_bn)
        self.pointwise = TemporalConv(channels, width, stride=stride, native_bn=native_bn)

    def forward(self, x, mask):
        out_mask = _downsample_mask(mask, self.stride)
        projected = [F.relu(layer(x, mask)) for layer in self.projections]
        branches = [layer(value, mask) for layer, value in zip(self.dilated, projected[:2])]
        branches.append(self.pool_norm(self.pool(projected[2]), out_mask))
        branches.append(self.pointwise(x, mask))
        return torch.cat(branches, dim=1).masked_fill(~out_mask, 0)


class CTRGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1, residual=True, native_bn=False):
        super().__init__()
        self.stride = stride
        self.use_residual = residual
        self.gcn = CTRGraphConv(in_channels, out_channels, adjacency, native_bn=native_bn)
        self.temporal = MultiScaleTemporalConv(out_channels, stride, native_bn=native_bn)
        self.residual = (TemporalConv(in_channels, out_channels, stride=stride)
                         if residual and (in_channels != out_channels or stride != 1) else None)

    def forward(self, x, mask):
        out = self.temporal(self.gcn(x, mask), mask)
        if self.use_residual:
            out = out + (x if self.residual is None else self.residual(x, mask))
        out_mask = _downsample_mask(mask, self.stride)
        return F.relu(out).masked_fill(~out_mask, 0), out_mask


class RTMWLocalCTR(nn.Module):
    """32-node CTR-GCN with early per-region fusion of all 133 joints."""

    STANDARD_CHANNELS = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256)
    COMPACT_CHANNELS = (48, 48, 48, 48, 96, 96, 96, 192, 192, 192)
    CHANNEL_PRESETS = {"compact": COMPACT_CHANNELS, "standard": STANDARD_CHANNELS}
    DEFAULT_CHANNELS = COMPACT_CHANNELS
    ARCHITECTURE = "rtmw_ctr32_face6_input_v4"

    def __init__(self, num_classes: int = 120, *, channels=None, auxiliary_channels: int = 16,
                 backbone_width: str = "compact", main_only: bool = False,
                 native_bn: bool | None = None) -> None:
        super().__init__()
        if backbone_width not in self.CHANNEL_PRESETS:
            raise ValueError(f"Unknown backbone width: {backbone_width!r}")
        self.backbone_width = backbone_width if channels is None else "custom"
        channels = tuple(self.CHANNEL_PRESETS[backbone_width] if channels is None else channels)
        if num_classes < 1 or auxiliary_channels < 1:
            raise ValueError("Class and auxiliary channel counts must be positive")
        if not channels or any(c < 4 or c % 4 for c in channels):
            raise ValueError("channels must contain positive multiples of four")
        self.channels = channels
        self.main_only = bool(main_only)
        self.native_bn = self.main_only if native_bn is None else bool(native_bn)
        if self.main_only:
            self.ARCHITECTURE = "rtmw_ctr32_only_v5"
        partition = build_region_partition("rtmw_133", 133)
        centers = torch.tensor(partition.center_joint_indices, dtype=torch.long)
        self.register_buffer("main_joint_indices", centers)
        self.register_buffer("joint_to_main", torch.tensor(partition.joint_to_region, dtype=torch.long))
        self.register_buffer("fine_stage", torch.tensor(not self.main_only))
        self._fine_enabled = not self.main_only
        self.register_load_state_dict_post_hook(self._restore_stage)
        graph = build_joint_spatial_partitions(133, partition, "rtmw_133", scope="full")
        main_graph = normalize_adjacency_partitions(graph.index_select(1, centers).index_select(2, centers))
        self.register_buffer("joint_graph", graph)
        self.register_buffer("main_graph", main_graph)
        # Per-joint statistics are shared across people, allowing masked empty
        # tracks without requiring a fixed number of people.
        self.input_norm = PointBatchNorm(3 * centers.numel(), native=self.native_bn)
        self.blocks = nn.ModuleList()
        in_channels = 3
        for index, out_channels in enumerate(channels):
            self.blocks.append(CTRGCNBlock(in_channels, out_channels, main_graph,
                                          stride=2 if index in (4, 7) else 1, residual=index != 0,
                                          native_bn=self.native_bn))
            in_channels = out_channels
        if not self.main_only:
            self.face_compression = FaceTokenCompression(graph, self.joint_to_main)
            self.auxiliary = AuxiliarySkeleton(self.face_compression.adjacency, auxiliary_channels, nonface_count=65)
            self.regional_pool = RegionalDetailPool(auxiliary_channels, self.face_compression.token_owners, centers.numel())
            self.auxiliary_to_main = nn.Conv2d(2 * auxiliary_channels, channels[0], 1, bias=False)
            self.auxiliary_scale = nn.Parameter(torch.tensor(0.1))
        self.classifier = nn.Linear(channels[-1], num_classes)
        self._initialize_weights()

    @property
    def experiment_name(self) -> str:
        width = self.backbone_width
        if width == "custom":
            width = "custom_" + "_".join(str(channel) for channel in self.channels)
        return f"{self.ARCHITECTURE}_{width}"

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.constant_(block.gcn.norm.bn.weight, 1e-6)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2 / self.classifier.out_features))
        nn.init.zeros_(self.classifier.bias)

    @property
    def fine_enabled(self) -> bool:
        """Compatibility flag: enable the auxiliary branch, never change main nodes."""
        return self._fine_enabled

    def _restore_stage(self, module, incompatible_keys) -> None:
        self.set_fine_enabled(bool(self.fine_stage.item()))

    def set_fine_enabled(self, enabled: bool) -> None:
        if enabled and self.main_only:
            raise ValueError("main_only models cannot enable an auxiliary branch")
        self._fine_enabled = bool(enabled)
        self.fine_stage.fill_(self._fine_enabled)

    @staticmethod
    def _pool(features, mask, batch, people):
        features = features.reshape(batch, people, *features.shape[1:])
        mask = mask.reshape(batch, people, *mask.shape[1:])
        count = mask.sum(dim=(1, 3, 4)).clamp_min(1)
        return features.sum(dim=(1, 3, 4)) / count

    def forward(self, x, valid_frame_mask=None, *, return_node_features=False):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        allowed_nodes = (32, 133) if self.main_only else (133,)
        if x.ndim != 5 or x.size(1) != 3 or x.size(3) not in allowed_nodes:
            raise ValueError(f"Expected B x 3 x T x N x M; N must be in {allowed_nodes}")
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
        auxiliary = auxiliary_mask = regional = region_mask = None
        if self.fine_enabled:
            auxiliary_input, auxiliary_mask = self.face_compression(x, mask)
            auxiliary = self.auxiliary(auxiliary_input, auxiliary_mask)
            regional, region_mask = self.regional_pool(auxiliary, auxiliary_mask)
        # A 32-node input is already ordered by main_joint_indices. The trainer
        # selects it in workers after RTMW torso normalization, before transfer.
        main = x if n == 32 else x.index_select(-1, self.main_joint_indices)
        main_mask = mask if n == 32 else mask.index_select(-1, self.main_joint_indices)
        norm_mask = main_mask.expand_as(main).permute(0, 1, 3, 2).reshape(b * m, c * 32, t, 1)
        main = main.permute(0, 1, 3, 2).reshape(b * m, c * 32, t, 1)
        main = self.input_norm(main, norm_mask).reshape(b * m, c, 32, t).permute(0, 1, 3, 2)
        temporal_stride = 1
        for index, block in enumerate(self.blocks):
            main, main_mask = block(main, main_mask)
            if index == 0 and regional is not None:
                detail = self.auxiliary_to_main(regional).masked_fill(~region_mask, 0)
                main = main + self.auxiliary_scale * detail
                # A valid regional detail can represent a missing center joint.
                main_mask = main_mask | region_mask
                main = main.masked_fill(~main_mask, 0)
            temporal_stride *= block.stride
        pooled = self._pool(main, main_mask, b, m)
        logits = self.classifier(pooled)
        if return_node_features:
            # Expand only for the legacy analysis view, never for classification.
            expanded = (auxiliary.index_select(-1, self.face_compression.original_to_token).masked_fill(~mask, 0)
                        if auxiliary is not None else None)
            return {
                "logits": logits,
                "node_features": main.reshape(b, m, *main.shape[1:]),
                "node_mask": main_mask.reshape(b, m, *main_mask.shape[1:]),
                "node_indices": self.main_joint_indices,
                "time_indices": torch.arange(main.size(2), device=x.device) * temporal_stride,
                "auxiliary_node_features": expanded.reshape(b, m, *expanded.shape[1:]) if expanded is not None else None,
                "auxiliary_node_mask": mask.reshape(b, m, 1, t, n) if auxiliary is not None else None,
                "auxiliary_node_indices": torch.arange(n, device=x.device) if auxiliary is not None else None,
                "auxiliary_token_features": auxiliary.reshape(b, m, *auxiliary.shape[1:]) if auxiliary is not None else None,
                "auxiliary_token_mask": (auxiliary_mask.reshape(b, m, *auxiliary_mask.shape[1:])
                                         if auxiliary is not None else None),
                "auxiliary_original_to_token": self.face_compression.original_to_token if auxiliary is not None else None,
                "regional_features": regional.reshape(b, m, *regional.shape[1:]) if regional is not None else None,
                "regional_mask": region_mask.reshape(b, m, *region_mask.shape[1:]) if region_mask is not None else None,
            }
        return logits
