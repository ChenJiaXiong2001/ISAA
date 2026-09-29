"""Strict CTR-GCN baseline with the official RTMW-133 regional graph.

This module is intentionally separate from :mod:`rtmw_local_ctr`.  It keeps the
semantics of the reference CTR-GCN implementation: ordinary ``BatchNorm``
layers, no validity masks, no auxiliary branch, and one learned three-part
topology for every spatiotemporal block.  The RTMW graph below is a torch-only
port of ``CTR-GCN/graph/regional.py`` so experiments do not depend on a second
checkout of the reference repository.

The public model accepts ``B x 3 x T x 133 x M`` input.  ``num_person`` is
fixed at construction time, as in the official model, because it determines
the input BatchNorm1d width.  Set it to the number of person tracks in the
dataset (the official RTMW configuration uses one).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn

from isaa.layouts.rtmw_133 import RTMW_25_NODE_INDICES, RTMW_32_NODE_INDICES, get_rtmw_node_indices


# The original NTU graph uses one-based indices in its source file.  These are
# the corresponding zero-based ``inward`` pairs.  ``edge2mat`` below reverses
# each pair in the same way as CTR-GCN's graph/tools.py.
_BASE_INWARD: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
    (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12),
    (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
    (21, 22), (22, 7), (23, 24), (24, 11),
)

_RTMW_MAIN_NODES: tuple[int, ...] = RTMW_25_NODE_INDICES


def _rtmw_node_regions() -> tuple[int, ...]:
    """Return the 133-node to 25-region mapping from the reference graph."""

    regions = [3] * 133
    assignments = {
        0: 3, 1: 3, 2: 3, 3: 3, 4: 3,
        5: 4, 6: 8, 7: 5, 8: 9, 9: 6, 10: 10,
        11: 12, 12: 16, 13: 13, 14: 17, 15: 14, 16: 18,
        17: 15, 18: 15, 19: 15, 20: 19, 21: 19, 22: 19,
    }
    for node, region in assignments.items():
        regions[node] = region
    for node in range(23, 91):
        regions[node] = 3

    regions[91] = 6
    for node in range(92, 96):
        regions[node] = 22
    for node in range(96, 112):
        regions[node] = 7
    for node in (99, 103, 107, 111):
        regions[node] = 21

    regions[112] = 10
    for node in range(113, 117):
        regions[node] = 24
    for node in range(117, 133):
        regions[node] = 11
    for node in (120, 124, 128, 132):
        regions[node] = 23
    return tuple(regions)


RTMW_MAIN_NODES = _RTMW_MAIN_NODES
RTMW_NODE_REGIONS = _rtmw_node_regions()


def _dedupe_edges(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted({(int(source), int(target)) for source, target in edges})


def _edge2mat(edges: Iterable[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    """Port ``graph.tools.edge2mat`` (target row, source column)."""

    matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for source, target in edges:
        matrix[target, source] = 1.0
    return matrix


def _normalize_digraph(adjacency: torch.Tensor) -> torch.Tensor:
    """Port ``graph.tools.normalize_digraph`` (column normalization)."""

    degree = adjacency.sum(dim=0)
    inverse = torch.where(degree > 0, degree.reciprocal(), torch.zeros_like(degree))
    return adjacency * inverse.unsqueeze(0)


def build_official_rtmw_adjacency(
    *,
    connect_subnodes_to_related_regions: bool = True,
    fully_connect_regions: bool = False,
) -> torch.Tensor:
    """Build the official CTR-GCN RTMW-133 spatial graph.

    The reference ``regional.Graph(layout='rtmw133')`` expands the original
    25-node NTU graph to 133 points.  The returned tensor has shape
    ``3 x 133 x 133`` and branch order ``self, inward, outward``.  Each branch
    uses the reference column normalization; isolated nodes retain zero
    entries in the directional branches.
    """

    num_nodes = 133
    if len(RTMW_MAIN_NODES) != 25 or len(RTMW_NODE_REGIONS) != num_nodes:
        raise RuntimeError("RTMW regional graph metadata is inconsistent")

    region_nodes: list[list[int]] = [[] for _ in range(25)]
    for node, region in enumerate(RTMW_NODE_REGIONS):
        if not 0 <= region < 25:
            raise RuntimeError(f"invalid RTMW region index {region} at node {node}")
        region_nodes[region].append(node)

    inward: list[tuple[int, int]] = []
    # Keep the original CTR-GCN topology between the 25 region anchors.
    for child_region, parent_region in _BASE_INWARD:
        inward.append((RTMW_MAIN_NODES[child_region], RTMW_MAIN_NODES[parent_region]))

    # Connect every extra keypoint to its region anchor.
    for region, nodes in enumerate(region_nodes):
        anchor = RTMW_MAIN_NODES[region]
        for node in nodes:
            if node != anchor:
                inward.append((node, anchor))
        if fully_connect_regions:
            inward.extend((source, target) for source in nodes for target in nodes if source != target)

    # Give subnodes the same coarse inter-region access as their anchors.
    if connect_subnodes_to_related_regions:
        for child_region, parent_region in _BASE_INWARD:
            parent_anchor = RTMW_MAIN_NODES[parent_region]
            for node in region_nodes[child_region]:
                if node != parent_anchor:
                    inward.append((node, parent_anchor))
    inward = _dedupe_edges(inward)
    outward = [(target, source) for source, target in inward]
    self_link = [(node, node) for node in range(num_nodes)]
    return torch.stack((
        _normalize_digraph(_edge2mat(self_link, num_nodes)),
        _normalize_digraph(_edge2mat(inward, num_nodes)),
        _normalize_digraph(_edge2mat(outward, num_nodes)),
    ))


def build_official_ntu_adjacency() -> torch.Tensor:
    """Build the original CTR-GCN NTU RGB+D 25-joint graph."""
    inward = list(_BASE_INWARD)
    outward = [(target, source) for source, target in inward]
    self_link = [(node, node) for node in range(25)]
    return torch.stack((
        _normalize_digraph(_edge2mat(self_link, 25)),
        _normalize_digraph(_edge2mat(inward, 25)),
        _normalize_digraph(_edge2mat(outward, 25)),
    ))


def build_rtmw_adjacency_for_nodes(node_count: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return an induced official graph for a progressive RTMW node stage.

    The 25-node stage intentionally keeps repeated RTMW indices because those
    positions correspond to distinct NTU semantic joints.  Selecting rows and
    columns after constructing the 133-node regional graph preserves that
    semantic duplication while keeping the CTR-GCN three-branch topology.
    """
    indices = get_rtmw_node_indices(node_count)
    full = build_official_rtmw_adjacency()
    index = torch.tensor(indices, dtype=torch.long)
    selected = full.index_select(1, index).index_select(2, index)
    # Re-normalize each induced branch after removing unselected nodes.  This
    # matches CTR-GCN's graph preprocessing for the reduced node set.
    selected = torch.stack(tuple(_normalize_digraph(branch) for branch in selected))
    return selected, indices


def build_rtmw32_auxiliary_adjacency() -> torch.Tensor:
    """133-node graph with learnable 32 anchors and anchor-only auxiliaries."""
    from isaa.graph.regions import build_region_partition

    num_nodes = 133
    anchors = tuple(RTMW_32_NODE_INDICES)
    anchor_set = set(anchors)
    partition = build_region_partition("rtmw_133", num_nodes)
    if partition.num_regions != len(anchors):
        raise ValueError("RTMW region count must match the 32 main anchors")
    inward: list[tuple[int, int]] = []
    # Preserve the original NTU body topology on the first 25 semantic anchors.
    for child, parent in _BASE_INWARD:
        inward.append((anchors[child], anchors[parent]))
    # Auxiliary nodes only connect to their owning region anchor.
    for node in range(num_nodes):
        if node not in anchor_set:
            region = partition.joint_to_region[node]
            anchor = anchors[region]
            inward.append((node, anchor))
    outward = [(target, source) for source, target in inward]
    self_link = [(node, node) for node in range(num_nodes)]
    return torch.stack((
        _normalize_digraph(_edge2mat(self_link, num_nodes)),
        _normalize_digraph(_edge2mat(inward, num_nodes)),
        _normalize_digraph(_edge2mat(outward, num_nodes)),
    ))


def conv_init(conv: nn.Conv2d) -> None:
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.BatchNorm1d | nn.BatchNorm2d, scale: float) -> None:
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(module: nn.Module) -> None:
    """Initialization helper copied from the reference CTR-GCN model."""

    classname = module.__class__.__name__
    if "Conv" in classname and hasattr(module, "weight"):
        weight = getattr(module, "weight")
        if weight is not None:
            nn.init.kaiming_normal_(weight, mode="fan_out")
        bias = getattr(module, "bias", None)
        if bias is not None:
            nn.init.constant_(bias, 0)
    elif "BatchNorm" in classname:
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if weight is not None:
            weight.data.normal_(1.0, 0.02)
        if bias is not None:
            bias.data.zero_()


class TemporalConv(nn.Module):
    """Reference ``unit_tcn`` / temporal branch convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, 1),
            padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MultiScaleTemporalConv(nn.Module):
    """Reference multi-scale temporal convolution.

    ``TCNGCNUnit`` passes dilations ``(1, 2)`` by default, yielding the four
    branches used by the official NTU configurations: two dilated branches,
    max-pooling, and a strided pointwise branch.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilations: Sequence[int] = (1, 2),
                 residual: bool = True, residual_kernel_size: int = 1) -> None:
        super().__init__()
        dilations = tuple(int(value) for value in dilations)
        if not dilations:
            raise ValueError("dilations must be non-empty")
        branch_count = len(dilations) + 2
        if out_channels % branch_count:
            raise ValueError("out_channels must be divisible by the number of temporal branches")
        branch_channels = out_channels // branch_count
        kernels = (tuple(kernel_size for _ in dilations)
                   if isinstance(kernel_size, int) else tuple(kernel_size))
        if len(kernels) != len(dilations):
            raise ValueError("kernel_size and dilations must have equal lengths")

        self.num_branches = branch_count
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size=branch_kernel,
                             stride=stride, dilation=dilation),
            )
            for branch_kernel, dilation in zip(kernels, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0,
                      stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels),
        ))

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels,
                                         kernel_size=residual_kernel_size, stride=stride)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        # ``valid_frame_mask`` is accepted for trainer compatibility but is
        # intentionally ignored to preserve the official CTR-GCN semantics.
        del valid_frame_mask
        result = self.residual(x)
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1) + result


class CTRGC(nn.Module):
    """Channel-wise topology refinement from the reference implementation."""

    def __init__(self, in_channels: int, out_channels: int,
                 rel_reduction: int = 8, mid_reduction: int = 1) -> None:
        super().__init__()
        if in_channels in (3, 9):
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        if self.rel_channels < 1:
            raise ValueError("in_channels is too small for the relation reduction")
        self.conv1 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None,
                alpha: torch.Tensor | float = 1, support: torch.Tensor | None = None) -> torch.Tensor:
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)
        relation = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        refinement = self.conv4(relation) * alpha
        if adjacency is not None:
            refinement = refinement + adjacency.unsqueeze(0).unsqueeze(0)
        if support is not None:
            refinement = refinement * support.to(device=x.device, dtype=x.dtype)[None, None]
        return torch.einsum("bcuv,bctv->bctu", refinement, x3)


class UnitGCN(nn.Module):
    """Reference adaptive three-part spatial graph convolution."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor,
                 coff_embedding: int = 4, adaptive: bool = True,
                 residual: bool = True) -> None:
        super().__init__()
        del coff_embedding  # retained for constructor compatibility
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("adjacency must have shape P x V x V")
        self.num_subset = int(adjacency.shape[0])
        self.convs = nn.ModuleList([
            CTRGC(in_channels, out_channels) for _ in range(self.num_subset)
        ])
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        self.adaptive = bool(adaptive)
        if self.adaptive:
            self.PA = nn.Parameter(adjacency.clone())
        else:
            self.register_buffer("A", adjacency.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(dim=-2)
        self.relu = nn.ReLU(inplace=True)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.PA if self.adaptive else self.A.to(device=x.device, dtype=x.dtype)
        output = None
        for index, conv in enumerate(self.convs):
            value = conv(x, adjacency[index], self.alpha)
            output = value if output is None else output + value
        output = self.bn(output)
        output = output + self.down(x)
        return self.relu(output)


class LocalUnitGCN(nn.Module):
    """CTR-GCN with adaptive topology restricted to the original graph support."""
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor,
                 learnable_nodes: Sequence[int] | None = None) -> None:
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.num_subset = int(adjacency.shape[0])
        self.PA = nn.Parameter(adjacency.clone())
        self.register_buffer("support", adjacency.ne(0))
        if learnable_nodes is None:
            learnable_mask = torch.ones(adjacency.shape[1:], dtype=torch.bool)
        else:
            learnable_mask = torch.zeros(adjacency.shape[1:], dtype=torch.bool)
            indices = torch.as_tensor(tuple(learnable_nodes), dtype=torch.long)
            learnable_mask[indices[:, None], indices[None, :]] = True
        self.register_buffer("learnable_mask", learnable_mask)
        self.PA.register_hook(lambda grad: grad * self.learnable_mask[None].to(grad.dtype))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.convs = nn.ModuleList([CTRGC(in_channels, out_channels) for _ in range(self.num_subset)])
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1e-6)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = None
        for index, conv in enumerate(self.convs):
            # Preserve CTRGCN's data-dependent relation refinement and learned
            # adjacency values, but disallow learned connections off the
            # original RTMW32 graph support.
            value = conv(x, self.PA[index], self.alpha, self.support[index])
            output = value if output is None else output + value
        return self.relu(self.bn(output) + self.down(x))


class TCNGCNUnit(nn.Module):
    """Reference ``TCN_GCN_unit`` block."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor,
                 stride: int = 1, residual: bool = True, adaptive: bool = True,
                 kernel_size: int = 5, dilations: Sequence[int] = (1, 2),
                 local_graph: bool = False, learnable_graph_nodes: Sequence[int] | None = None) -> None:
        super().__init__()
        self.gcn1 = (LocalUnitGCN(in_channels, out_channels, adjacency,
                                  learnable_nodes=learnable_graph_nodes) if local_graph else
                     UnitGCN(in_channels, out_channels, adjacency, adaptive=adaptive))
        self.tcn1 = MultiScaleTemporalConv(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride,
            dilations=dilations, residual=False,
        )
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels,
                                         kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class OriginalCTRGCN(nn.Module):
    """Official CTR-GCN backbone adapted only for RTMW tensor dimensions."""

    ARCHITECTURE = "original_ctrgcn_rtmw133"
    STANDARD_CHANNELS = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256)

    def __init__(self, num_classes: int = 120, num_point: int = 133,
                 num_person: int = 1, graph: torch.Tensor | None = None,
                 in_channels: int = 3, drop_out: float = 0.0,
                 adaptive: bool = True, channels: Sequence[int] | None = None,
                 local_graph: bool = False,
                 learnable_graph_nodes: Sequence[int] | None = None) -> None:
        super().__init__()
        if graph is None:
            if num_point == 133:
                graph = build_official_rtmw_adjacency()
            else:
                raise ValueError("graph is required when num_point is not 133")
        adjacency = torch.as_tensor(graph, dtype=torch.float32)
        if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("graph must have shape P x V x V")
        if adjacency.shape[0] != 3 or adjacency.shape[1] != num_point:
            raise ValueError("graph shape must be 3 x num_point x num_point")
        if num_classes < 1 or num_point < 1 or num_person < 1 or in_channels < 1:
            raise ValueError("num_classes, num_point, num_person and in_channels must be positive")
        if drop_out < 0 or drop_out >= 1:
            raise ValueError("drop_out must be in [0, 1)")
        channel_list = tuple(self.STANDARD_CHANNELS if channels is None else channels)
        if len(channel_list) != 10 or any(channel < 1 for channel in channel_list):
            raise ValueError("channels must contain ten positive widths")
        if any(channel % 4 for channel in channel_list):
            raise ValueError("all channel widths must be divisible by four for the temporal branches")

        self.num_class = int(num_classes)
        self.num_point = int(num_point)
        self.num_person = int(num_person)
        self.in_channels = int(in_channels)
        self.channels = channel_list
        self.local_graph = bool(local_graph)
        # Keep the fixed graph available for inspection/device moves without
        # adding a duplicate ``A`` entry to official CTR-GCN checkpoints.
        self.register_buffer("A", adjacency.clone(), persistent=False)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        input_channels = in_channels
        for index, output_channels in enumerate(channel_list):
            setattr(self, f"l{index + 1}", TCNGCNUnit(
                input_channels, output_channels, self.A,
                stride=2 if index in (4, 7) else 1,
                residual=index != 0,
                adaptive=adaptive,
                local_graph=self.local_graph,
                learnable_graph_nodes=learnable_graph_nodes,
            ))
            input_channels = output_channels
        # Keep the reference attribute name (``fc``) for checkpoint inspection;
        # ``classifier`` below is a read-only convenience alias.
        self.fc = nn.Linear(channel_list[-1], num_classes)
        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Identity()
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_classes))
        nn.init.zeros_(self.fc.bias)
        bn_init(self.data_bn, 1)

    @property
    def blocks(self) -> tuple[TCNGCNUnit, ...]:
        """Return the ten reference layers without registering duplicate names."""

        return tuple(getattr(self, f"l{index}") for index in range(1, 11))

    @property
    def classifier(self) -> nn.Linear:
        """Convenience alias for the reference ``fc`` head."""

        return self.fc

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Accepted for compatibility with the shared training loop; the
        # reference implementation deliberately has no mask-aware path.
        del valid_frame_mask
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5:
            raise ValueError("Expected input shape B x C x T x V x M")
        batch, channels, frames, points, people = x.shape
        if (channels, points, people) != (self.in_channels, self.num_point, self.num_person):
            raise ValueError(
                f"Expected C,V,M=({self.in_channels},{self.num_point},{self.num_person}), "
                f"got ({channels},{points},{people})"
            )
        # Exactly the reference data_bn layout: BN channels are person × joint × input-channel.
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(
            batch, people * points * channels, frames
        )
        x = self.data_bn(x)
        x = x.view(batch, people, points, channels, frames).permute(
            0, 1, 3, 4, 2
        ).contiguous().view(batch * people, channels, frames, points)
        for index in range(1, 11):
            x = getattr(self, f"l{index}")(x)

        output_channels = x.size(1)
        x = x.view(batch, people, output_channels, -1).mean(dim=3).mean(dim=1)
        return self.fc(self.drop_out(x))


# Names matching the reference source make checkpoint and code comparisons easy.
unit_tcn = TemporalConv
MultiScale_TemporalConv = MultiScaleTemporalConv
unit_gcn = UnitGCN
TCN_GCN_unit = TCNGCNUnit
Model = OriginalCTRGCN


__all__ = [
    "RTMW_MAIN_NODES", "RTMW_NODE_REGIONS", "build_official_rtmw_adjacency",
    "build_rtmw_adjacency_for_nodes",
    "CTRGC", "TemporalConv", "MultiScaleTemporalConv", "UnitGCN", "TCNGCNUnit",
    "OriginalCTRGCN", "unit_tcn", "MultiScale_TemporalConv", "unit_gcn",
    "TCN_GCN_unit", "Model",
]
