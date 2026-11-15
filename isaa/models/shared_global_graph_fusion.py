"""Pooled wide-relative model with one sparse global graph.

The original body, two-hand and face encoders remain region-specific.  A
single 70-node graph is used for one sparse exchange layer after those
encoders, so the regions can communicate without materializing a dense
70-by-70 adjacency tensor.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from isaa.graph.adjacency import build_joint_spatial_partitions
from isaa.graph.regions import build_region_partition
from isaa.layouts import register_skeleton_presets
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion


SHARED_GLOBAL_GRAPH_VARIANT = "body-local-shared-global-graph-100"


class SparseGlobalGraphExchange(nn.Module):
    """Mask-aware sparse message passing over a fixed directed edge list."""

    def __init__(self, source: torch.Tensor, destination: torch.Tensor,
                 weights: torch.Tensor, width: int, num_nodes: int) -> None:
        super().__init__()
        self.register_buffer("source", source.to(dtype=torch.long))
        self.register_buffer("destination", destination.to(dtype=torch.long))
        self.register_buffer("edge_weights", weights.to(dtype=torch.float32))
        self.num_nodes = int(num_nodes)
        self.mix = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or mask.shape != (x.shape[0], 1, x.shape[2], x.shape[3]):
            raise ValueError("SparseGlobalGraphExchange expects N x C x T x V and N x 1 x T x V")
        source_valid = mask.index_select(-1, self.source).to(x.dtype)
        values = x.index_select(-1, self.source) * source_valid
        edge_weights = self.edge_weights.to(device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        values = values * edge_weights
        # Keep the aggregation out-of-place.  ``index_add_`` inside an
        # Inductor CUDA-Graph capture can leave a stale alias after many
        # iterations (the A6000 run eventually reported an illegal memory
        # access while freeing pinned memory).  ``scatter_add`` uses the same
        # sparse edge semantics without mutating a graph input/output buffer.
        edge_index = self.destination.view(1, 1, 1, -1).expand_as(values)
        aggregate = torch.zeros_like(x).scatter_add(-1, edge_index, values)
        valid_index = self.destination.view(1, 1, 1, -1).expand_as(source_valid)
        denominator = torch.zeros_like(mask, dtype=x.dtype).scatter_add(
            -1, valid_index, source_valid * edge_weights
        )
        aggregate = aggregate / denominator.clamp_min(1e-6)
        return self.mix(aggregate).masked_fill(~mask, 0)


def _append_matrix_edges(edges: dict[tuple[int, int], float], matrix: torch.Tensor,
                         offset: int = 0) -> None:
    """Add nonzero directed entries from a spatial graph partition."""
    for row, column in torch.nonzero(matrix > 0, as_tuple=False).tolist():
        key = (offset + int(row), offset + int(column))
        edges[key] += float(matrix[row, column].item())


def _append_bidirectional_edge(edges: dict[tuple[int, int], float], left: int,
                               right: int, weight: float = 1.0) -> None:
    edges[(left, right)] += weight
    edges[(right, left)] += weight


class SharedGlobalGraphFusion(BodyLocalHandCTRWideRelativeFusion):
    """Keep regional encoders and add one sparse exchange on a shared graph."""

    ARCHITECTURE = "rtmw_ctr22_handctr21_wide_face6_regional_sparse_global_graph_v1"
    GLOBAL_WIDTH = 48
    GLOBAL_NODE_COUNT = 70

    def __init__(self, num_classes: int = 120, *, window_size: int = 100,
                 max_persons: int = 2) -> None:
        if window_size != 100:
            raise ValueError("Shared global graph fusion requires the 100-frame baseline")
        super().__init__(num_classes=num_classes)
        self.window_size = int(window_size)
        self.max_persons = int(max_persons)
        self.global_node_order = tuple(self.BODY_INDICES) + tuple(self.HAND_INDICES) + tuple(
            f"face_token_{index}" for index in range(6)
        )
        if len(self.global_node_order) != self.GLOBAL_NODE_COUNT:
            raise RuntimeError("Unexpected shared graph node order")
        self.region_slices = {
            "body": (0, len(self.BODY_INDICES)),
            "left_hand": (len(self.BODY_INDICES), len(self.BODY_INDICES) + 21),
            "right_hand": (len(self.BODY_INDICES) + 21, len(self.BODY_INDICES) + 42),
            "face": (len(self.BODY_INDICES) + 42, self.GLOBAL_NODE_COUNT),
        }

        source, destination, weights = self._build_sparse_graph()
        self.global_graph = SparseGlobalGraphExchange(
            source, destination, weights, self.GLOBAL_WIDTH, self.GLOBAL_NODE_COUNT
        )
        self.body_to_global = nn.Conv2d(self.CHANNELS[-1], self.GLOBAL_WIDTH, 1, bias=False)
        self.global_to_body = nn.Conv2d(self.GLOBAL_WIDTH, self.CHANNELS[-1], 1, bias=False)
        # Keep the new path a small residual at initialization while allowing
        # gradients through the exchange layer from the first update.
        self.global_scale = nn.Parameter(torch.tensor(0.1))
        self.global_graph_config = {
            "node_order": list(self.global_node_order),
            "node_count": self.GLOBAL_NODE_COUNT,
            "regions": {name: list(bounds) for name, bounds in self.region_slices.items()},
            "edge_count": int(source.numel()),
            "exchange": "mask_aware_sparse_index_add_residual",
            "cross_region_edges": [
                {"body_raw": 9, "hand_raw": 91},
                {"body_raw": 10, "hand_raw": 112},
                {"body_raw": 0, "face_token": 0},
                {"body_raw": 0, "face_token": 1},
                {"body_raw": 0, "face_token": 2},
                {"body_raw": 0, "face_token": 3},
                {"body_raw": 0, "face_token": 4},
                {"body_raw": 0, "face_token": 5},
            ],
            "initial_scale": 0.1,
        }

    def _build_sparse_graph(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        register_skeleton_presets()
        partition = build_region_partition("rtmw_133", 133)
        full_graph = build_joint_spatial_partitions(
            133, partition, "rtmw_133", scope="full"
        )
        edges: dict[tuple[int, int], float] = defaultdict(float)
        body_ids = torch.tensor(self.BODY_INDICES, dtype=torch.long)
        hand_ids = torch.tensor(self.HAND_INDICES, dtype=torch.long)
        left_ids, right_ids = hand_ids[:21], hand_ids[21:]
        for matrix in full_graph:
            _append_matrix_edges(
                edges,
                matrix.index_select(0, body_ids).index_select(1, body_ids),
            )
            _append_matrix_edges(
                edges,
                matrix.index_select(0, left_ids).index_select(1, left_ids),
                len(self.BODY_INDICES),
            )
            _append_matrix_edges(
                edges,
                matrix.index_select(0, right_ids).index_select(1, right_ids),
                len(self.BODY_INDICES) + 21,
            )

        # The parent face compressor already defines the six-token topology.
        face_graph = self.local_blocks[0].adjacency[42:, 42:].detach().cpu()
        face_offset = len(self.BODY_INDICES) + 42
        _append_matrix_edges(edges, face_graph, face_offset)

        body_position = {joint: index for index, joint in enumerate(self.BODY_INDICES)}
        left_offset = len(self.BODY_INDICES)
        right_offset = left_offset + 21
        _append_bidirectional_edge(edges, body_position[9], left_offset, 1.0)
        _append_bidirectional_edge(edges, body_position[10], right_offset, 1.0)
        # Connect the coarse nose/body head to every face token.  The face
        # tokens are ordered by the existing FaceTokenCompression mapping.
        for face_index in range(6):
            _append_bidirectional_edge(edges, body_position[0], face_offset + face_index, 0.5)

        ordered = sorted(edges.items())
        source = torch.tensor([left for (left, _), _ in ordered], dtype=torch.long)
        destination = torch.tensor([right for (_, right), _ in ordered], dtype=torch.long)
        weights = torch.tensor([weight for _, weight in ordered], dtype=torch.float32)
        return source, destination, weights

    @staticmethod
    def _resize_mask(mask: torch.Tensor, target: int) -> torch.Tensor:
        if mask.shape[2] == target:
            return mask
        return F.interpolate(mask.float(), size=(target, mask.shape[3]), mode="nearest").bool()

    @staticmethod
    def _resize_feature(feature: torch.Tensor, target: int) -> torch.Tensor:
        if feature.shape[2] == target:
            return feature
        return F.interpolate(
            feature, size=(target, feature.shape[3]), mode="bilinear", align_corners=False
        )

    def _exchange_regions(self, body, left, right, face):
        body_x, body_mask = body
        left_x, left_mask = left
        right_x, right_mask = right
        face_x, face_mask = face
        target = max(body_x.shape[2], left_x.shape[2], right_x.shape[2], face_x.shape[2])
        body_u = self._resize_feature(body_x, target)
        left_u = self._resize_feature(left_x, target)
        right_u = self._resize_feature(right_x, target)
        face_u = self._resize_feature(face_x, target)
        body_m = self._resize_mask(body_mask, target)
        left_m = self._resize_mask(left_mask, target)
        right_m = self._resize_mask(right_mask, target)
        face_m = self._resize_mask(face_mask, target)
        features = torch.cat((self.body_to_global(body_u), left_u, right_u, face_u), dim=-1)
        mask = torch.cat((body_m, left_m, right_m, face_m), dim=-1)
        exchanged = self.global_graph(features, mask) * self.global_scale
        body_delta = self.global_to_body(exchanged[..., self.region_slices["body"][0]:self.region_slices["body"][1]])
        left_delta = exchanged[..., self.region_slices["left_hand"][0]:self.region_slices["left_hand"][1]]
        right_delta = exchanged[..., self.region_slices["right_hand"][0]:self.region_slices["right_hand"][1]]
        face_delta = exchanged[..., self.region_slices["face"][0]:self.region_slices["face"][1]]
        body_delta = self._resize_feature(body_delta, body_x.shape[2]).masked_fill(~body_mask, 0)
        left_delta = self._resize_feature(left_delta, left_x.shape[2]).masked_fill(~left_mask, 0)
        right_delta = self._resize_feature(right_delta, right_x.shape[2]).masked_fill(~right_mask, 0)
        face_delta = self._resize_feature(face_delta, face_x.shape[2]).masked_fill(~face_mask, 0)
        return ((body_x + body_delta).masked_fill(~body_mask, 0), body_mask), \
            ((left_x + left_delta).masked_fill(~left_mask, 0), left_mask), \
            ((right_x + right_delta).masked_fill(~right_mask, 0), right_mask), \
            ((face_x + face_delta).masked_fill(~face_mask, 0), face_mask)

    def _exchange_encoded_branches(self, body, left, right, face):
        return self._exchange_regions(body, left, right, face)

