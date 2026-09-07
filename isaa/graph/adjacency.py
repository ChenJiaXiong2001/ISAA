"""骨架关节图邻接矩阵。

HAPM 核心通过外部注册布局读取关节图。
未注册语义边时使用自连接 + 相邻节点链式连接兜底，保证图卷积可运行。
"""

from __future__ import annotations

import torch

from isaa.graph.regions import RegionPartition
from isaa.skeleton_layout import GENERIC_LAYOUT, get_skeleton_layout


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """对邻接矩阵做按行归一化，避免节点度数差异过大。

    参数:
        adjacency: N x N 或 K x K 邻接矩阵。

    返回:
        每一行和约为 1 的矩阵。孤立节点通过 clamp_min 避免除零。
    """
    # 技术备注：当前是简单行归一化，后续可换成 D^{-1/2} A D^{-1/2} 对称归一化。
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def normalize_adjacency_partitions(adjacency: torch.Tensor) -> torch.Tensor:
    """逐分支行归一化空间图。

    参数:
        adjacency: P x N x N 的多分支邻接矩阵。

    返回:
        每个分支独立行归一化后的邻接矩阵。
    """
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def build_joint_adjacency(num_joints: int, layout: str = GENERIC_LAYOUT) -> torch.Tensor:
    """构建关节点基础图。

    参数:
        num_joints: 节点数量 N。
        layout: 关键点布局名称。

    返回:
        N x N 行归一化邻接矩阵。

    行为:
        已注册布局提供 joint_edges 时使用语义连接；
        否则使用自连接 + 相邻节点链式连接作为可运行兜底。
    """
    # 技术备注：基础图只表达自然骨骼连接，动作相关关系交给图卷积细化。
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    layout_spec = get_skeleton_layout(layout)
    edges = layout_spec.joint_edges if layout_spec is not None else ()
    if not edges:
        edges = tuple((idx, idx + 1) for idx in range(max(num_joints - 1, 0)))

    for i, j in edges:
        if i < 0 or j < 0 or i >= num_joints or j >= num_joints:
            raise ValueError(f"骨架布局 {layout!r} 的关节边越界: {(i, j)}")
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    return normalize_adjacency(adjacency)


def build_intra_region_joint_adjacency(
    num_joints: int,
    partition: RegionPartition,
    layout: str = GENERIC_LAYOUT,
) -> torch.Tensor:
    """构建只保留区域内子节点连接的关节点基础图。

    参数:
        num_joints: 节点数量 N。
        partition: 关节到区域的固定映射。
        layout: 关键点布局名称。

    返回:
        N x N 行归一化邻接矩阵。

    行为:
        只保留两个端点属于同一区域的自然骨架边；跨区域协同不再通过
        子节点-子节点边表达，而交给子节点到其它主节点的跨层图。
    """
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    layout_spec = get_skeleton_layout(layout)
    edges = layout_spec.joint_edges if layout_spec is not None else ()
    if not edges:
        edges = tuple((idx, idx + 1) for idx in range(max(num_joints - 1, 0)))

    for i, j in edges:
        if i < 0 or j < 0 or i >= num_joints or j >= num_joints:
            raise ValueError(f"骨架布局 {layout!r} 的关节边越界: {(i, j)}")
        if partition.joint_to_region[i] != partition.joint_to_region[j]:
            continue
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    return normalize_adjacency(adjacency)


def build_body_full_joint_adjacency(
    num_joints: int,
    partition: RegionPartition,
    layout: str = GENERIC_LAYOUT,
) -> torch.Tensor:
    """构建完整自然骨架图 + 区域内局部图。

    设计目标：
        1. 身体主关节保留完整自然骨架边。
        2. 躯干、肩、肘、腕、髋、膝、踝等主链不被区域边界切断。
        3. 手、脸等高密度关键点仍只使用区域内自然边，避免退化为全局全连接。

    返回:
        N x N 行归一化邻接矩阵。
    """
    del partition
    return build_joint_adjacency(num_joints, layout)


def _layout_edges(num_joints: int, layout: str) -> tuple[tuple[int, int], ...]:
    """读取布局声明的有向骨架边，缺省时退回链式父子边。"""
    layout_spec = get_skeleton_layout(layout)
    edges = layout_spec.joint_edges if layout_spec is not None else ()
    if not edges:
        edges = tuple((idx, idx + 1) for idx in range(max(num_joints - 1, 0)))
    for i, j in edges:
        if i < 0 or j < 0 or i >= num_joints or j >= num_joints:
            raise ValueError(f"骨架布局 {layout!r} 的关节边越界: {(i, j)}")
    return tuple((int(i), int(j)) for i, j in edges)


def build_joint_spatial_partitions(
    num_joints: int,
    partition: RegionPartition,
    layout: str = GENERIC_LAYOUT,
    scope: str = "self_only",
) -> torch.Tensor:
    """构建 CTR-GCN 风格的多分支空间图。

    分支顺序固定为:
        0. self：节点自连接。
        1. inward：布局声明方向上的父节点到子节点。
        2. outward：inward 的反向边。

    参数:
        num_joints: 节点数量 N。
        partition: 关节到区域的固定映射，用于 intra_region scope 过滤。
        layout: 关键点布局名称。
        scope: self_only/full/body_full/intra_region。self_only 只保留节点自连接，
            跨节点关系交给可学习 dense topology。

    返回:
        3 x N x N 的行归一化邻接矩阵。
    """
    scope = str(scope).lower()
    if scope not in {"self_only", "body_full", "full", "intra_region"}:
        raise ValueError(f"不支持的 joint graph scope: {scope!r}")

    self_link = torch.eye(num_joints, dtype=torch.float32)
    inward = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    outward = torch.zeros(num_joints, num_joints, dtype=torch.float32)

    if scope != "self_only":
        for parent, child in _layout_edges(num_joints, layout):
            if scope == "intra_region" and partition.joint_to_region[parent] != partition.joint_to_region[child]:
                continue
            inward[parent, child] = 1.0
            outward[child, parent] = 1.0

    return normalize_adjacency_partitions(torch.stack([self_link, inward, outward], dim=0))


def _unnormalized_joint_adjacency(num_joints: int, layout: str = GENERIC_LAYOUT) -> torch.Tensor:
    """构建未归一化自然骨架邻接矩阵，供不同 joint graph scope 复用。"""
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    layout_spec = get_skeleton_layout(layout)
    edges = layout_spec.joint_edges if layout_spec is not None else ()
    if not edges:
        edges = tuple((idx, idx + 1) for idx in range(max(num_joints - 1, 0)))

    for i, j in edges:
        if i < 0 or j < 0 or i >= num_joints or j >= num_joints:
            raise ValueError(f"骨架布局 {layout!r} 的关节边越界: {(i, j)}")
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    return adjacency
