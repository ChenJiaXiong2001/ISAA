"""固定功能区域划分。

HAPM 核心不内置任何具体关键点框架的区域表。
外部入口注册了语义区域时使用注册内容；没有语义区域时使用 generic 分组。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from isaa.skeleton_layout import RegionSpec, get_skeleton_layout
from isaa.spec import DEFAULT_NUM_REGIONS


@dataclass(frozen=True)
class RegionPartition:
    """人体功能区域定义。

    joint_to_region 使用 0 基下标：第 i 个关节属于第 joint_to_region[i] 个区域。
    names 和 joint_to_region 共同定义后续主节点读取和细节分支的语义基础。
    center_joint_indices 是每个区域的代表中心节点，由区域内最短路距离最小的节点自动确定。
    """

    names: tuple[str, ...]
    joint_to_region: tuple[int, ...]
    center_joint_indices: tuple[int, ...]

    @property
    def num_regions(self) -> int:
        """返回区域数量 K。"""
        return len(self.names)

    def joints_of(self, region_index: int) -> list[int]:
        """返回某个区域包含的所有关节下标。

        参数:
            region_index: 区域下标。

        返回:
            属于该区域的关节下标列表。
        """
        return [idx for idx, rid in enumerate(self.joint_to_region) if rid == region_index]


def _partition_from_region_specs(
    layout_name: str,
    region_specs: tuple[RegionSpec, ...],
    num_joints: int,
    joint_edges: tuple[tuple[int, int], ...] = (),
) -> RegionPartition:
    """从外部注册的区域定义构建 RegionPartition。

    参数:
        layout_name: 布局名称，用于错误提示。
        region_specs: 外部注册的区域定义。
        num_joints: 配置声明的关节点数量。
        joint_edges: 外部注册的自然骨架边，用于自动计算区域中心节点。

    抛错:
        区域下标越界、重复或遗漏节点时抛 ValueError。
    """
    joint_to_region = [-1] * num_joints
    names = tuple(spec.name for spec in region_specs)

    if len(set(names)) != len(names):
        raise ValueError(f"骨架布局 {layout_name!r} 存在重复区域名")

    for region_index, spec in enumerate(region_specs):
        for joint_index in spec.joint_indices:
            if joint_index < 0 or joint_index >= num_joints:
                raise ValueError(
                    f"骨架布局 {layout_name!r} 的区域 {spec.name!r} 包含越界关节 {joint_index}"
                )
            if joint_to_region[joint_index] >= 0:
                raise ValueError(f"骨架布局 {layout_name!r} 的关节 {joint_index} 被重复分配")
            joint_to_region[joint_index] = region_index

    if any(region_index < 0 for region_index in joint_to_region):
        missing = [idx for idx, region_index in enumerate(joint_to_region) if region_index < 0]
        raise ValueError(f"骨架布局 {layout_name!r} 的区域映射缺少关节: {missing}")
    center_joint_indices = _center_joint_indices(
        joint_to_region=tuple(joint_to_region),
        num_regions=len(names),
        joint_edges=joint_edges,
    )
    return RegionPartition(
        names=names,
        joint_to_region=tuple(joint_to_region),
        center_joint_indices=center_joint_indices,
    )


def _generic_partition(
    num_joints: int,
    num_regions: int = DEFAULT_NUM_REGIONS,
    joint_edges: tuple[tuple[int, int], ...] = (),
) -> RegionPartition:
    """未知节点布局的通用区域分组。

    参数:
        num_joints: 输入节点数。
        num_regions: 期望区域数，默认最多 15。
        joint_edges: 可选自然骨架边；没有时使用顺序链式边计算中心节点。

    返回:
        将节点按顺序均匀分配到若干通用区域的 RegionPartition。

    注意:
        generic 区域没有手/脚/脸语义，因此 detail.mode=auto 通常会关闭细节分支。
    """
    num_regions = max(1, min(num_regions, max(num_joints, 1)))
    names = tuple(f"通用区域{idx + 1}" for idx in range(num_regions))
    joint_to_region = tuple(min(idx * num_regions // max(num_joints, 1), num_regions - 1) for idx in range(num_joints))
    center_joint_indices = _center_joint_indices(
        joint_to_region=joint_to_region,
        num_regions=num_regions,
        joint_edges=joint_edges,
    )
    return RegionPartition(names=names, joint_to_region=joint_to_region, center_joint_indices=center_joint_indices)


def _center_joint_indices(
    joint_to_region: tuple[int, ...],
    num_regions: int,
    joint_edges: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    """按区域内最短路距离自动选择每个区域的中心节点。

    如果区域内部没有可用自然骨架边，候选节点只能到达自身；排序会退回到
    区域内编号最小的节点。这样找不到理想中心时仍会用同一区域其它节点替代。
    """
    num_joints = len(joint_to_region)
    if not joint_edges:
        joint_edges = tuple((idx, idx + 1) for idx in range(max(num_joints - 1, 0)))
    neighbors: list[list[int]] = [[] for _ in range(num_joints)]
    for left, right in joint_edges:
        if left < 0 or right < 0 or left >= num_joints or right >= num_joints:
            continue
        if joint_to_region[left] != joint_to_region[right]:
            continue
        neighbors[left].append(right)
        neighbors[right].append(left)

    centers: list[int] = []
    for region_index in range(num_regions):
        joints = [idx for idx, rid in enumerate(joint_to_region) if rid == region_index]
        if not joints:
            raise ValueError(f"区域 {region_index} 没有关节，无法计算中心节点")
        joint_set = set(joints)
        best_joint = joints[0]
        best_score: tuple[int, int, int] | None = None
        for candidate in joints:
            distances = _shortest_distances_within_region(candidate, joint_set, neighbors)
            unreachable = len(joints) - len(distances)
            total_distance = sum(distances.values())
            score = (unreachable, total_distance, candidate)
            if best_score is None or score < best_score:
                best_score = score
                best_joint = candidate
        centers.append(best_joint)
    return tuple(centers)


def _shortest_distances_within_region(
    source: int,
    joint_set: set[int],
    neighbors: list[list[int]],
) -> dict[int, int]:
    """在单一区域诱导子图内从 source 做 BFS，返回到可达区域节点的距离。"""
    distances = {source: 0}
    queue: deque[int] = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in neighbors[current]:
            if neighbor not in joint_set or neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def build_region_partition(layout: str, num_joints: int) -> RegionPartition:
    """构建关节到人体功能区域的固定映射。

    参数:
        layout: 区域布局名称。
        num_joints: 关键点数量。

    返回:
        注册布局提供 region_specs 时返回语义区域；否则返回 generic 均匀区域。
    """
    layout_spec = get_skeleton_layout(layout)
    if layout_spec is not None:
        if layout_spec.num_joints is not None and layout_spec.num_joints != num_joints:
            raise ValueError(
                f"骨架布局 {layout!r} 期望 {layout_spec.num_joints} 个关节，配置为 {num_joints}"
            )
        if layout_spec.region_specs:
            return _partition_from_region_specs(
                layout_name=layout,
                region_specs=layout_spec.region_specs,
                num_joints=num_joints,
                joint_edges=layout_spec.joint_edges,
            )
        if layout_spec.fallback_num_regions is not None:
            return _generic_partition(
                num_joints,
                num_regions=layout_spec.fallback_num_regions,
                joint_edges=layout_spec.joint_edges,
            )
    return _generic_partition(num_joints)


def region_membership_matrix(partition: RegionPartition, num_joints: int) -> torch.Tensor:
    """生成 K x N 的区域成员矩阵，用于中心代表回写和区域成员查询。

    参数:
        partition: 区域划分。
        num_joints: 模型实际使用的节点数 N。

    返回:
        K x N float 矩阵。matrix[k, n]=1 表示关节 n 属于区域 k。
    """
    # 技术备注：成员矩阵同时服务中心代表回写、区域成员查询和解释性遮挡实验。
    matrix = torch.zeros(partition.num_regions, num_joints, dtype=torch.float32)
    for joint_index, region_index in enumerate(partition.joint_to_region[:num_joints]):
        matrix[region_index, joint_index] = 1.0
    return matrix


def region_center_matrix(partition: RegionPartition, num_joints: int) -> torch.Tensor:
    """生成 K x N 的中心节点选择矩阵。

    matrix[k, n]=1 表示第 k 个区域以关节 n 作为中心代表。该矩阵用于直接读取
    中心节点特征，替代旧的区域内池化代表。
    """
    matrix = torch.zeros(partition.num_regions, num_joints, dtype=torch.float32)
    for region_index, joint_index in enumerate(partition.center_joint_indices):
        if joint_index < 0 or joint_index >= num_joints:
            raise ValueError(f"区域 {region_index} 的中心节点越界: {joint_index}")
        matrix[region_index, joint_index] = 1.0
    return matrix
