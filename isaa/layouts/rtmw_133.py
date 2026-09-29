"""RTMW / COCO-WholeBody 133 点骨架布局预设。

布局集中放在 isaa.layouts 中，避免模型核心与具体关键点框架绑定。
入口脚本调用 register() 后，配置中的 data.skeleton_layout="rtmw_133" 才会生效。
"""

from __future__ import annotations

from isaa.skeleton_layout import RegionSpec, SkeletonLayoutSpec, register_skeleton_layout


RTMW_133_LAYOUT_NAME = "rtmw_133"
RTMW_133_NUM_JOINTS = 133
RTMW_133_INPUT_CHANNELS = 5
RTMW_133_WINDOW_SIZE = 64
RTMW_133_NUM_REGIONS = 32
RTMW_133_NUM_SEGMENTS = 16
RTMW_133_PATTERN_DIM = 256
RTMW_133_BLOCK_CHANNELS = (64, 64, 64, 96, 128, 128, 128, 192, 256, 256)

# Stable node orders used by the progressive RTMW experiments.  The 25-node
# order follows the 25 semantic anchors used by the original CTR-GCN
# regional graph.  Repeated indices are intentional: RTMW has fewer physical
# anchors for a few NTU semantic joints, so the same observed point is exposed
# at each corresponding semantic position.  The 32-node order keeps one
# representative center for every RTMW region.
RTMW_25_NODE_INDICES = (
    11, 11, 5, 0, 5, 7, 9, 9, 6, 8, 10, 10, 11,
    13, 15, 17, 12, 14, 16, 20, 5, 103, 95, 124, 116,
)
RTMW_32_NODE_INDICES = (
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 56, 91, 95, 99, 103, 107,
    111, 112, 116, 120, 124, 128, 132,
)


def _expand_rtmw_nodes(target: int) -> tuple[int, ...]:
    """Deterministically extend the 32-node centers for coarse-to-fine runs."""
    base = list(RTMW_32_NODE_INDICES)
    used = set(base)
    for index in range(RTMW_133_NUM_JOINTS):
        if index not in used:
            base.append(index)
        if len(base) >= target:
            break
    return tuple(base[:target])


def get_rtmw_node_indices(node_count: int) -> tuple[int, ...]:
    """Return the canonical RTMW node order for a training stage.

    ``25`` and ``32`` are the two progressive coarse-to-fine stages.  ``133``
    preserves the complete RTMW observation.  Keeping this mapping in the
    layout module ensures the ZIP loader, preprocessing tool and both model
    variants use exactly the same node order.
    """
    node_count = int(node_count)
    if node_count == 25:
        return RTMW_25_NODE_INDICES
    if node_count == 32:
        return RTMW_32_NODE_INDICES
    if node_count in {50, 71}:
        return _expand_rtmw_nodes(node_count)
    if node_count == 133:
        return tuple(range(RTMW_133_NUM_JOINTS))
    raise ValueError("RTMW node_count 只支持 25、32、50、71 或 133")


RTMW_133_REGION_SPECS = (
    RegionSpec("头部区域", (0, 1, 2, 3, 4), "鼻、双眼、双耳聚合头部点"),
    RegionSpec("左肩区域", (5,), "左肩"),
    RegionSpec("右肩区域", (6,), "右肩"),
    RegionSpec("左肘区域", (7,), "左肘"),
    RegionSpec("右肘区域", (8,), "右肘"),
    RegionSpec("左腕区域", (9,), "左腕"),
    RegionSpec("右腕区域", (10,), "右腕"),
    RegionSpec("左髋区域", (11,), "左髋"),
    RegionSpec("右髋区域", (12,), "右髋"),
    RegionSpec("左膝区域", (13,), "左膝"),
    RegionSpec("右膝区域", (14,), "右膝"),
    RegionSpec("左踝区域", (15,), "左踝"),
    RegionSpec("右踝区域", (16,), "右踝"),
    RegionSpec("左脚尖区域", (17,), "左脚足部点 1"),
    RegionSpec("左脚跟区域", (18,), "左脚足部点 2"),
    RegionSpec("左脚外侧区域", (19,), "左脚足部点 3"),
    RegionSpec("右脚尖区域", (20,), "右脚足部点 1"),
    RegionSpec("右脚跟区域", (21,), "右脚足部点 2"),
    RegionSpec("右脚外侧区域", (22,), "右脚足部点 3"),
    RegionSpec("面部区域", tuple(range(23, 91)), "COCO-WholeBody 面部 68 点"),
    RegionSpec("左手区域", tuple(idx for idx in range(91, 112) if idx not in {95, 99, 103, 107, 111}), "左手非指尖点"),
    RegionSpec("左拇指指尖区域", (95,), "左手 thumb tip"),
    RegionSpec("左食指指尖区域", (99,), "左手 index fingertip"),
    RegionSpec("左中指指尖区域", (103,), "左手 middle fingertip"),
    RegionSpec("左无名指指尖区域", (107,), "左手 ring fingertip"),
    RegionSpec("左小指指尖区域", (111,), "左手 little fingertip"),
    RegionSpec("右手区域", tuple(idx for idx in range(112, RTMW_133_NUM_JOINTS) if idx not in {116, 120, 124, 128, 132}), "右手非指尖点"),
    RegionSpec("右拇指指尖区域", (116,), "右手 thumb tip"),
    RegionSpec("右食指指尖区域", (120,), "右手 index fingertip"),
    RegionSpec("右中指指尖区域", (124,), "右手 middle fingertip"),
    RegionSpec("右无名指指尖区域", (128,), "右手 ring fingertip"),
    RegionSpec("右小指指尖区域", (132,), "右手 little fingertip"),
)


def _hand_edges(base: int) -> tuple[tuple[int, int], ...]:
    """生成单只手 21 点拓扑边。

    参数:
        base: 该手在 133 点全身骨架中的起始下标。

    返回:
        以全身 0 基编号表示的无向边列表。
    """
    local_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    )
    return tuple((base + src, base + dst) for src, dst in local_edges)


def _joint_edges() -> tuple[tuple[int, int], ...]:
    """生成 RTMW-133 全身关节图边。

    返回:
        包含身体、足部、左右手和面部链式边的无向边列表。
    """
    body_edges = (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 17),
        (15, 18),
        (15, 19),
        (17, 18),
        (18, 19),
        (16, 20),
        (16, 21),
        (16, 22),
        (20, 21),
        (21, 22),
    )
    face_edges = tuple((idx, idx + 1) for idx in range(23, 90))
    return body_edges + _hand_edges(91) + _hand_edges(112) + face_edges


RTMW_133_LAYOUT = SkeletonLayoutSpec(
    name=RTMW_133_LAYOUT_NAME,
    num_joints=RTMW_133_NUM_JOINTS,
    region_specs=RTMW_133_REGION_SPECS,
    joint_edges=_joint_edges(),
    torso_indices=(5, 6, 11, 12),
    fallback_num_regions=RTMW_133_NUM_REGIONS,
    default_joint_graph_scope="self_only",
)


def register() -> None:
    """向 HAPM 通用注册表注入 RTMW-133 布局。"""
    register_skeleton_layout(RTMW_133_LAYOUT)
