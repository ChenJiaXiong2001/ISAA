"""骨架布局注册表。

该模块定义通用骨架布局协议，并在需要时自动注册项目内置布局。
具体布局的区域规划固定在 ``isaa.layouts`` 预设中，模型按布局名称直接消费。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


GENERIC_LAYOUT = "generic"


@dataclass(frozen=True)
class RegionSpec:
    """输入前固定的大区域定义。

    字段:
        name: 区域名称，例如“左手区域”或其它布局自定义名称。
        joint_indices: 该区域包含的 0 基关节编号。
        description: 面向人的说明，用于文档和调试。
    """

    name: str
    joint_indices: tuple[int, ...]
    description: str = ""


@dataclass(frozen=True)
class SkeletonLayoutSpec:
    """可注册的骨架布局说明。

    字段:
        name: 布局名称，必须与配置中的 data.skeleton_layout 对齐。
        num_joints: 该布局的标准关节点数量；None 表示不做数量约束。
        region_specs: 可选语义区域划分；为空时 HAPM 会使用 generic 区域。
        joint_edges: 可选关节图边，按 0 基编号给出无向边。
        torso_indices: 可选躯干参考点，顺序为左肩、右肩、左髋、右髋。
        fallback_num_regions: 没有 region_specs 时 generic 区域的数量建议。
        default_joint_graph_scope: 模型代码按布局使用的默认关节图范围。
            self_only 表示只保留自连接，跨节点关系交给可学习邻接矩阵。
    """

    name: str
    num_joints: int | None = None
    region_specs: tuple[RegionSpec, ...] = ()
    joint_edges: tuple[tuple[int, int], ...] = ()
    torso_indices: tuple[int, int, int, int] | None = None
    fallback_num_regions: int | None = None
    default_joint_graph_scope: str = "self_only"


_LAYOUT_REGISTRY: dict[str, SkeletonLayoutSpec] = {}


def register_skeleton_layout(spec: SkeletonLayoutSpec) -> None:
    """注册一个外部骨架布局。

    参数:
        spec: 骨架布局说明。

    行为:
        同名布局会被覆盖，方便入口重复调用注册函数而不产生副作用。
        具体布局内容不在这里解释，HAPM 核心只按通用字段消费。
    """
    if not spec.name:
        raise ValueError("骨架布局名称不能为空")
    if spec.name == GENERIC_LAYOUT:
        raise ValueError("generic 是内置兜底布局名称，不允许外部覆盖")
    _LAYOUT_REGISTRY[spec.name] = spec


def get_skeleton_layout(name: str | None, *, auto_register: bool = True) -> SkeletonLayoutSpec | None:
    """按名称读取已注册布局。

    参数:
        name: 配置中的布局名称。
        auto_register: 未找到时是否自动注册项目内置布局。

    返回:
        找到时返回 SkeletonLayoutSpec；generic、空名称或未注册名称返回 None。
    """
    if not name or name == GENERIC_LAYOUT:
        return None
    if auto_register and name not in _LAYOUT_REGISTRY:
        auto_register_skeleton_layouts()
    return _LAYOUT_REGISTRY.get(name)


def has_skeleton_layout(name: str | None, *, auto_register: bool = True) -> bool:
    """判断布局名称是否可用。

    参数:
        name: 配置中的布局名称。
        auto_register: 未找到时是否自动注册项目内置布局。

    返回:
        generic 永远可用；其它名称必须先由入口注册。
    """
    if not name or name == GENERIC_LAYOUT:
        return True
    if auto_register and name not in _LAYOUT_REGISTRY:
        auto_register_skeleton_layouts()
    return name in _LAYOUT_REGISTRY


def registered_skeleton_layouts() -> tuple[str, ...]:
    """返回当前已注册的外部骨架布局名称。"""
    return tuple(sorted(_LAYOUT_REGISTRY))


def auto_register_skeleton_layouts(package_name: str = "isaa.layouts") -> None:
    """按项目约定可选注册外部骨架布局预设。

    参数:
        package_name: 预设包名称。默认读取本项目的 isaa.layouts。

    行为:
        如果该包存在且暴露 register_skeleton_presets()，则调用它完成注册。
        如果包不存在，则静默跳过，让后续配置校验给出明确的未知布局错误。

    设计边界:
        这里不导入任何具体布局名称，也不依赖专用骨架框架代码；
        只是提供“项目外部预设包”的通用发现钩子。
    """
    try:
        module = importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name == package_name:
            return
        raise
    registrar = getattr(module, "register_skeleton_presets", None)
    if callable(registrar):
        registrar()


def resolve_layout_name(data_cfg: dict) -> str:
    """从数据配置中解析骨架布局名称。

    参数:
        data_cfg: config["data"]。

    返回:
        优先使用 data.skeleton_layout；兼容旧字段 data.region_layout。
    """
    return str(data_cfg.get("skeleton_layout", data_cfg.get("region_layout", GENERIC_LAYOUT)))
