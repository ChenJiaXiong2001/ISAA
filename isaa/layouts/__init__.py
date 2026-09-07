"""外部骨架布局预设入口。

这里集中注册项目随带的具体骨架框架。模型通过布局注册表读取这些预设，
入口脚本负责调用 register_skeleton_presets() 完成注入。
"""

from __future__ import annotations

from isaa.layouts import rtmw_133


def register_skeleton_presets() -> None:
    """注册当前研究使用的 RTMW-133 骨架布局。"""
    rtmw_133.register()
