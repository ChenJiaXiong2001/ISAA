"""骨架序列预处理与轻量增强。

这里实现骨架输入的坐标中心化、速度流、置信度通道构建和训练增强。
模型内部统一使用 C x T x N 或 C x T x N x M，二维默认 C=5:
x, y, dx, dy, score；三维默认 C=7: x, y, z, dx, dy, dz, score。
注册布局提供躯干参考点时使用尺度归一化，否则退回中心点平移归一化。
"""

from __future__ import annotations

import math

import torch

from isaa.skeleton_layout import GENERIC_LAYOUT, get_skeleton_layout


def torso_reference_indices(layout: str = GENERIC_LAYOUT) -> tuple[int, int, int, int] | None:
    """返回躯干尺度归一化需要的四个参考关节编号。

    参数:
        layout: 关键点布局名称。

    返回:
        left_shoulder、right_shoulder、left_hip、right_hip 四个下标；
        布局未注册或未提供参考点时返回 None。

    注意:
        具体参考点由外部骨架布局预设提供，HAPM 核心不硬编码。
    """
    layout_spec = get_skeleton_layout(layout)
    if layout_spec is None:
        return None
    return layout_spec.torso_indices


def normalize_by_center(x: torch.Tensor, center_index: int = 0, coordinate_dims: int = 2) -> torch.Tensor:
    """以指定中心关节为原点做坐标归一化。

    参数:
        x: C x T x N 或 C x T x N x M 的骨架张量。
        center_index: 用作原点的关节下标。

    返回:
        前 coordinate_dims 个坐标通道平移后的张量，形状不变。

    适用场景:
        未知布局或没有可靠躯干参考点时使用该通用兜底方法。
    """
    # 技术备注：只移动坐标通道，避免把 score 或其它通道当坐标一起平移。
    coordinate_dims = min(max(1, int(coordinate_dims)), x.size(0))
    if x.size(0) < coordinate_dims:
        return x
    if center_index < 0 or center_index >= x.size(2):
        center_index = 0
    out = x.clone()
    center = out[:coordinate_dims, :, center_index : center_index + 1]
    out[:coordinate_dims] = out[:coordinate_dims] - center
    return out


def normalize_by_torso(
    x: torch.Tensor,
    layout: str = GENERIC_LAYOUT,
    eps: float = 1e-6,
    coordinate_dims: int = 2,
) -> torch.Tensor:
    """使用注册布局的肩髋中心和对角肩髋距离归一化坐标。

    参数:
        x: 至少包含 coordinate_dims 个坐标通道的骨架张量。
        layout: 关键点布局。注册布局可提供肩/髋语义点。
        eps: 最小尺度，防止除零。

    返回:
        坐标通道按躯干尺度归一化后的张量，其他通道保持原样。

    兜底行为:
        未注册布局、未提供参考点、关节数不足或通道数不足时，不强行使用肩髋点，
        而是退回 normalize_by_center。
    """
    coordinate_dims = min(max(1, int(coordinate_dims)), x.size(0))
    if x.size(0) < coordinate_dims:
        return x

    refs = torso_reference_indices(layout)
    if refs is None or max(refs) >= x.size(2):
        return normalize_by_center(x, center_index=0, coordinate_dims=coordinate_dims)

    left_shoulder, right_shoulder, left_hip, right_hip = refs
    out = x.clone()
    coords = out[:coordinate_dims]
    torso_index = torch.tensor([left_shoulder, right_shoulder, left_hip, right_hip], device=x.device)
    center = coords.index_select(dim=2, index=torso_index).mean(dim=2, keepdim=True)

    diag_left = coords[:, :, left_shoulder, ...] - coords[:, :, right_hip, ...]
    diag_right = coords[:, :, right_shoulder, ...] - coords[:, :, left_hip, ...]
    scale = torch.linalg.vector_norm(diag_left, dim=0) + torch.linalg.vector_norm(diag_right, dim=0)
    scale = scale.clamp_min(eps)
    scale = scale.unsqueeze(0).unsqueeze(2)

    out[:coordinate_dims] = (coords - center) / scale
    return out


def temporal_crop_or_pad(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """将不同长度的骨架序列裁剪或零填充到统一时间长度。

    参数:
        x: C x T x N 或 C x T x N x M 的单样本骨架。
        window_size: 目标时间长度。

    返回:
        时间维长度等于 window_size 的张量。

    注意:
        这是旧兼容函数，不返回 mask，短序列会补零。
        新数据路径优先使用 temporal_crop_or_pad_with_mask。
    """
    # 技术备注：文档固定 T=64；这里使用确定性裁剪/填充保证运行验证可复现。
    frames = x.size(1)
    if frames == window_size:
        return x
    if frames > window_size:
        return x[:, :window_size, ...]

    pad_shape = (x.size(0), window_size - frames, *x.shape[2:])
    pad = x.new_zeros(pad_shape)
    return torch.cat([x, pad], dim=1)


def temporal_crop_or_pad_with_mask(
    x: torch.Tensor,
    window_size: int,
    *,
    random_start: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """裁剪/补齐序列，并返回真实帧 mask。

    参数:
        x: C x T x N 或 C x T x N x M 的单样本骨架。
        window_size: 目标时间长度。

    返回:
        padded_x: 时间维长度等于 window_size。
        mask: T 维 bool 张量，True 表示原始有效帧，False 表示补齐帧。

    补齐策略:
        短序列使用最后一帧复制，而不是补零，降低 padding 对 BN/TCN 的扰动。
        同时返回 mask，让模型在注意力和 P_rt 中屏蔽补齐帧。
    """
    frames = x.size(1)
    valid_frames = min(frames, window_size)
    mask = torch.zeros(window_size, dtype=torch.bool, device=x.device)
    mask[:valid_frames] = True

    if frames == window_size:
        return x, mask
    if frames > window_size:
        if random_start:
            max_start = frames - window_size
            start = int(torch.randint(0, max_start + 1, (), device=x.device).item())
        else:
            start = 0
        return x[:, start : start + window_size, ...], mask

    if frames <= 0:
        raise ValueError("骨架序列至少需要 1 帧")
    repeat_shape = (x.size(0), window_size - frames, *x.shape[2:])
    pad = x[:, -1:, ...].expand(repeat_shape).clone()
    return torch.cat([x, pad], dim=1), mask


def build_motion_stream(x: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
    """构造帧间差分运动特征，并保持时间长度不变。

    参数:
        x: 坐标张量，通常为 2 x T x N 或 B x 2 x T x N。
        time_dim: 时间维所在下标。

    返回:
        与 x 同形状的差分张量。首帧运动固定为 0，其余帧为当前帧减前一帧。
    """
    # 技术备注：对应文档中的 dx、dy，首帧速度固定为 0。
    motion = torch.zeros_like(x)
    current = [slice(None)] * x.ndim
    previous = [slice(None)] * x.ndim
    target = [slice(None)] * x.ndim
    current[time_dim] = slice(1, None)
    previous[time_dim] = slice(None, -1)
    target[time_dim] = slice(1, None)
    motion[tuple(target)] = x[tuple(current)] - x[tuple(previous)]
    return motion


def build_skeleton_feature_channels(
    x: torch.Tensor,
    layout: str = GENERIC_LAYOUT,
    *,
    coordinate_dims: int = 2,
    score_index: int | None = None,
) -> torch.Tensor:
    """将坐标/置信度输入整理成坐标、速度和 score 通道。

    参数:
        x: C x T x N 或 C x T x N x M。前 coordinate_dims 个通道必须是坐标。
        layout: 关键点布局，用于决定是否使用注册的躯干尺度归一化。
        coordinate_dims: 坐标维度；二维为 2，三维为 3。
        score_index: 原始输入中 score 的通道下标；None 时按常见格式推断。

    返回:
        2*coordinate_dims+1 x T x N 或 2*coordinate_dims+1 x T x N x M 的特征张量。

    通道规则:
        二维输出为 x,y,dx,dy,score；
        三维输出为 x,y,z,dx,dy,dz,score；
        score_index 显式提供时优先使用；
        原始通道只有 x/y 时，score 全 1。
    """
    coordinate_dims = int(coordinate_dims)
    if coordinate_dims not in {2, 3}:
        raise ValueError(f"HAPM 当前只支持 2D 或 3D 坐标，coordinate_dims={coordinate_dims}")
    if x.size(0) < coordinate_dims:
        raise ValueError(f"HAPM 输入至少需要 {coordinate_dims} 个坐标通道")

    x = normalize_by_torso(x, layout=layout, coordinate_dims=coordinate_dims)
    coords = x[:coordinate_dims]
    motion = build_motion_stream(coords)
    if score_index is not None and 0 <= int(score_index) < x.size(0):
        score = x[int(score_index) : int(score_index) + 1].clamp(0.0, 1.0)
    elif coordinate_dims == 3 and x.size(0) >= 7:
        score = x[6:7].clamp(0.0, 1.0)
    elif coordinate_dims == 3 and x.size(0) >= 4:
        score = x[3:4].clamp(0.0, 1.0)
    elif coordinate_dims == 2 and x.size(0) >= 5:
        score = x[4:5].clamp(0.0, 1.0)
    elif coordinate_dims == 2 and x.size(0) >= 3:
        score = x[2:3].clamp(0.0, 1.0)
    else:
        score = x.new_ones(1, *x.shape[1:])
    return torch.cat([coords, motion, score], dim=0)


def random_rotation_matrix(
    coordinate_dims: int,
    max_degrees: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a random 2D/3D rotation matrix for skeleton augmentation."""
    if max_degrees <= 0.0:
        return torch.eye(coordinate_dims, device=device, dtype=dtype)
    max_radians = float(max_degrees) * math.pi / 180.0
    if coordinate_dims == 2:
        angle = (torch.rand((), device=device, dtype=dtype) * 2.0 - 1.0) * max_radians
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        return torch.stack(
            (
                torch.stack((cos_a, -sin_a)),
                torch.stack((sin_a, cos_a)),
            )
        )
    if coordinate_dims == 3:
        angles = (torch.rand(3, device=device, dtype=dtype) * 2.0 - 1.0) * max_radians
        ax, ay, az = angles.unbind()
        cx, sx = torch.cos(ax), torch.sin(ax)
        cy, sy = torch.cos(ay), torch.sin(ay)
        cz, sz = torch.cos(az), torch.sin(az)
        rx = torch.stack(
            (
                torch.stack((torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx))),
                torch.stack((torch.zeros_like(cx), cx, -sx)),
                torch.stack((torch.zeros_like(cx), sx, cx)),
            )
        )
        ry = torch.stack(
            (
                torch.stack((cy, torch.zeros_like(cy), sy)),
                torch.stack((torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy))),
                torch.stack((-sy, torch.zeros_like(cy), cy)),
            )
        )
        rz = torch.stack(
            (
                torch.stack((cz, -sz, torch.zeros_like(cz))),
                torch.stack((sz, cz, torch.zeros_like(cz))),
                torch.stack((torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz))),
            )
        )
        return rz @ ry @ rx
    raise ValueError(f"random_rotation_matrix only supports 2D/3D, got {coordinate_dims}")


def apply_skeleton_augmentation(
    x: torch.Tensor,
    *,
    coordinate_dims: int,
    rotation_degrees: float = 0.0,
    scale_range: tuple[float, float] = (1.0, 1.0),
    jitter_std: float = 0.0,
) -> torch.Tensor:
    """Apply lightweight coordinate augmentation to one skeleton sample."""
    coordinate_dims = int(coordinate_dims)
    if coordinate_dims not in {2, 3} or x.size(0) < coordinate_dims:
        return x
    out = x.clone()
    coords = out[:coordinate_dims]
    if rotation_degrees > 0.0:
        rotation = random_rotation_matrix(
            coordinate_dims,
            float(rotation_degrees),
            device=x.device,
            dtype=x.dtype,
        )
        coords = torch.einsum("dc,ct...->dt...", rotation, coords)
    low, high = scale_range
    if high > 0.0 and low > 0.0 and (high != 1.0 or low != 1.0):
        scale = torch.empty((), device=x.device, dtype=x.dtype).uniform_(float(low), float(high))
        coords = coords * scale
    if jitter_std > 0.0:
        coords = coords + torch.randn_like(coords) * float(jitter_std)
    out[:coordinate_dims] = coords
    return out


def light_coordinate_jitter(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """轻量坐标扰动，用于参与模式一致性约束的增强视图。

    参数:
        x: 任意形状骨架张量。
        sigma: 高斯噪声标准差。<=0 时不做扰动。

    返回:
        加噪后的张量。当前训练循环尚未使用该增强函数。
    """
    # 技术备注：当前仅提供增强算子，默认可运行链路不强制启用 L_cons。
    if sigma <= 0:
        return x
    return x + torch.randn_like(x) * sigma
