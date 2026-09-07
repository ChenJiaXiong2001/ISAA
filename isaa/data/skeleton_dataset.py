"""骨架动作数据集骨架实现。

文档小结对齐：
- 这里完成骨架输入构建和 DataLoader。
- 支持 dummy、批量 npz，以及按样本流式读取的 RTMW ZIP。
- 已注册布局可提供语义增强；generic 布局按配置节点数通用处理。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from isaa.data.transforms import (
    apply_skeleton_augmentation,
    build_motion_stream,
    build_skeleton_feature_channels,
    normalize_by_center,
    random_rotation_matrix,
)
from isaa.skeleton_layout import get_skeleton_layout, resolve_layout_name
from isaa.spec import (
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_NUM_JOINTS,
    DEFAULT_REGION_LAYOUT,
    DEFAULT_WINDOW_SIZE,
    SUPPORTED_INPUT_ORDERS,
)
from isaa.utils.runtime import resolve_dataloader_settings, resolve_dataset_runtime_settings


def _require_positive_int(name: str, value: object) -> int:
    """解析公共数据入口的正整数参数。

    参数:
        name: 参数名，用于错误提示。
        value: 待解析值。

    返回:
        正整数。

    抛错:
        布尔值、非整数小数、非数字或小于等于 0 时抛 ValueError。
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是正整数，当前为 {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} 必须是正整数，当前为 {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正整数，当前为 {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} 必须是正整数，当前为 {value!r}")
    return parsed


class DummySkeletonDataset(Dataset):
    """用于调通训练链路的虚拟骨架数据集。

    输入来源：
        不读取文件，按随机种子在内存中生成固定形状骨架。

    输出格式：
        __getitem__ 返回 skeleton、label、valid_frame_mask 三元组。
        skeleton 形状为 C x T x N；label 是类别编号；valid_frame_mask 形状为 T。

    使用场景：
        只用于 smoke test，检查模型、损失和训练循环是否可运行。
        不代表真实数据分布，也不能用于评估模型效果。
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        input_channels: int = DEFAULT_INPUT_CHANNELS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        num_joints: int = DEFAULT_NUM_JOINTS,
        seed: int = 42,
    ) -> None:
        """生成固定随机种子的 dummy 数据。

        参数:
            num_samples: 样本数量。
            num_classes: 分类类别数，随机标签范围为 [0, num_classes)。
            input_channels: 输出骨架通道数，默认 5 通道 HAPM 特征。
            window_size: 时间窗口长度 T。
            num_joints: 关节点数量 N。
            seed: 随机种子，保证 smoke test 可复现。

        细节:
            当 input_channels 为 5 时，显式构造 x/y、dx/dy、score。
            其他通道数只生成随机张量，用于测试模型维度兼容性。
        """
        # 技术备注：虚拟数据按 x/y/dx/dy/score 生成，只检查代码是否能跑通，不代表模型有效性。
        num_samples = _require_positive_int("num_samples", num_samples)
        num_classes = _require_positive_int("num_classes", num_classes)
        input_channels = _require_positive_int("input_channels", input_channels)
        window_size = _require_positive_int("window_size", window_size)
        num_joints = _require_positive_int("num_joints", num_joints)
        generator = torch.Generator().manual_seed(seed)
        if input_channels == 5:
            coords = torch.randn(num_samples, 2, window_size, num_joints, generator=generator) * 0.5
            motion = build_motion_stream(coords, time_dim=2)
            score = torch.rand(num_samples, 1, window_size, num_joints, generator=generator).mul(0.5).add(0.5)
            self.skeletons = torch.cat([coords, motion, score], dim=1)
        else:
            self.skeletons = torch.randn(
                num_samples,
                input_channels,
                window_size,
                num_joints,
                generator=generator,
            )
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
        self.valid_frame_mask = torch.ones(window_size, dtype=torch.bool)

    def __len__(self) -> int:
        """返回 dummy 样本数。

        该值来自 labels 的长度，和初始化传入的 num_samples 一致。
        DataLoader 会用它决定每个 epoch 的迭代次数。
        """
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """读取单个 dummy 样本。

        参数:
            index: 样本下标。

        返回:
            skeleton: C x T x N 的骨架张量。
            label: 标量 long tensor。
            valid_frame_mask: T 维 bool mask，dummy 数据全部为 True。
        """
        return self.skeletons[index], self.labels[index], self.valid_frame_mask


class SkeletonTensorDataset(Dataset):
    """从 npz 文件读取骨架张量的最小数据集实现。

    npz 契约:
        必须包含 x 和 y 两个数组，可选 valid_frame_mask。
        x 是 4D 或 5D 批量骨架张量；y 是一维类别标签。

    维度契约:
        由 input_order 显式描述 x 的维度顺序，避免根据形状猜测。
        支持 BCTN、BCTNM、BTNC、BTMNC、BTNMC、BMCTN。

    通道语义:
        input_format=raw_xy_score 时，会将坐标/置信度整理成 x,y,dx,dy,score。
        input_format=precomputed_5ch 时，认为输入已经是 5 通道特征，不再二次处理。

    输出格式:
        __getitem__ 返回 skeleton、label、valid_frame_mask 三元组。
        skeleton 为 C x T x N 或 C x T x N x M；mask 为 T 维 bool 张量。
    """

    def __init__(
        self,
        data_path: str | Path,
        window_size: int,
        num_joints: int,
        layout: str = DEFAULT_REGION_LAYOUT,
        input_channels: int = DEFAULT_INPUT_CHANNELS,
        center_index: int = 0,
        num_classes: int | None = None,
        input_format: str = "raw_xy_score",
        input_order: str = "BCTN",
        cache_processed: bool = False,
        share_memory: bool = False,
        augment: bool = False,
        augmentation_config: dict[str, Any] | None = None,
        deterministic_temporal_crop: str = "start",
    ) -> None:
        """加载 npz 骨架数据并转换为模型输入格式。

        参数:
            data_path: npz 文件路径，文件内必须包含 x 和 y。
            window_size: 训练使用的固定时间窗口长度。
            num_joints: 配置声明的关节点数量，用于校验输入 N 维。
            layout: 骨架布局名称；注册布局可使用语义区域配置。
            input_channels: 模型期望通道数，目前真实数据链路固定输出 5 通道。
            center_index: 坐标中心化使用的关节点下标。
            num_classes: 可选类别数，用于校验标签是否越界。
            input_format: 原始 x/y/score 或已预计算 5 通道特征。
            input_order: x 数组维度顺序，必须显式描述以避免形状猜测。

        细节:
            该初始化会完成文件存在性、字段、维度、标签、类别范围等硬校验。
            之后执行维度转置、中心化、通道构造、时间裁剪/填充和 mask 生成。
        """
        # 技术备注：真实骨架数据建议提前离线清洗成 npz/pkl，训练时只做轻量变换。
        window_size = _require_positive_int("window_size", window_size)
        num_joints = _require_positive_int("num_joints", num_joints)
        input_channels = _require_positive_int("input_channels", input_channels)
        if num_classes is not None:
            num_classes = _require_positive_int("num_classes", num_classes)
        input_order = input_order.upper()
        if input_order not in SUPPORTED_INPUT_ORDERS:
            raise ValueError(f"暂不支持的 input_order: {input_order}")
        if input_format == "precomputed_5ch" and input_channels != 5:
            raise ValueError("input_format=precomputed_5ch 要求 input_channels=5")
        if input_format == "precomputed_7ch" and input_channels != 7:
            raise ValueError("input_format=precomputed_7ch 要求 input_channels=7")
        if input_format == "raw_xyz_score" and input_channels != 7:
            raise ValueError("input_format=raw_xyz_score 要求 input_channels=7")
        if str(data_path).strip() == "":
            raise ValueError("dataset=npz 时必须配置 data_path 或对应 split_path")
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"数据文件不存在或不是文件: {path}")

        with np.load(path) as data:
            missing = [key for key in ("x", "y") if key not in data.files]
            if missing:
                raise ValueError(f"npz 数据缺少字段: {missing}; 需要包含 x 和 y")

            # 技术备注：np.load 返回的 NpzFile 在 Windows 上会持有文件句柄，
            # 这里复制出数组后立即关闭归档，避免异常校验路径泄漏句柄。
            x_array = np.array(data["x"])
            y_array = np.array(data["y"])
            mask_array = np.array(data["valid_frame_mask"]) if "valid_frame_mask" in data.files else None
        if input_format not in {"raw_xy_score", "raw_xyz_score", "precomputed_5ch", "precomputed_7ch"}:
            raise ValueError(f"暂不支持的 input_format: {input_format}")
        if not input_order.startswith("B"):
            raise ValueError(f"input_order 必须以 B 开头，当前为 {input_order!r}")
        if len(input_order) != x_array.ndim:
            raise ValueError(f"input_order={input_order} 与 npz x 形状 {x_array.shape} 维度数不一致")
        if x_array.ndim not in {4, 5}:
            raise ValueError(f"npz x 必须是 4D 或 5D 批量骨架张量，当前形状: {x_array.shape}")
        channel_count = int(x_array.shape[input_order.index("C")])
        if input_format == "raw_xy_score" and channel_count < 3:
            raise ValueError(f"raw_xy_score 需要 x/y/score 至少 3 通道，当前为 {channel_count}")
        if input_format == "raw_xyz_score" and channel_count < 4:
            raise ValueError(
                f"raw_xyz_score 需要 x/y/z/score 至少 4 通道，当前为 {channel_count}；"
                "请用 --coordinate-mode camera_xyz 重新生成 npz"
            )
        if any(dim <= 0 for dim in x_array.shape):
            raise ValueError(f"npz x 各维度必须大于 0，当前形状: {x_array.shape}")
        if not np.issubdtype(x_array.dtype, np.number):
            raise ValueError(f"npz x 必须是数值骨架张量，当前 dtype: {x_array.dtype}")
        if not np.isfinite(x_array).all():
            raise ValueError("npz x 包含 NaN 或 Inf 骨架值")
        if y_array.ndim != 1:
            raise ValueError(f"npz y 必须是一维标签数组，当前形状: {y_array.shape}")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError(f"npz x/y 样本数不一致: x={x_array.shape[0]} y={y_array.shape[0]}")
        if y_array.size == 0:
            raise ValueError("npz 数据不能为空")
        if not np.issubdtype(y_array.dtype, np.number):
            raise ValueError(f"npz y 必须是数值标签，当前 dtype: {y_array.dtype}")
        if not np.isfinite(y_array).all():
            raise ValueError("npz y 包含 NaN 或 Inf 标签")
        if not np.equal(y_array, np.floor(y_array)).all():
            raise ValueError("npz y 必须是整数类别标签")
        if mask_array is not None:
            if mask_array.ndim != 2:
                raise ValueError(f"npz valid_frame_mask 必须是 B x T，当前形状: {mask_array.shape}")
            if mask_array.shape[0] != x_array.shape[0]:
                raise ValueError(
                    "npz valid_frame_mask 与 x 样本数不一致: "
                    f"mask={mask_array.shape[0]} x={x_array.shape[0]}"
                )

        label_min = int(y_array.min())
        label_max = int(y_array.max())
        if label_min < 0:
            raise ValueError(f"标签不能为负数，当前最小标签: {label_min}")
        if num_classes is not None and label_max >= int(num_classes):
            raise ValueError(f"标签越界: 最大标签 {label_max} >= num_classes {int(num_classes)}")

        skeletons = torch.as_tensor(x_array, dtype=torch.float32)
        self.skeletons = _to_channel_first_batch(skeletons, input_order=input_order).contiguous()
        self.labels = torch.as_tensor(y_array, dtype=torch.long)
        self.valid_frame_masks = None if mask_array is None else torch.as_tensor(mask_array, dtype=torch.bool)
        self.window_size = window_size
        self.num_joints = num_joints
        self.layout = layout
        self.input_channels = input_channels
        self.center_index = center_index
        self.input_format = input_format
        self.coordinate_dims = _coordinate_dims_for_input_format(input_format, input_channels)
        self.score_index = _score_index_for_input_format(input_format, input_channels)
        self.augment = bool(augment)
        self.augmentation_config = augmentation_config or {}
        self.deterministic_temporal_crop = str(deterministic_temporal_crop).strip().lower()
        if self.deterministic_temporal_crop not in {"start", "center", "end"}:
            raise ValueError(
                "deterministic_temporal_crop 只支持 start/center/end，"
                f"当前为 {deterministic_temporal_crop!r}"
            )
        self.sample_order = "CTNM" if self.skeletons.dim() == 5 else "CTN"
        self.preprocessed = False
        self.preprocessed_base = False

        if cache_processed:
            if self.input_format in {"raw_xy_score", "raw_xyz_score"}:
                self._cache_augmented_base_samples()
            elif not self.augment:
                self._cache_processed_samples()
        if share_memory:
            self.share_memory_()

    def __len__(self) -> int:
        """返回 npz 中样本数量。

        该值等于 y 的长度，并在初始化时已经检查与 x.shape[0] 一致。
        """
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """读取并规范化单个 npz 样本。

        处理步骤:
            1. 按 input_order 转为 C x T x N 或 C x T x N x M。
            2. 将节点数对齐到配置中的 num_joints。
            3. 将时间长度裁剪/补齐到 window_size，并生成 valid_frame_mask。
            4. 根据 input_format 决定是否构造 5 通道特征。

        返回:
            skeleton: 模型可直接接收的单样本骨架张量。
            label: long 类型类别标签。
            valid_frame_mask: True 表示原始有效帧，False 表示补齐帧。
        """
        if self.preprocessed:
            return self.skeletons[index], self.labels[index], self.valid_frame_masks[index]
        if self.preprocessed_base:
            x, valid_frame_mask = self._prepare_cached_base_sample(index)
            return x, self.labels[index], valid_frame_mask
        x, valid_frame_mask = self._prepare_sample(index)
        return x, self.labels[index], valid_frame_mask

    def _prepare_cached_base_sample(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop cached full-length features and apply cheap feature-space augmentation."""
        x = self.skeletons[index]
        file_mask = None if self.valid_frame_masks is None else self.valid_frame_masks[index].to(device=x.device)
        random_crop = self.augment and bool(self.augmentation_config.get("random_temporal_crop", False))
        x, valid_frame_mask = _temporal_crop_or_pad_sample_with_mask(
            x,
            file_mask,
            self.window_size,
            random_start=random_crop,
            deterministic_crop=self.deterministic_temporal_crop,
        )
        if self.augment:
            x = _apply_feature_space_augmentation(
                x,
                coordinate_dims=self.coordinate_dims,
                rotation_degrees=float(self.augmentation_config.get("rotation_degrees", 0.0)),
                scale_range=_scale_range_from_config(self.augmentation_config.get("scale_range", (1.0, 1.0))),
                jitter_std=float(self.augmentation_config.get("coordinate_jitter_std", 0.0)),
            )
        return x.contiguous(), valid_frame_mask.contiguous()

    def _prepare_sample(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """将原始单样本转换为模型输入张量和有效帧 mask。"""
        x = _to_channel_first_sample(self.skeletons[index], input_order=self.sample_order)
        if self.layout != DEFAULT_REGION_LAYOUT and x.size(2) != self.num_joints:
            raise ValueError(
                f"骨架布局 {self.layout!r} 要求 {self.num_joints} 个关节，"
                f"样本 {index} 为 {x.size(2)} 个；请先离线转换到正确节点顺序"
            )
        x = _align_joint_count(x, expected_num_joints=self.num_joints)
        random_crop = self.augment and bool(self.augmentation_config.get("random_temporal_crop", False))
        file_mask = None
        if self.valid_frame_masks is not None:
            file_mask = self.valid_frame_masks[index].to(device=x.device)
            if file_mask.numel() != x.size(1) and not (
                x.size(1) == self.window_size and file_mask.numel() == self.window_size
            ):
                raise ValueError(
                    "npz valid_frame_mask 时间长度必须匹配原始或裁剪后时间长度，"
                    f"当前 mask={file_mask.numel()} frames={x.size(1)} window={self.window_size}"
                )
        x, valid_frame_mask = _temporal_crop_or_pad_sample_with_mask(
            x,
            file_mask,
            self.window_size,
            random_start=random_crop,
            deterministic_crop=self.deterministic_temporal_crop,
        )
        if self.augment and self.input_format in {"raw_xy_score", "raw_xyz_score"}:
            x = apply_skeleton_augmentation(
                x,
                coordinate_dims=self.coordinate_dims,
                rotation_degrees=float(self.augmentation_config.get("rotation_degrees", 0.0)),
                scale_range=_scale_range_from_config(self.augmentation_config.get("scale_range", (1.0, 1.0))),
                jitter_std=float(self.augmentation_config.get("coordinate_jitter_std", 0.0)),
            )
        if self.input_channels in {5, 7}:
            if self.input_format in {"raw_xy_score", "raw_xyz_score"}:
                x = build_skeleton_feature_channels(
                    x,
                    layout=self.layout,
                    coordinate_dims=self.coordinate_dims,
                    score_index=self.score_index,
                )
            elif self.input_format == "precomputed_5ch":
                if x.size(0) != 5:
                    raise ValueError(f"precomputed_5ch 要求样本通道数为 5，当前为 {x.size(0)}")
                x = x[:5]
            elif self.input_format == "precomputed_7ch":
                if x.size(0) != 7:
                    raise ValueError(f"precomputed_7ch 要求样本通道数为 7，当前为 {x.size(0)}")
                x = x[:7]
            else:
                raise ValueError(f"暂不支持的 input_format: {self.input_format}")
        else:
            x = normalize_by_center(x, center_index=self.center_index, coordinate_dims=self.coordinate_dims)
        return x.contiguous(), valid_frame_mask.contiguous()

    def _cache_processed_samples(self) -> None:
        """预先缓存处理后的训练窗口，减少每个 epoch 的 CPU 重复工作。"""
        processed_skeletons = []
        processed_masks = []
        for index in range(len(self)):
            x, valid_frame_mask = self._prepare_sample(index)
            processed_skeletons.append(x)
            processed_masks.append(valid_frame_mask)
        self.skeletons = torch.stack(processed_skeletons, dim=0).contiguous()
        self.valid_frame_masks = torch.stack(processed_masks, dim=0).contiguous()
        self.preprocessed = True

    def _cache_augmented_base_samples(self) -> None:
        """Cache full-length feature tensors while preserving random crop/augmentation at read time."""
        self.skeletons = _batch_build_skeleton_feature_channels(
            self.skeletons,
            layout=self.layout,
            coordinate_dims=self.coordinate_dims,
            score_index=self.score_index,
        ).contiguous()
        if self.valid_frame_masks is None:
            self.valid_frame_masks = torch.ones(
                self.skeletons.size(0),
                self.skeletons.size(2),
                dtype=torch.bool,
            )
        else:
            self.valid_frame_masks = self.valid_frame_masks.contiguous()
        self.input_format = f"precomputed_{self.input_channels}ch"
        self.sample_order = "CTNM" if self.skeletons.dim() == 5 else "CTN"
        self.preprocessed_base = True

    def share_memory_(self) -> None:
        """把缓存张量放入共享内存，降低 Windows 多 worker 的复制压力。"""
        for name, value in (
            ("skeletons", self.skeletons),
            ("labels", self.labels),
            ("valid_frame_masks", self.valid_frame_masks),
        ):
            if isinstance(value, torch.Tensor):
                try:
                    value.share_memory_()
                except RuntimeError as exc:
                    print(f"dataset: share_memory skipped for {name}: {exc}")


class SeededEpochSampler(Sampler[int]):
    """按 epoch 生成可复现随机顺序的训练采样器。

    续训从上一个完整 epoch 后继续，因此每个 epoch 的样本顺序固定可复现。
    """

    def __init__(self, data_source: Dataset, seed: int, epoch: int = 1) -> None:
        """保存数据集和基础随机种子。"""
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，用于生成该 epoch 的固定 shuffle 顺序。"""
        self.epoch = int(epoch)

    def __iter__(self):
        """返回当前 epoch 的随机下标序列。"""
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        """返回样本数。"""
        return len(self.data_source)


def _sample_order(input_order: str) -> str:
    """去掉 batch 维后的单样本维度顺序。

    参数:
        input_order: 带 B 的批量维度顺序，例如 BCTN。

    返回:
        不含 B 的单样本顺序，例如 CTN。

    抛错:
        如果 input_order 不以 B 开头，说明无法确定 batch 维，直接报错。
    """
    order = input_order.upper()
    if not order.startswith("B"):
        raise ValueError(f"input_order 必须以 B 开头，当前为 {input_order!r}")
    return order[1:]


def _coordinate_dims_for_input_format(input_format: str, input_channels: int) -> int:
    """Resolve whether an input sample carries 2D or 3D coordinates."""
    if input_format in {"raw_xyz_score", "precomputed_7ch"}:
        return 3
    if int(input_channels) == 7:
        return 3
    return 2


def _score_index_for_input_format(input_format: str, input_channels: int) -> int | None:
    """Return the raw score channel index for supported coordinate formats."""
    if input_format == "raw_xyz_score":
        return 3
    if input_format == "raw_xy_score":
        return 2
    if input_format == "precomputed_7ch" or int(input_channels) == 7:
        return 6
    if input_format == "precomputed_5ch" or int(input_channels) == 5:
        return 4
    return None


def _scale_range_from_config(value: object) -> tuple[float, float]:
    """Parse an augmentation scale range from YAML-friendly values."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low = float(value[0])
        high = float(value[1])
    else:
        center = float(value)
        low = 1.0 - center
        high = 1.0 + center
    if low <= 0.0 or high <= 0.0:
        raise ValueError(f"augmentation.scale_range 必须为正数范围，当前为 {value!r}")
    if low > high:
        low, high = high, low
    return low, high


def _batch_build_skeleton_feature_channels(
    x: torch.Tensor,
    layout: str,
    *,
    coordinate_dims: int,
    score_index: int | None,
) -> torch.Tensor:
    """Vectorize full-dataset raw coordinate normalization and motion construction."""
    coordinate_dims = int(coordinate_dims)
    if coordinate_dims not in {2, 3}:
        raise ValueError(f"HAPM 当前只支持 2D 或 3D 坐标，coordinate_dims={coordinate_dims}")
    if x.size(1) < coordinate_dims:
        raise ValueError(f"HAPM 输入至少需要 {coordinate_dims} 个坐标通道")

    coords = x[:, :coordinate_dims].clone()
    layout_spec = get_skeleton_layout(layout)
    refs = None if layout_spec is None else layout_spec.torso_indices
    if refs is not None and max(refs) < x.size(3):
        left_shoulder, right_shoulder, left_hip, right_hip = refs
        torso_index = torch.tensor([left_shoulder, right_shoulder, left_hip, right_hip], device=x.device)
        center = coords.index_select(dim=3, index=torso_index).mean(dim=3, keepdim=True)
        diag_left = coords[:, :, :, left_shoulder, ...] - coords[:, :, :, right_hip, ...]
        diag_right = coords[:, :, :, right_shoulder, ...] - coords[:, :, :, left_hip, ...]
        scale = torch.linalg.vector_norm(diag_left, dim=1) + torch.linalg.vector_norm(diag_right, dim=1)
        scale = scale.clamp_min(1e-6).unsqueeze(1).unsqueeze(3)
        coords = (coords - center) / scale
    else:
        center_index = 0 if x.size(3) > 0 else None
        if center_index is not None:
            center = coords[:, :, :, center_index : center_index + 1, ...]
            coords = coords - center

    motion = build_motion_stream(coords, time_dim=2)
    if score_index is not None and 0 <= int(score_index) < x.size(1):
        score = x[:, int(score_index) : int(score_index) + 1].clamp(0.0, 1.0)
    else:
        score = x.new_ones(x.size(0), 1, *x.shape[2:])
    return torch.cat([coords, motion, score], dim=1)


def _apply_feature_space_augmentation(
    x: torch.Tensor,
    *,
    coordinate_dims: int,
    rotation_degrees: float,
    scale_range: tuple[float, float],
    jitter_std: float,
) -> torch.Tensor:
    """Apply cheap augmentation to cached feature tensors and refresh motion channels."""
    coordinate_dims = int(coordinate_dims)
    if coordinate_dims not in {2, 3} or x.size(0) < coordinate_dims * 2:
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
    out[coordinate_dims : coordinate_dims * 2] = build_motion_stream(coords)
    return out


def _to_channel_first_sample(x: torch.Tensor, input_order: str) -> torch.Tensor:
    """按显式 input_order 将样本统一为 C x T x N 或 C x T x N x M。

    参数:
        x: 单个样本张量，不含 batch 维。
        input_order: 单样本维度顺序，例如 CTN、TNC、TMNC。

    返回:
        通道优先张量。单人返回 C x T x N，多人返回 C x T x N x M。

    设计原因:
        真实数据中 T、N、M、C 的数值可能碰巧等于 2/3/5，
        靠形状猜通道位置不可靠，因此必须使用显式 input_order。
    """
    order = input_order.upper()
    if x.ndim != len(order):
        raise ValueError(f"input_order={order} 与样本形状 {tuple(x.shape)} 维度数不一致")
    if order == "CTN":
        return x
    if order == "CTNM":
        return x
    if order == "TNC":
        return x.permute(2, 0, 1).contiguous()
    if order == "TMNC":
        return x.permute(3, 0, 2, 1).contiguous()
    if order == "TNMC":
        return x.permute(3, 0, 1, 2).contiguous()
    if order == "MCTN":
        return x.permute(1, 2, 3, 0).contiguous()
    raise ValueError(f"暂不支持的 input_order: B{order}")


def _to_channel_first_batch(x: torch.Tensor, input_order: str) -> torch.Tensor:
    """按显式 input_order 将批量张量预先统一为 B x C x T x N(...)."""
    order = input_order.upper()
    if x.ndim != len(order):
        raise ValueError(f"input_order={order} 与批量形状 {tuple(x.shape)} 维度数不一致")
    if order == "BCTN":
        return x
    if order == "BCTNM":
        return x
    if order == "BTNC":
        return x.permute(0, 3, 1, 2)
    if order == "BTMNC":
        return x.permute(0, 4, 1, 3, 2)
    if order == "BTNMC":
        return x.permute(0, 4, 1, 2, 3)
    if order == "BMCTN":
        return x.permute(0, 2, 3, 4, 1)
    raise ValueError(f"暂不支持的 input_order: {order}")


def _align_joint_count(x: torch.Tensor, expected_num_joints: int) -> torch.Tensor:
    """将单样本节点数对齐到配置值，少补零，多截断。

    参数:
        x: C x T x N 或 C x T x N x M 的单样本骨架。
        expected_num_joints: 配置要求的节点数。

    返回:
        节点维 N 等于 expected_num_joints 的张量。

    注意:
        补零只是为了保证形状可运行，不会生成对应置信度 mask。
        真实数据最好在离线阶段整理成正确节点数和正确节点顺序。
    """
    if x.size(2) == expected_num_joints:
        return x
    if x.size(2) > expected_num_joints:
        return x[:, :, :expected_num_joints, ...]
    pad_shape = (x.size(0), x.size(1), expected_num_joints - x.size(2), *x.shape[3:])
    return torch.cat([x, x.new_zeros(pad_shape)], dim=2)


def _temporal_crop_or_pad_mask(mask: torch.Tensor, window_size: int) -> torch.Tensor:
    """将外部提供的帧 mask 对齐到固定时间长度。"""
    mask = mask.flatten().bool()
    frames = mask.numel()
    if frames == window_size:
        return mask
    if frames > window_size:
        return mask[:window_size]
    pad = mask.new_zeros(window_size - frames)
    return torch.cat([mask, pad], dim=0)


def _sample_valid_temporal_crop_start(mask: torch.Tensor, window_size: int) -> int | None:
    """在随机裁剪时优先选择保留最多真实帧的窗口。"""
    mask = mask.flatten().bool()
    if mask.numel() < window_size:
        return None
    counts = mask.to(dtype=torch.int64)
    prefix = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)])
    window_counts = prefix[window_size:] - prefix[:-window_size]
    max_count = int(window_counts.max().item())
    if max_count <= 0:
        return None
    candidates = torch.nonzero(window_counts == max_count, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return None
    choice = torch.randint(0, candidates.numel(), (), device=candidates.device)
    return int(candidates[choice].item())


def _temporal_crop_or_pad_sample_with_mask(
    x: torch.Tensor,
    file_mask: torch.Tensor | None,
    window_size: int,
    *,
    random_start: bool,
    deterministic_crop: str = "start",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop/pad a sample and optional file mask with the same temporal offset."""
    frames = x.size(1)
    if frames <= 0:
        raise ValueError("骨架序列至少需要 1 帧")
    if file_mask is not None:
        file_mask = file_mask.flatten().bool()

    if frames >= window_size:
        if frames == window_size:
            start = 0
        elif random_start:
            valid_start = (
                _sample_valid_temporal_crop_start(file_mask, window_size)
                if file_mask is not None and file_mask.numel() == frames
                else None
            )
            if valid_start is None:
                start = int(torch.randint(0, frames - window_size + 1, (), device=x.device).item())
            else:
                start = valid_start
        else:
            crop_mode = str(deterministic_crop).strip().lower()
            if crop_mode == "center":
                start = (frames - window_size) // 2
            elif crop_mode == "end":
                start = frames - window_size
            elif crop_mode == "start":
                start = 0
            else:
                raise ValueError(f"未知 deterministic_crop: {deterministic_crop!r}")
        cropped = x[:, start : start + window_size, ...]
        mask = torch.ones(window_size, dtype=torch.bool, device=x.device)
        if file_mask is not None:
            if file_mask.numel() == frames:
                mask = mask & file_mask[start : start + window_size].to(device=x.device)
            else:
                mask = mask & _temporal_crop_or_pad_mask(file_mask.to(device=x.device), window_size)
        return cropped, mask

    repeat_shape = (x.size(0), window_size - frames, *x.shape[2:])
    pad = x[:, -1:, ...].expand(repeat_shape).clone()
    padded = torch.cat([x, pad], dim=1)
    mask = torch.zeros(window_size, dtype=torch.bool, device=x.device)
    mask[:frames] = True
    if file_mask is not None:
        mask = mask & _temporal_crop_or_pad_mask(file_mask.to(device=x.device), window_size)
    return padded, mask


def build_dataset(
    config: dict[str, Any],
    split: str = "train",
    *,
    device: torch.device | None = None,
    dataloader_settings: dict[str, Any] | None = None,
) -> Dataset:
    """根据配置创建 Dataset。

    参数:
        config: 完整运行配置。
        split: train 或 val。dummy 模式下用于切换随机种子；npz 模式下用于读取 train_path/val_path。
        device: 当前训练设备，用于自动决定 pin_memory。
        dataloader_settings: 已解析的 DataLoader 运行参数；未提供时自动解析。

    返回:
        PyTorch Dataset。每个样本为 skeleton、label、valid_frame_mask。

    当前支持:
        dummy: 随机数据，用于检查链路。
        npz: 本地骨架数组，用于 toy 或真实数据接入。
        rtmw_zip: 按样本流式读取 ZIP 中的 NTU120 RTMW-133 骨架。
    """
    # 技术备注：split 目前用于区分 dummy 随机种子；真实数据划分可在 npz 生成阶段完成。
    data_cfg = config["data"]
    dataset_name = data_cfg.get("dataset", "dummy")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = dataloader_settings or resolve_dataloader_settings(config, device=device, split=split)
    dataset_runtime = resolve_dataset_runtime_settings(
        config,
        device=device,
        split=split,
        dataloader_workers=int(settings["num_workers"]),
    )

    if dataset_name == "dummy":
        dataset = DummySkeletonDataset(
            num_samples=int(data_cfg["num_samples"]),
            num_classes=int(data_cfg["num_classes"]),
            input_channels=int(data_cfg["input_channels"]),
            window_size=int(data_cfg["window_size"]),
            num_joints=int(data_cfg["num_joints"]),
            seed=int(config.get("seed", 42)) + (0 if split == "train" else 1000),
        )
    elif dataset_name == "npz":
        split_key = f"{split}_path"
        data_path = data_cfg.get(split_key) or data_cfg.get("data_path", "")
        augmentation_cfg = data_cfg.get("augmentation", {})
        if not isinstance(augmentation_cfg, dict):
            augmentation_cfg = {}
        augment = split == "train" and bool(augmentation_cfg.get("enabled", False))
        dataset = SkeletonTensorDataset(
            data_path=data_path,
            window_size=int(data_cfg["window_size"]),
            num_joints=int(data_cfg["num_joints"]),
            layout=resolve_layout_name(data_cfg),
            input_channels=int(data_cfg.get("input_channels", DEFAULT_INPUT_CHANNELS)),
            num_classes=int(data_cfg["num_classes"]),
            input_format=data_cfg.get("input_format", "raw_xy_score"),
            input_order=data_cfg.get("input_order", "BCTN"),
            cache_processed=dataset_runtime["cache_processed"],
            share_memory=dataset_runtime["share_memory"],
            augment=augment,
            augmentation_config=augmentation_cfg,
            deterministic_temporal_crop="start" if split == "train" else "center",
        )
    elif dataset_name == "rtmw_zip":
        from isaa.data.rtmw_zip_dataset import RTMWZipDataset

        split_key = f"{split}_path"
        archive_path = (
            data_cfg.get(split_key)
            or data_cfg.get("archive_path")
            or data_cfg.get("data_path", "")
        )
        augmentation_cfg = data_cfg.get("augmentation", {})
        if not isinstance(augmentation_cfg, dict):
            augmentation_cfg = {}
        augment = split == "train" and bool(augmentation_cfg.get("enabled", False))
        max_samples = data_cfg.get(
            f"max_{split}_samples",
            data_cfg.get("max_samples_per_split", 0),
        )
        dataset = RTMWZipDataset(
            archive_path=archive_path,
            split=split,
            split_protocol=data_cfg.get("split_protocol", "xsub120"),
            window_size=int(data_cfg["window_size"]),
            num_joints=int(data_cfg["num_joints"]),
            num_classes=int(data_cfg["num_classes"]),
            layout=resolve_layout_name(data_cfg),
            max_persons=int(data_cfg.get("max_persons", 2)),
            score_normalization=data_cfg.get("score_normalization", "auto"),
            augment=augment,
            augmentation_config=augmentation_cfg,
            deterministic_temporal_crop="start" if split == "train" else "center",
            max_samples=int(max_samples),
        )
    else:
        raise ValueError(f"暂不支持的数据集类型: {dataset_name}")
    return dataset


def build_dataloader_from_dataset(
    config: dict[str, Any],
    dataset: Dataset,
    split: str = "train",
    *,
    dataloader_settings: dict[str, Any],
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """用已构建 Dataset 创建 DataLoader，便于自动 batch 探测后复用缓存。"""
    data_cfg = config["data"]
    seed = int(config.get("seed", 42))
    settings = dataloader_settings
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(settings["num_workers"]),
        "pin_memory": bool(settings["pin_memory"]),
        "generator": torch.Generator().manual_seed(seed + (0 if split == "train" else 100000)),
    }
    if distributed:
        loader_kwargs["sampler"] = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=split == "train",
            seed=seed,
            drop_last=False,
        )
    elif split == "train":
        loader_kwargs["sampler"] = SeededEpochSampler(dataset, seed=int(config.get("seed", 42)))
    else:
        loader_kwargs["shuffle"] = False
    if int(settings["num_workers"]) > 0:
        loader_kwargs["persistent_workers"] = bool(settings["persistent_workers"])
        loader_kwargs["timeout"] = int(settings.get("timeout", 0))
        if settings.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(settings["prefetch_factor"])
        if "in_order" in inspect.signature(DataLoader).parameters:
            loader_kwargs["in_order"] = bool(settings.get("in_order", True))

    return DataLoader(dataset, **loader_kwargs)


def build_dataloader(
    config: dict[str, Any],
    split: str = "train",
    *,
    device: torch.device | None = None,
    dataloader_settings: dict[str, Any] | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """根据配置创建 DataLoader。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = dataloader_settings or resolve_dataloader_settings(config, device=device, split=split)
    dataset = build_dataset(config, split=split, device=device, dataloader_settings=settings)
    return build_dataloader_from_dataset(
        config,
        dataset,
        split=split,
        dataloader_settings=settings,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
