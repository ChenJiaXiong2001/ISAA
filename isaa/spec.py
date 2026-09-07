"""HAPM 默认规格与运行规格校验。

该模块只保存模型可运行所需的通用默认值和配置校验逻辑。
具体骨架框架应由入口脚本注册到 isaa.skeleton_layout，再由配置引用。
"""

from __future__ import annotations

import math

from isaa.skeleton_layout import (
    GENERIC_LAYOUT,
    auto_register_skeleton_layouts,
    get_skeleton_layout,
    has_skeleton_layout,
    resolve_layout_name,
)


DEFAULT_REGION_LAYOUT = GENERIC_LAYOUT
DEFAULT_NUM_JOINTS = 133
DEFAULT_INPUT_CHANNELS = 5
DEFAULT_WINDOW_SIZE = 64
DEFAULT_NUM_REGIONS = 15
DEFAULT_NUM_SEGMENTS = 16
DEFAULT_PATTERN_DIM = 256
MAX_NUM_PROTOTYPES = 4
DEFAULT_BLOCK_CHANNELS = (64, 64, 64, 96, 128, 128, 128, 192, 256, 256)
SUPPORTED_DATASETS = {"dummy", "npz", "rtmw_zip"}
SUPPORTED_RTMW_SPLIT_PROTOCOLS = {"xsub", "xsub120", "xset", "xset120"}
SUPPORTED_RTMW_SCORE_NORMALIZATIONS = {"auto", "clip", "sigmoid"}
SUPPORTED_TRAINING_STAGES = {"baseline", "participation_pretrain", "prototype_finetune"}
SUPPORTED_INPUT_FORMATS = {"raw_xy_score", "raw_xyz_score", "precomputed_5ch", "precomputed_7ch", "fused_streams"}
SUPPORTED_INPUT_ORDERS = {"BCTN", "BCTNM", "BTNC", "BTMNC", "BTNMC", "BMCTN"}
SUPPORTED_MODEL_STREAMS = {"joint", "bone", "joint_motion", "bone_motion"}
SUPPORTED_GRAPH_CONVS = {"dynamic", "ctr_multi", "ctr_channel"}
SUPPORTED_OPTIMIZERS = {"adamw", "sgd"}
SUPPORTED_LR_SCHEDULERS = {"none", "multistep", "cosine"}
SUPPORTED_OPTIMIZATION_PROFILES = {"safe", "balanced", "throughput"}
SUPPORTED_AMP_DTYPES = {"auto", "float16", "bfloat16"}
SUPPORTED_MULTI_GPU_STRATEGIES = {"data_parallel", "ddp", "off"}
SUPPORTED_DDP_BACKENDS = {"auto", "nccl", "gloo"}
SUPPORTED_DDP_DEBUG_LEVELS = {"off", "info", "detail"}
DISALLOWED_MODEL_GRAPH_CONFIG_KEYS = {
    "joint_graph_scope",
    "cross_graph_conv",
    "cross_include_self_region",
}
DISALLOWED_MODEL_DETAIL_CONFIG_KEY = "detail"
AUTO_VALUE = "auto"


def _parse_positive_int(name: str, value: object, errors: list[str]) -> int | None:
    """解析正整数配置，并将错误累积到 errors。

    参数:
        name: 配置字段名，用于错误提示。
        value: 待解析值。
        errors: 错误列表，函数会将错误追加到其中。

    返回:
        解析成功时返回正整数；失败时返回 None。
    """
    if isinstance(value, bool):
        errors.append(f"{name} must be a positive integer, got {value!r}")
        return None
    if isinstance(value, float) and not value.is_integer():
        errors.append(f"{name} must be a positive integer, got {value!r}")
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a positive integer, got {value!r}")
        return None
    if parsed <= 0:
        errors.append(f"{name} must be positive, got {value!r}")
        return None
    return parsed


def _parse_nonnegative_int(name: str, value: object, errors: list[str]) -> int | None:
    """解析非负整数配置，并将错误累积到 errors。

    参数:
        name: 配置字段名，用于错误提示。
        value: 待解析值。
        errors: 错误列表，函数会将错误追加到其中。

    返回:
        解析成功时返回非负整数；失败时返回 None。
    """
    if isinstance(value, bool):
        errors.append(f"{name} must be a non-negative integer, got {value!r}")
        return None
    if isinstance(value, float) and not value.is_integer():
        errors.append(f"{name} must be a non-negative integer, got {value!r}")
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a non-negative integer, got {value!r}")
        return None
    if parsed < 0:
        errors.append(f"{name} must be non-negative, got {value!r}")
        return None
    return parsed


def _parse_nonnegative_int_or_auto(name: str, value: object, errors: list[str]) -> int | None:
    """解析非负整数或 auto。"""
    if isinstance(value, str) and value.strip().lower() == AUTO_VALUE:
        return None
    return _parse_nonnegative_int(name, value, errors)


def _parse_positive_int_or_auto(name: str, value: object, errors: list[str]) -> int | None:
    """解析正整数或 auto。"""
    if isinstance(value, str) and value.strip().lower() == AUTO_VALUE:
        return None
    return _parse_positive_int(name, value, errors)


def _parse_bool_or_auto(name: str, value: object, errors: list[str]) -> bool | None:
    """解析布尔值或 auto。"""
    if isinstance(value, str) and value.strip().lower() == AUTO_VALUE:
        return None
    if not isinstance(value, bool):
        errors.append(f"{name} must be a boolean or 'auto', got {value!r}")
        return None
    return value


def _parse_float(
    name: str,
    value: object,
    errors: list[str],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> float | None:
    """解析有限浮点数配置，并按需校验范围。

    参数:
        name: 配置字段名，用于错误提示。
        value: 待解析值。
        errors: 错误列表，函数会将错误追加到其中。
        min_value: 可选下界。
        max_value: 可选上界。
        min_inclusive: 是否允许等于下界。
        max_inclusive: 是否允许等于上界。

    返回:
        解析成功时返回 float；失败时返回 None。
    """
    if isinstance(value, bool):
        errors.append(f"{name} must be a finite number, got {value!r}")
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a finite number, got {value!r}")
        return None
    if not math.isfinite(parsed):
        errors.append(f"{name} must be finite, got {value!r}")
        return None
    if min_value is not None:
        too_small = parsed < min_value if min_inclusive else parsed <= min_value
        if too_small:
            op = ">=" if min_inclusive else ">"
            errors.append(f"{name} must be {op} {min_value}, got {value!r}")
    if max_value is not None:
        too_large = parsed > max_value if max_inclusive else parsed >= max_value
        if too_large:
            op = "<=" if max_inclusive else "<"
            errors.append(f"{name} must be {op} {max_value}, got {value!r}")
    return parsed


def _parse_float_or_auto(
    name: str,
    value: object,
    errors: list[str],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> float | None:
    """解析有限浮点数或 auto。"""
    if isinstance(value, str) and value.strip().lower() == AUTO_VALUE:
        return None
    return _parse_float(
        name,
        value,
        errors,
        min_value=min_value,
        max_value=max_value,
        min_inclusive=min_inclusive,
        max_inclusive=max_inclusive,
    )


def ensure_runtime_spec(config: dict) -> None:
    """校验运行规格的基本可用性。

    参数:
        config: 完整配置字典。

    校验内容:
        数据集类型、骨架布局注册状态、输入格式、输入维度顺序、
        训练阶段、detail mode、正整数/浮点数字段、channels 长度
        和 precomputed_5ch 约束。

    抛错:
        收集所有发现的问题后一次性抛 ValueError，方便用户同时修复多个配置错误。
    """
    errors = []
    if not isinstance(config, dict):
        raise ValueError(f"HAPM 运行规格无效: config must be a dict, got {type(config).__name__}")

    def section(name: str) -> dict:
        """读取顶层配置 section，结构错误时返回空 dict 并累计错误。"""
        value = config.get(name, {})
        if not isinstance(value, dict):
            errors.append(f"{name} must be a dict, got {value!r}")
            return {}
        return value

    data_cfg = section("data")
    runtime_cfg = section("runtime")
    model_cfg = section("model")
    training_cfg = section("training")
    loss_cfg = section("loss")
    raw_stream_cfg = model_cfg.get("streams", {})
    raw_augmentation_cfg = data_cfg.get("augmentation", {})
    if not isinstance(raw_stream_cfg, (dict, list, tuple)):
        errors.append(f"model.streams must be a dict or list, got {raw_stream_cfg!r}")
        stream_cfg: dict | list | tuple = {}
    else:
        stream_cfg = raw_stream_cfg
    if not isinstance(raw_augmentation_cfg, dict):
        errors.append(f"data.augmentation must be a dict, got {raw_augmentation_cfg!r}")
        augmentation_cfg = {}
    else:
        augmentation_cfg = raw_augmentation_cfg

    _parse_nonnegative_int("seed", config.get("seed", 42), errors)

    dataset = data_cfg.get("dataset", "dummy")
    if dataset not in SUPPORTED_DATASETS:
        errors.append(f"data.dataset must be one of {sorted(SUPPORTED_DATASETS)}, got {dataset!r}")
    if dataset == "rtmw_zip":
        split_protocol = str(data_cfg.get("split_protocol", "xsub120")).strip().lower()
        if split_protocol not in SUPPORTED_RTMW_SPLIT_PROTOCOLS:
            errors.append(
                "data.split_protocol must be one of "
                f"{sorted(SUPPORTED_RTMW_SPLIT_PROTOCOLS)}, got {split_protocol!r}"
            )
        score_normalization = str(data_cfg.get("score_normalization", "auto")).strip().lower()
        if score_normalization not in SUPPORTED_RTMW_SCORE_NORMALIZATIONS:
            errors.append(
                "data.score_normalization must be one of "
                f"{sorted(SUPPORTED_RTMW_SCORE_NORMALIZATIONS)}, got {score_normalization!r}"
            )
        _parse_positive_int("data.max_persons", data_cfg.get("max_persons", 2), errors)
        for key in ("max_samples_per_split", "max_train_samples", "max_val_samples"):
            if key in data_cfg:
                _parse_nonnegative_int(f"data.{key}", data_cfg[key], errors)

    skeleton_layout = resolve_layout_name(data_cfg)
    if not has_skeleton_layout(skeleton_layout):
        auto_register_skeleton_layouts()
    if not has_skeleton_layout(skeleton_layout):
        errors.append(
            "data.skeleton_layout must be 'generic' or a layout registered by the entry script, "
            f"got {skeleton_layout!r}"
        )

    input_format = data_cfg.get("input_format", "raw_xy_score")
    if input_format not in SUPPORTED_INPUT_FORMATS:
        errors.append(f"data.input_format must be one of {sorted(SUPPORTED_INPUT_FORMATS)}, got {input_format!r}")

    input_order = str(data_cfg.get("input_order", "BCTN")).upper()
    if input_order not in SUPPORTED_INPUT_ORDERS:
        errors.append(f"data.input_order must be one of {sorted(SUPPORTED_INPUT_ORDERS)}, got {input_order!r}")

    stage = training_cfg.get("stage", "baseline")
    if stage not in SUPPORTED_TRAINING_STAGES:
        errors.append(f"training.stage must be one of {sorted(SUPPORTED_TRAINING_STAGES)}, got {stage!r}")

    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        errors.append(f"training.optimizer must be one of {sorted(SUPPORTED_OPTIMIZERS)}, got {optimizer_name!r}")

    lr_scheduler = str(training_cfg.get("lr_scheduler", "none")).lower()
    if lr_scheduler not in SUPPORTED_LR_SCHEDULERS:
        errors.append(
            f"training.lr_scheduler must be one of {sorted(SUPPORTED_LR_SCHEDULERS)}, got {lr_scheduler!r}"
        )

    if DISALLOWED_MODEL_DETAIL_CONFIG_KEY in model_cfg:
        errors.append("detail residual options moved from YAML to code; remove model.detail")
    graph_config_keys = sorted(DISALLOWED_MODEL_GRAPH_CONFIG_KEYS.intersection(model_cfg))
    if graph_config_keys:
        errors.append(
            "graph structure options moved from YAML to code; remove model keys "
            f"{graph_config_keys!r}"
        )

    numeric_fields = {
        "data.num_classes": data_cfg.get("num_classes"),
        "data.num_joints": data_cfg.get("num_joints"),
        "data.input_channels": data_cfg.get("input_channels"),
        "data.window_size": data_cfg.get("window_size"),
        "data.num_samples": data_cfg.get("num_samples"),
        "data.batch_size": data_cfg.get("batch_size"),
        "model.num_segments": model_cfg.get("num_segments"),
        "model.pattern_dim": model_cfg.get("pattern_dim"),
        "model.pattern_hidden_dim": model_cfg.get("pattern_hidden_dim"),
        "model.num_blocks": model_cfg.get("num_blocks"),
        "model.num_prototypes": model_cfg.get("num_prototypes"),
        "training.epochs": training_cfg.get("epochs"),
    }
    parsed_fields = {
        name: _parse_positive_int(name, value, errors)
        for name, value in numeric_fields.items()
    }
    num_prototypes = parsed_fields.get("model.num_prototypes")
    if num_prototypes is not None and num_prototypes > MAX_NUM_PROTOTYPES:
        errors.append(
            f"model.num_prototypes must be between 1 and {MAX_NUM_PROTOTYPES}, "
            f"got {num_prototypes!r}"
        )
    _parse_positive_int(
        "training.epoch_checkpoint_interval",
        training_cfg.get("epoch_checkpoint_interval", 1),
        errors,
    )
    resume = training_cfg.get("resume", False)
    if not isinstance(resume, bool):
        errors.append(f"training.resume must be a boolean, got {resume!r}")
    resume_path = training_cfg.get("resume_path", "")
    if not isinstance(resume_path, str):
        errors.append(f"training.resume_path must be a string, got {resume_path!r}")
    pretrained_path = training_cfg.get("pretrained_path", "")
    if not isinstance(pretrained_path, str):
        errors.append(f"training.pretrained_path must be a string, got {pretrained_path!r}")
    pretrained_strict = training_cfg.get("pretrained_strict", True)
    if not isinstance(pretrained_strict, bool):
        errors.append(f"training.pretrained_strict must be a boolean, got {pretrained_strict!r}")
    save_epoch_checkpoints = training_cfg.get("save_epoch_checkpoints", True)
    if not isinstance(save_epoch_checkpoints, bool):
        errors.append(f"training.save_epoch_checkpoints must be a boolean, got {save_epoch_checkpoints!r}")
    save_best_checkpoint = training_cfg.get("save_best_checkpoint", True)
    if not isinstance(save_best_checkpoint, bool):
        errors.append(f"training.save_best_checkpoint must be a boolean, got {save_best_checkpoint!r}")
    _parse_nonnegative_int_or_auto("data.num_workers", data_cfg.get("num_workers", 0), errors)

    auto_tune = runtime_cfg.get("auto_tune", True)
    if not isinstance(auto_tune, bool):
        errors.append(f"runtime.auto_tune must be a boolean, got {auto_tune!r}")
    _parse_positive_int_or_auto("runtime.torch_num_threads", runtime_cfg.get("torch_num_threads", "auto"), errors)
    _parse_positive_int_or_auto(
        "runtime.torch_num_interop_threads",
        runtime_cfg.get("torch_num_interop_threads", "auto"),
        errors,
    )
    _parse_positive_int_or_auto(
        "runtime.max_dataloader_workers",
        runtime_cfg.get("max_dataloader_workers", "auto"),
        errors,
    )
    _parse_positive_int_or_auto(
        "runtime.large_npz_worker_limit_mb",
        runtime_cfg.get("large_npz_worker_limit_mb", 128),
        errors,
    )
    _parse_positive_int_or_auto("runtime.prefetch_factor", runtime_cfg.get("prefetch_factor", "auto"), errors)
    _parse_nonnegative_int("runtime.dataloader_timeout_seconds", runtime_cfg.get("dataloader_timeout_seconds", 0), errors)
    optimization_profile = str(runtime_cfg.get("optimization_profile", "throughput")).lower()
    if optimization_profile not in SUPPORTED_OPTIMIZATION_PROFILES:
        errors.append(
            "runtime.optimization_profile must be one of "
            f"{sorted(SUPPORTED_OPTIMIZATION_PROFILES)}, got {optimization_profile!r}"
        )
    _parse_bool_or_auto("runtime.pin_memory", runtime_cfg.get("pin_memory", "auto"), errors)
    _parse_bool_or_auto("runtime.persistent_workers", runtime_cfg.get("persistent_workers", "auto"), errors)
    _parse_bool_or_auto("runtime.dataloader_in_order", runtime_cfg.get("dataloader_in_order", "auto"), errors)
    _parse_bool_or_auto("runtime.cudnn_benchmark", runtime_cfg.get("cudnn_benchmark", "auto"), errors)
    _parse_bool_or_auto("runtime.allow_tf32", runtime_cfg.get("allow_tf32", "auto"), errors)
    cuda_visible_devices = runtime_cfg.get("cuda_visible_devices", "")
    if not isinstance(cuda_visible_devices, str):
        errors.append(f"runtime.cuda_visible_devices must be a string, got {cuda_visible_devices!r}")
    multi_gpu = runtime_cfg.get("multi_gpu", False)
    if not isinstance(multi_gpu, bool):
        errors.append(f"runtime.multi_gpu must be a boolean, got {multi_gpu!r}")
    multi_gpu_strategy = str(runtime_cfg.get("multi_gpu_strategy", "off")).lower()
    if multi_gpu_strategy not in SUPPORTED_MULTI_GPU_STRATEGIES:
        errors.append(
            "runtime.multi_gpu_strategy must be one of "
            f"{sorted(SUPPORTED_MULTI_GPU_STRATEGIES)}, got {multi_gpu_strategy!r}"
        )
    ddp_backend = str(runtime_cfg.get("ddp_backend", "auto")).lower()
    if ddp_backend not in SUPPORTED_DDP_BACKENDS:
        errors.append(f"runtime.ddp_backend must be one of {sorted(SUPPORTED_DDP_BACKENDS)}, got {ddp_backend!r}")
    _parse_positive_int("runtime.ddp_nproc_per_node", runtime_cfg.get("ddp_nproc_per_node", 1), errors)
    _parse_bool_or_auto("runtime.cuda_prefetch", runtime_cfg.get("cuda_prefetch", "auto"), errors)
    _parse_bool_or_auto("runtime.preload_first_batch", runtime_cfg.get("preload_first_batch", "auto"), errors)
    _parse_nonnegative_int_or_auto(
        "runtime.preload_train_batches",
        runtime_cfg.get("preload_train_batches", "auto"),
        errors,
    )
    _parse_bool_or_auto("runtime.ddp_static_graph", runtime_cfg.get("ddp_static_graph", "auto"), errors)
    _parse_positive_int("runtime.ddp_timeout_seconds", runtime_cfg.get("ddp_timeout_seconds", 600), errors)
    _parse_positive_int_or_auto("runtime.ddp_bucket_cap_mb", runtime_cfg.get("ddp_bucket_cap_mb", 128), errors)
    ddp_debug = str(runtime_cfg.get("ddp_debug", "info")).lower()
    if ddp_debug not in SUPPORTED_DDP_DEBUG_LEVELS:
        errors.append(f"runtime.ddp_debug must be one of {sorted(SUPPORTED_DDP_DEBUG_LEVELS)}, got {ddp_debug!r}")
    for key in (
        "ddp_async_error_handling",
        "ddp_find_unused_parameters",
        "ddp_gradient_as_bucket_view",
        "ddp_broadcast_buffers",
        "log_all_ranks",
    ):
        value = runtime_cfg.get(key, False)
        if not isinstance(value, bool):
            errors.append(f"runtime.{key} must be a boolean, got {value!r}")
    _parse_bool_or_auto("runtime.amp", runtime_cfg.get("amp", "auto"), errors)
    amp_dtype = str(runtime_cfg.get("amp_dtype", "auto")).lower()
    if amp_dtype not in SUPPORTED_AMP_DTYPES:
        errors.append(f"runtime.amp_dtype must be one of {sorted(SUPPORTED_AMP_DTYPES)}, got {amp_dtype!r}")
    _parse_bool_or_auto("runtime.cache_processed_dataset", runtime_cfg.get("cache_processed_dataset", "auto"), errors)
    _parse_bool_or_auto("runtime.share_memory_dataset", runtime_cfg.get("share_memory_dataset", "auto"), errors)
    _parse_float(
        "runtime.max_cache_dataset_gb",
        runtime_cfg.get("max_cache_dataset_gb", 24.0),
        errors,
        min_value=0.1,
    )
    _parse_float(
        "runtime.large_npz_worker_memory_fraction",
        runtime_cfg.get("large_npz_worker_memory_fraction", 0.35),
        errors,
        min_value=0.05,
        max_value=0.95,
    )

    _parse_float("model.dropout", model_cfg.get("dropout"), errors, min_value=0.0, max_value=1.0)
    graph_conv = str(model_cfg.get("graph_conv", "ctr_multi")).lower()
    if graph_conv not in SUPPORTED_GRAPH_CONVS:
        errors.append(f"model.graph_conv must be one of {sorted(SUPPORTED_GRAPH_CONVS)}, got {graph_conv!r}")
    graph_conv_config = model_cfg.get("graph_conv_config", {})
    if not isinstance(graph_conv_config, dict):
        errors.append(f"model.graph_conv_config must be a dict, got {graph_conv_config!r}")
        graph_conv_config = {}
    _parse_positive_int(
        "model.graph_conv_config.relation_reduction",
        graph_conv_config.get("relation_reduction", 8),
        errors,
    )
    _parse_positive_int(
        "model.graph_conv_config.min_relation_channels",
        graph_conv_config.get("min_relation_channels", 8),
        errors,
    )
    _parse_float(
        "model.graph_conv_config.topology_scale",
        graph_conv_config.get("topology_scale", 1.0),
        errors,
        min_value=0.0,
    )
    _parse_float(
        "model.graph_conv_config.learnable_scale",
        graph_conv_config.get("learnable_scale", 0.1),
        errors,
        min_value=0.0,
    )
    learnable_dense = graph_conv_config.get("learnable_dense", True)
    if not isinstance(learnable_dense, bool):
        errors.append(f"model.graph_conv_config.learnable_dense must be a boolean, got {learnable_dense!r}")
    diagonal_fast_path = graph_conv_config.get("diagonal_fast_path", True)
    if not isinstance(diagonal_fast_path, bool):
        errors.append(f"model.graph_conv_config.diagonal_fast_path must be a boolean, got {diagonal_fast_path!r}")
    pattern_normalize = model_cfg.get("pattern_normalize", False)
    if not isinstance(pattern_normalize, bool):
        errors.append(f"model.pattern_normalize must be a boolean, got {pattern_normalize!r}")
    _parse_float(
        "model.pattern_dropout",
        model_cfg.get("pattern_dropout"),
        errors,
        min_value=0.0,
        max_value=1.0,
    )
    _parse_float("model.proto_alpha", model_cfg.get("proto_alpha"), errors)
    _parse_float("training.lr", training_cfg.get("lr"), errors, min_value=0.0, min_inclusive=False)
    _parse_float("training.weight_decay", training_cfg.get("weight_decay"), errors, min_value=0.0)
    _parse_float("training.momentum", training_cfg.get("momentum", 0.9), errors, min_value=0.0, max_value=1.0)
    _parse_float(
        "training.adam_beta1",
        training_cfg.get("adam_beta1", 0.9),
        errors,
        min_value=0.0,
        max_value=1.0,
        max_inclusive=False,
    )
    _parse_float(
        "training.adam_beta2",
        training_cfg.get("adam_beta2", 0.999),
        errors,
        min_value=0.0,
        max_value=1.0,
        max_inclusive=False,
    )
    _parse_float("training.adam_eps", training_cfg.get("adam_eps", 1e-8), errors, min_value=0.0, min_inclusive=False)
    _parse_nonnegative_int("training.warmup_epochs", training_cfg.get("warmup_epochs", 0), errors)
    _parse_positive_int(
        "training.participation_start_epoch",
        training_cfg.get("participation_start_epoch", 1),
        errors,
    )
    _parse_nonnegative_int(
        "training.participation_warmup_epochs",
        training_cfg.get("participation_warmup_epochs", 0),
        errors,
    )
    _parse_positive_int(
        "training.prototype_start_epoch",
        training_cfg.get("prototype_start_epoch", 1),
        errors,
    )
    _parse_nonnegative_int(
        "training.prototype_warmup_epochs",
        training_cfg.get("prototype_warmup_epochs", 0),
        errors,
    )
    _parse_float(
        "training.lr_decay_rate",
        training_cfg.get("lr_decay_rate", 0.1),
        errors,
        min_value=0.0,
        max_value=1.0,
        min_inclusive=False,
    )
    _parse_float("training.min_lr", training_cfg.get("min_lr", 0.0), errors, min_value=0.0)
    _parse_float("training.gradient_clip_norm", training_cfg.get("gradient_clip_norm", 0.0), errors, min_value=0.0)
    _parse_float(
        "training.label_smoothing",
        training_cfg.get("label_smoothing", 0.0),
        errors,
        min_value=0.0,
        max_value=1.0,
        max_inclusive=False,
    )
    _parse_positive_int(
        "training.label_smoothing_start_epoch",
        training_cfg.get("label_smoothing_start_epoch", 1),
        errors,
    )
    nesterov = training_cfg.get("nesterov", True)
    if not isinstance(nesterov, bool):
        errors.append(f"training.nesterov must be a boolean, got {nesterov!r}")

    if isinstance(stream_cfg, dict):
        streams_enabled = stream_cfg.get("enabled", False)
        if not isinstance(streams_enabled, bool):
            errors.append(f"model.streams.enabled must be a boolean, got {streams_enabled!r}")
        stream_names = stream_cfg.get("names", ["joint"])
        stream_fusion = str(stream_cfg.get("fusion", "ensemble")).lower()
        if stream_fusion not in {"ensemble", "input_concat", "grouped_concat"}:
            errors.append("model.streams.fusion must be one of ['ensemble', 'input_concat', 'grouped_concat']")
        stream_groups = stream_cfg.get("groups")
        if stream_groups is not None:
            if not isinstance(stream_groups, (list, tuple)) or len(stream_groups) == 0:
                errors.append("model.streams.groups must be a non-empty list when provided")
            else:
                flattened_groups = []
                for group_index, group in enumerate(stream_groups):
                    if not isinstance(group, (list, tuple)) or len(group) == 0:
                        errors.append(f"model.streams.groups[{group_index}] must be a non-empty list")
                        continue
                    for item_index, name in enumerate(group):
                        flattened_groups.append(name)
                        if not isinstance(name, str) or name not in SUPPORTED_MODEL_STREAMS:
                            errors.append(
                                "model.streams.groups"
                                f"[{group_index}][{item_index}] must be one of "
                                f"{sorted(SUPPORTED_MODEL_STREAMS)}, got {name!r}"
                            )
    else:
        stream_names = stream_cfg
    if not isinstance(stream_names, (list, tuple)):
        errors.append(f"model.streams.names must be a list, got {stream_names!r}")
    else:
        if len(stream_names) == 0:
            errors.append("model.streams.names must not be empty")
        for index, name in enumerate(stream_names):
            if not isinstance(name, str) or name not in SUPPORTED_MODEL_STREAMS:
                errors.append(
                    f"model.streams.names[{index}] must be one of {sorted(SUPPORTED_MODEL_STREAMS)}, got {name!r}"
                )
        if isinstance(stream_cfg, dict):
            stream_groups = stream_cfg.get("groups")
            if isinstance(stream_groups, (list, tuple)) and stream_groups:
                flattened_groups = [
                    name
                    for group in stream_groups
                    if isinstance(group, (list, tuple))
                    for name in group
                    if isinstance(name, str)
                ]
                missing = sorted(set(stream_names) - set(flattened_groups))
                extra = sorted(set(flattened_groups) - set(stream_names))
                if missing or extra:
                    errors.append(
                        "model.streams.groups must exactly cover model.streams.names: "
                        f"missing={missing} extra={extra}"
                    )

    augmentation_enabled = augmentation_cfg.get("enabled", False)
    if not isinstance(augmentation_enabled, bool):
        errors.append(f"data.augmentation.enabled must be a boolean, got {augmentation_enabled!r}")
    random_temporal_crop = augmentation_cfg.get("random_temporal_crop", False)
    if not isinstance(random_temporal_crop, bool):
        errors.append(f"data.augmentation.random_temporal_crop must be a boolean, got {random_temporal_crop!r}")
    _parse_float(
        "data.augmentation.rotation_degrees",
        augmentation_cfg.get("rotation_degrees", 0.0),
        errors,
        min_value=0.0,
        max_value=180.0,
    )
    _parse_float(
        "data.augmentation.coordinate_jitter_std",
        augmentation_cfg.get("coordinate_jitter_std", 0.0),
        errors,
        min_value=0.0,
    )
    scale_range = augmentation_cfg.get("scale_range", [1.0, 1.0])
    if not isinstance(scale_range, (list, tuple)) or len(scale_range) != 2:
        errors.append(f"data.augmentation.scale_range must be a two-value list, got {scale_range!r}")
    else:
        _parse_float("data.augmentation.scale_range[0]", scale_range[0], errors, min_value=0.0, min_inclusive=False)
        _parse_float("data.augmentation.scale_range[1]", scale_range[1], errors, min_value=0.0, min_inclusive=False)
    lr_steps = training_cfg.get("lr_steps", [])
    if not isinstance(lr_steps, (list, tuple)):
        errors.append(f"training.lr_steps must be a list of positive epoch numbers, got {lr_steps!r}")
    else:
        for index, value in enumerate(lr_steps):
            _parse_positive_int(f"training.lr_steps[{index}]", value, errors)

    for key in (
        "lambda_rt_sparse",
        "lambda_region_sparse",
        "lambda_part_consistency",
        "lambda_detail",
        "lambda_proto_compact",
        "lambda_proto_separation",
        "lambda_proto_diversity",
        "lambda_proto_consistency",
        "proto_margin",
        "proto_diversity_delta",
    ):
        if key in loss_cfg:
            _parse_float(f"loss.{key}", loss_cfg.get(key), errors, min_value=0.0)

    channels = model_cfg.get("channels", ())
    if not isinstance(channels, (list, tuple)) or not channels:
        errors.append("model.channels must be a non-empty list")
    else:
        num_blocks = parsed_fields.get("model.num_blocks")
        if num_blocks is not None and len(channels) != num_blocks:
            errors.append("model.channels length must equal model.num_blocks")
        for index, value in enumerate(channels):
            _parse_positive_int(f"model.channels[{index}]", value, errors)

    input_channels = parsed_fields.get("data.input_channels")
    if input_format in {"raw_xy_score", "precomputed_5ch"} and input_channels is not None and input_channels != 5:
        errors.append(f"data.input_format={input_format} requires data.input_channels=5")
    if input_format in {"raw_xyz_score", "precomputed_7ch"} and input_channels is not None and input_channels != 7:
        errors.append(f"data.input_format={input_format} requires data.input_channels=7")

    layout_spec = get_skeleton_layout(skeleton_layout)
    num_joints = parsed_fields.get("data.num_joints")
    if layout_spec is not None and layout_spec.num_joints is not None and num_joints is not None:
        if num_joints != layout_spec.num_joints:
            errors.append(
                f"data.num_joints must be {layout_spec.num_joints} for skeleton_layout={skeleton_layout!r}, "
                f"got {num_joints}"
            )

    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"HAPM 运行规格无效: {joined}")
