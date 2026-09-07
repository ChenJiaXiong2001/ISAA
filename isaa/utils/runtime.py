"""Runtime auto-tuning helpers for CPU/CUDA execution."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import torch


AUTO_VALUE = "auto"
SUPPORTED_OPTIMIZATION_PROFILES = {"safe", "balanced", "throughput"}
SUPPORTED_AMP_DTYPES = {"auto", "float16", "bfloat16"}


def is_auto(value: object) -> bool:
    """Return True when a config value asks for automatic tuning."""
    return isinstance(value, str) and value.strip().lower() == AUTO_VALUE


def logical_cpu_count() -> int:
    """Return a conservative logical CPU count."""
    return max(1, os.cpu_count() or 1)


def env_int(name: str, default: int) -> int:
    """Parse an integer environment variable with a fallback."""
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def system_memory_bytes() -> int | None:
    """Return total physical memory when it can be discovered."""
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size) * int(page_count)


def parse_nonnegative_auto(value: object, *, name: str) -> int | None:
    """Parse a non-negative integer, returning None for auto."""
    if is_auto(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是非负整数或 auto，当前为 {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} 必须是非负整数或 auto，当前为 {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是非负整数或 auto，当前为 {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} 必须是非负整数或 auto，当前为 {value!r}")
    return parsed


def parse_positive_auto(value: object, *, name: str) -> int | None:
    """Parse a positive integer, returning None for auto."""
    parsed = parse_nonnegative_auto(value, name=name)
    if parsed is None:
        return None
    if parsed <= 0:
        raise ValueError(f"{name} 必须是正整数或 auto，当前为 {value!r}")
    return parsed


def parse_float_auto(
    value: object,
    *,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> float | None:
    """Parse a finite float, returning None for auto."""
    if is_auto(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数字或 auto，当前为 {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数字或 auto，当前为 {value!r}") from exc
    if not torch.isfinite(torch.tensor(parsed)).item():
        raise ValueError(f"{name} 必须是有限数字或 auto，当前为 {value!r}")
    if min_value is not None:
        too_small = parsed < min_value if min_inclusive else parsed <= min_value
        if too_small:
            op = ">=" if min_inclusive else ">"
            raise ValueError(f"{name} 必须 {op} {min_value}，当前为 {value!r}")
    if max_value is not None:
        too_large = parsed > max_value if max_inclusive else parsed >= max_value
        if too_large:
            op = "<=" if max_inclusive else "<"
            raise ValueError(f"{name} 必须 {op} {max_value}，当前为 {value!r}")
    return parsed


def _runtime_cfg(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("runtime", {})
    return value if isinstance(value, dict) else {}


def auto_tune_enabled(config: dict[str, Any]) -> bool:
    """Whether runtime auto tuning is enabled."""
    return bool(_runtime_cfg(config).get("auto_tune", True))


def optimization_profile(config: dict[str, Any]) -> str:
    """Return the configured runtime optimization profile."""
    profile = str(_runtime_cfg(config).get("optimization_profile", "throughput")).strip().lower()
    if profile not in SUPPORTED_OPTIMIZATION_PROFILES:
        raise ValueError(
            "runtime.optimization_profile 必须是 "
            f"{sorted(SUPPORTED_OPTIMIZATION_PROFILES)}，当前为 {profile!r}"
        )
    return profile


def _npz_size_bytes(config: dict[str, Any], split: str) -> int:
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict) or data_cfg.get("dataset") != "npz":
        return 0
    paths = []
    split_path = data_cfg.get(f"{split}_path")
    if split_path:
        paths.append(split_path)
    elif data_cfg.get("data_path"):
        paths.append(data_cfg["data_path"])
    total = 0
    for item in paths:
        path = Path(str(item))
        if path.is_file():
            total += path.stat().st_size
    return total


def resolve_dataloader_settings(
    config: dict[str, Any],
    *,
    device: torch.device,
    split: str,
) -> dict[str, Any]:
    """Resolve DataLoader worker/pinning settings from config, CPU count and CUDA."""
    data_cfg = config.get("data", {})
    runtime_cfg = _runtime_cfg(config)
    dataset_name = data_cfg.get("dataset", "dummy") if isinstance(data_cfg, dict) else "dummy"
    cpu_count = logical_cpu_count()

    profile = optimization_profile(config)
    requested_workers = data_cfg.get("num_workers", AUTO_VALUE) if isinstance(data_cfg, dict) else AUTO_VALUE
    parsed_workers = parse_nonnegative_auto(requested_workers, name="data.num_workers")
    if parsed_workers is None:
        if not auto_tune_enabled(config) or dataset_name == "dummy":
            num_workers = 0
        else:
            max_workers = parse_positive_auto(
                runtime_cfg.get("max_dataloader_workers", AUTO_VALUE),
                name="runtime.max_dataloader_workers",
            )
            if max_workers is None:
                if device.type == "cuda":
                    if profile == "throughput":
                        max_workers = min(24, max(4, cpu_count - 4))
                    elif profile == "balanced":
                        max_workers = min(12, max(2, cpu_count // 2))
                    else:
                        max_workers = min(4, max(1, cpu_count // 4))
                else:
                    max_workers = 8 if profile == "throughput" else 4

            large_npz_mb = parse_positive_auto(
                runtime_cfg.get("large_npz_worker_limit_mb", 128),
                name="runtime.large_npz_worker_limit_mb",
            )
            if large_npz_mb is None:
                large_npz_mb = 128
            npz_size_mb = _npz_size_bytes(config, split) / (1024 * 1024)
            is_windows_large_npz = os.name == "nt" and dataset_name == "npz" and npz_size_mb >= float(large_npz_mb)
            worker_memory_fraction = parse_float_auto(
                runtime_cfg.get("large_npz_worker_memory_fraction", 0.35),
                name="runtime.large_npz_worker_memory_fraction",
                min_value=0.05,
                max_value=0.95,
            )
            total_memory = system_memory_bytes()
            large_npz_too_big = False
            if is_windows_large_npz and total_memory is not None and worker_memory_fraction is not None:
                large_npz_too_big = _npz_size_bytes(config, split) > total_memory * worker_memory_fraction

            if large_npz_too_big:
                # Avoid Windows worker spawn pressure when the source archive is huge for host RAM.
                num_workers = 0
            elif device.type == "cuda":
                reserve = 2 if profile == "throughput" else 4
                num_workers = min(max_workers, max(2, cpu_count - reserve), max(0, cpu_count - 1))
            else:
                num_workers = min(max_workers, max(0, cpu_count // 4))
    else:
        num_workers = parsed_workers

    pin_memory_value = runtime_cfg.get("pin_memory", AUTO_VALUE)
    if is_auto(pin_memory_value):
        pin_memory = device.type == "cuda"
    else:
        pin_memory = bool(pin_memory_value)

    persistent_value = runtime_cfg.get("persistent_workers", AUTO_VALUE)
    if is_auto(persistent_value):
        persistent_workers = num_workers > 0
    else:
        persistent_workers = bool(persistent_value) and num_workers > 0

    prefetch_value = runtime_cfg.get("prefetch_factor", AUTO_VALUE)
    if num_workers <= 0:
        prefetch_factor = None
    elif is_auto(prefetch_value):
        prefetch_factor = 4 if device.type == "cuda" and profile == "throughput" else 2
    else:
        prefetch_factor = parse_positive_auto(prefetch_value, name="runtime.prefetch_factor")

    timeout = parse_nonnegative_auto(
        runtime_cfg.get("dataloader_timeout_seconds", 0),
        name="runtime.dataloader_timeout_seconds",
    )
    if timeout is None:
        timeout = 0

    in_order_value = runtime_cfg.get("dataloader_in_order", AUTO_VALUE)
    if is_auto(in_order_value):
        in_order = not (device.type == "cuda" and int(num_workers) > 0)
    else:
        in_order = bool(in_order_value)

    return {
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "timeout": int(timeout),
        "in_order": bool(in_order),
    }


def resolve_torch_thread_settings(
    config: dict[str, Any],
    *,
    device: torch.device,
    dataloader_workers: int,
) -> dict[str, int]:
    """Resolve torch intra-op and inter-op thread counts."""
    runtime_cfg = _runtime_cfg(config)
    profile = optimization_profile(config)
    cpu_count = logical_cpu_count()

    requested_threads = runtime_cfg.get("torch_num_threads", AUTO_VALUE)
    torch_threads = parse_positive_auto(requested_threads, name="runtime.torch_num_threads")
    if torch_threads is None:
        if not auto_tune_enabled(config):
            torch_threads = torch.get_num_threads()
        elif device.type == "cuda":
            max_threads = 6 if profile == "throughput" else 8
            torch_threads = max(1, min(max_threads, cpu_count - dataloader_workers))
        else:
            torch_threads = max(1, cpu_count - dataloader_workers)

    requested_interop = runtime_cfg.get("torch_num_interop_threads", AUTO_VALUE)
    interop_threads = parse_positive_auto(requested_interop, name="runtime.torch_num_interop_threads")
    if interop_threads is None:
        if not auto_tune_enabled(config):
            interop_threads = torch.get_num_interop_threads()
        else:
            interop_threads = max(1, min(4, torch_threads))

    return {
        "torch_num_threads": int(torch_threads),
        "torch_num_interop_threads": int(interop_threads),
    }


def resolve_dataset_runtime_settings(
    config: dict[str, Any],
    *,
    device: torch.device,
    split: str,
    dataloader_workers: int,
) -> dict[str, bool]:
    """Resolve expensive dataset processing options."""
    data_cfg = config.get("data", {})
    runtime_cfg = _runtime_cfg(config)
    dataset_name = data_cfg.get("dataset", "dummy") if isinstance(data_cfg, dict) else "dummy"
    if dataset_name != "npz":
        return {"cache_processed": False, "share_memory": False}

    cache_value = runtime_cfg.get("cache_processed_dataset", AUTO_VALUE)
    if is_auto(cache_value):
        max_cache_gb = parse_float_auto(
            runtime_cfg.get("max_cache_dataset_gb", 24.0),
            name="runtime.max_cache_dataset_gb",
            min_value=0.1,
        )
        if max_cache_gb is None:
            max_cache_gb = 24.0
        npz_size_bytes = _npz_size_bytes(config, split)
        cache_processed = npz_size_bytes == 0 or npz_size_bytes <= max_cache_gb * (1024**3)
    else:
        cache_processed = bool(cache_value)

    share_value = runtime_cfg.get("share_memory_dataset", AUTO_VALUE)
    if is_auto(share_value):
        share_memory = (
            os.name == "nt"
            and device.type == "cuda"
            and dataloader_workers > 0
            and dataset_name == "npz"
        )
    else:
        share_memory = bool(share_value) and dataloader_workers > 0

    return {
        "cache_processed": bool(cache_processed),
        "share_memory": bool(share_memory),
    }


def resolve_mixed_precision_settings(config: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    """Resolve AMP settings for train/eval/probing loops."""
    runtime_cfg = _runtime_cfg(config)
    amp_value = runtime_cfg.get("amp", AUTO_VALUE)
    if is_auto(amp_value):
        enabled = auto_tune_enabled(config) and device.type == "cuda"
    else:
        enabled = bool(amp_value)
    if device.type != "cuda":
        enabled = False

    dtype_name = str(runtime_cfg.get("amp_dtype", AUTO_VALUE)).strip().lower()
    if dtype_name not in SUPPORTED_AMP_DTYPES:
        raise ValueError(f"runtime.amp_dtype 必须是 {sorted(SUPPORTED_AMP_DTYPES)}，当前为 {dtype_name!r}")
    if dtype_name == AUTO_VALUE:
        if enabled and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype_name = "bfloat16"
        else:
            dtype_name = "float16"
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return {
        "enabled": bool(enabled),
        "dtype_name": dtype_name,
        "dtype": dtype,
        "use_grad_scaler": bool(enabled and dtype_name == "float16"),
    }


def configure_torch_runtime(
    config: dict[str, Any],
    *,
    device: torch.device,
    dataloader_workers: int,
) -> dict[str, Any]:
    """Apply torch runtime settings and return a printable summary."""
    thread_settings = resolve_torch_thread_settings(config, device=device, dataloader_workers=dataloader_workers)
    torch.set_num_threads(thread_settings["torch_num_threads"])
    try:
        torch.set_num_interop_threads(thread_settings["torch_num_interop_threads"])
    except RuntimeError:
        # PyTorch only allows changing inter-op threads before parallel work starts.
        thread_settings["torch_num_interop_threads"] = torch.get_num_interop_threads()

    runtime_cfg = _runtime_cfg(config)
    cudnn_benchmark = runtime_cfg.get("cudnn_benchmark", AUTO_VALUE)
    if is_auto(cudnn_benchmark):
        cudnn_benchmark = device.type == "cuda"
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    allow_tf32 = runtime_cfg.get("allow_tf32", AUTO_VALUE)
    if is_auto(allow_tf32):
        allow_tf32 = device.type == "cuda" and auto_tune_enabled(config)
    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        matmul_backend = getattr(torch.backends.cuda, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    matmul_precision = str(runtime_cfg.get("float32_matmul_precision", "high"))
    if device.type == "cuda" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)

    cuda_name = None
    cuda_memory_gb = None
    if torch.cuda.is_available():
        try:
            cuda_name = torch.cuda.get_device_name(device if device.type == "cuda" else 0)
            if device.type == "cuda":
                cuda_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        except Exception:
            cuda_name = "available"

    return {
        "cpu_count": logical_cpu_count(),
        "system_memory_gb": None if system_memory_bytes() is None else system_memory_bytes() / (1024**3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": cuda_name,
        "cuda_memory_gb": cuda_memory_gb,
        "device": str(device),
        "optimization_profile": optimization_profile(config),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "allow_tf32": bool(allow_tf32),
        "float32_matmul_precision": matmul_precision,
        **thread_settings,
    }
