"""Create a derived five-channel RTMW dataset with cached torso-relative xy.

Input data.npy:  N x 3 x T x 133 x M (raw x, y, score)
Output data.npy: N x 5 x T x 133 x M (raw x, y, score, relative x, relative y)
The source dataset is never modified.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def preprocess_split(source: Path, destination: Path, chunk_size: int) -> None:
    data_path = source / "data.npy"
    labels_path = source / "labels.npy"
    if not data_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Expected data.npy and labels.npy under {source}")
    src = np.load(data_path, mmap_mode="r")
    if src.ndim != 5 or src.shape[1] != 3 or src.shape[3] != 133:
        raise ValueError(f"Expected N x 3 x T x 133 x M in {data_path}; got {src.shape}")
    destination.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        destination / "data.npy", mode="w+", dtype=np.float32,
        shape=(src.shape[0], 5, *src.shape[2:]),
    )
    torso = np.asarray((5, 6, 11, 12), dtype=np.int64)
    for start in range(0, src.shape[0], chunk_size):
        end = min(start + chunk_size, src.shape[0])
        # Work as N x T x M x V x C for per-frame/per-person center math.
        batch = np.asarray(src[start:end], dtype=np.float32).transpose(0, 2, 4, 3, 1)
        points = batch[..., torso, :2]
        valid = (batch[..., torso, 2] > 0) & np.isfinite(points).all(-1)
        weights = valid.astype(np.float32)
        center = (np.nan_to_num(points) * weights[..., None]).sum(-2) / np.maximum(weights.sum(-1, keepdims=True), 1.0)
        all_xy = np.nan_to_num(batch[..., :2])
        relative = all_xy - center[..., None, :]
        # Keep invalid joint coordinates at zero, matching the training mask.
        visible = (batch[..., 2] > 0) & np.isfinite(batch[..., :3]).all(-1)
        relative *= visible[..., None]
        raw = np.nan_to_num(batch[..., :3])
        raw *= visible[..., None]
        features = np.concatenate((raw, relative), axis=-1).transpose(0, 4, 1, 3, 2)
        out[start:end] = features
    out.flush()

    for name in ("labels.npy", "frame_mask.npy"):
        src_file = source / name
        if src_file.is_file():
            shutil.copy2(src_file, destination / name)
    print(f"{source} -> {destination}: {tuple(out.shape)} [raw_x,raw_y,score,torso_rel_x,torso_rel_y]", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="directory containing train/ and val/")
    parser.add_argument("--output", required=True, type=Path, help="new derived dataset directory")
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    for split in ("train", "val"):
        preprocess_split(args.input / split, args.output / split, args.chunk_size)


if __name__ == "__main__":
    main()
