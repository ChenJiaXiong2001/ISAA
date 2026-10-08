"""Create a derived RTMW dataset with cached torso-relative hand features.

Input data.npy:  N x 3 x T x 133 x M (raw x, y, score)
Output data.npy: N x 5 or 8 x T x 133 x M. With
``--cross-hand-distance``, the final three channels are the torso-scale-
normalized distance and unit direction from each hand joint to its
corresponding joint on the opposite hand.
The source dataset is never modified.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def preprocess_split(source: Path, destination: Path, chunk_size: int,
                    cross_hand_distance: bool = False) -> None:
    data_path = source / "data.npy"
    labels_path = source / "labels.npy"
    if not data_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Expected data.npy and labels.npy under {source}")
    src = np.load(data_path, mmap_mode="r")
    if src.ndim != 5 or src.shape[1] != 3 or src.shape[3] != 133:
        raise ValueError(f"Expected N x 3 x T x 133 x M in {data_path}; got {src.shape}")
    destination.mkdir(parents=True, exist_ok=True)
    output_channels = 8 if cross_hand_distance else 5
    out = np.lib.format.open_memmap(
        destination / "data.npy", mode="w+", dtype=np.float32,
        shape=(src.shape[0], output_channels, *src.shape[2:]),
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
        if cross_hand_distance:
            # RTMW hand order is left 91:112 and right 112:133.  Use the
            # corresponding joint on the opposite hand, normalized by torso
            # height so the feature is less sensitive to subject scale.
            shoulder_count = weights[:, :, :, :2].sum(-1)
            hip_count = weights[:, :, :, 2:].sum(-1)
            shoulder_center = (np.nan_to_num(points[:, :, :, :2]) *
                               weights[:, :, :, :2, None]).sum(-2) / np.maximum(shoulder_count[..., None], 1.0)
            hip_center = (np.nan_to_num(points[:, :, :, 2:]) *
                          weights[:, :, :, 2:, None]).sum(-2) / np.maximum(hip_count[..., None], 1.0)
            torso_scale = np.linalg.norm(shoulder_center - hip_center, axis=-1, keepdims=True)
            scale_valid = (shoulder_count > 0) & (hip_count > 0) & np.isfinite(torso_scale[..., 0])
            torso_scale = np.where(scale_valid[..., None], np.maximum(torso_scale, 1e-3), 1.0)
            left = relative[..., 91:112, :]
            right = relative[..., 112:133, :]
            pair_valid = visible[..., 91:112] & visible[..., 112:133]
            pair_vector = right - left
            distance = np.linalg.norm(pair_vector, axis=-1) / torso_scale
            distance *= pair_valid.astype(np.float32)
            distance_all = np.zeros((*distance.shape[:-1], 133), dtype=np.float32)
            distance_all[..., 91:112] = distance
            distance_all[..., 112:133] = distance
            # Store a unit direction pointing toward the corresponding joint
            # on the other hand.  The right hand receives the opposite vector,
            # so each node has a local "where is my partner?" direction.
            direction = pair_vector / np.maximum(
                np.linalg.norm(pair_vector, axis=-1, keepdims=True), 1e-6
            )
            direction *= pair_valid[..., None].astype(np.float32)
            direction_all = np.zeros((*direction.shape[:-2], 133, 2), dtype=np.float32)
            direction_all[..., 91:112, :] = direction
            direction_all[..., 112:133, :] = -direction
            features = np.concatenate((
                raw, relative, distance_all[..., None], direction_all,
            ), axis=-1)
        else:
            features = np.concatenate((raw, relative), axis=-1)
        features = features.transpose(0, 4, 1, 3, 2)
        out[start:end] = features
    out.flush()

    for name in ("labels.npy", "frame_mask.npy"):
        src_file = source / name
        if src_file.is_file():
            shutil.copy2(src_file, destination / name)
    channel_names = "raw_x,raw_y,score,torso_rel_x,torso_rel_y"
    if cross_hand_distance:
        channel_names += ",cross_hand_distance,cross_hand_direction_x,cross_hand_direction_y"
    print(f"{source} -> {destination}: {tuple(out.shape)} [{channel_names}]", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="directory containing train/ and val/")
    parser.add_argument("--output", required=True, type=Path, help="new derived dataset directory")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--cross-hand-distance", action="store_true",
                        help="append normalized corresponding left/right hand-joint distance")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    for split in ("train", "val"):
        preprocess_split(args.input / split, args.output / split, args.chunk_size,
                         args.cross_hand_distance)


if __name__ == "__main__":
    main()
