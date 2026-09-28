"""Convert an RTMW ZIP/directory into fixed-window 32-node NumPy files.

Example:
  python tools/preprocess_rtmw.py --archive data/ntu60_skeletons_rtmw.zip \
      --output data/ntu60_rtmw_npy/xsub60 --split xsub60 --num-classes 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.layouts import register_skeleton_presets
from isaa.models.rtmw_local_ctr import RTMWLocalCTR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="RTMW ZIP 文件")
    parser.add_argument("--output", required=True, help="输出 split 目录")
    parser.add_argument("--split", choices=("xsub60", "xset60", "xsub120", "xset120"), default="xsub60")
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-main-nodes", type=int, default=32,
                        help="Keep this many RTMW region-center nodes; ignored with --keep-all-nodes")
    parser.add_argument("--keep-all-nodes", action="store_true",
                        help="Write all 133 RTMW nodes for the full-detail experiment")
    return parser.parse_args()


def convert_dataset(archive: Path, output_root: Path, args: argparse.Namespace,
                    main_indices: torch.Tensor) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val"):
        dataset = RTMWZipDataset(
            archive, split=split_name, split_protocol=args.split,
            window_size=args.window_size, num_joints=133,
            num_classes=args.num_classes, layout="rtmw_133", max_persons=2,
            augment=False, max_samples=args.max_samples,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False,
                            persistent_workers=args.num_workers > 0,
                            prefetch_factor=2 if args.num_workers > 0 else None)
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        data_mm = None
        labels_mm = np.lib.format.open_memmap(split_dir / "labels.npy", mode="w+",
                                              dtype=np.int64, shape=(len(dataset),))
        mask_mm = np.lib.format.open_memmap(split_dir / "frame_mask.npy", mode="w+",
                                            dtype=np.bool_, shape=(len(dataset), args.window_size))
        offset = 0
        for features, labels, frame_mask in loader:
            # Dataset features are x,y,dx,dy,score. The model baseline consumes
            # relative x/y and score, and only the 32 main nodes.
            batch = features[:, [0, 1, 4]].index_select(3, main_indices).numpy().astype(np.float32, copy=False)
            if data_mm is None:
                data_mm = np.lib.format.open_memmap(
                    split_dir / "data.npy", mode="w+", dtype=np.float32,
                    shape=(len(dataset), batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]),
                )
            end = offset + batch.shape[0]
            data_mm[offset:end] = batch
            labels_mm[offset:end] = labels.numpy()
            mask_mm[offset:end] = frame_mask.numpy()
            offset = end
            print(f"{split_name}: {offset}/{len(dataset)}", flush=True)
        if data_mm is None or offset != len(dataset):
            raise RuntimeError(f"预处理未完成: {split_name} {offset}/{len(dataset)}")
        data_mm.flush(); labels_mm.flush(); mask_mm.flush()
        # The fixed window repeats the last valid frame for short clips. The
        # plain BN1d run intentionally treats the stored window as dense.
        metadata = {"shape": list(data_mm.shape), "dtype": "float32", "split_protocol": args.split,
                    "window_size": args.window_size, "main_joint_indices": main_indices.tolist(),
                    "frame_mask_saved": True, "plain_bn1d_uses_all_frames": True}
        (split_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    register_skeleton_presets()
    archive = Path(args.archive).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    model = RTMWLocalCTR(num_classes=args.num_classes, main_only=True,
                         main_node_count=args.num_main_nodes)
    main_indices = (torch.arange(133, dtype=torch.long)
                    if args.keep_all_nodes else model.main_joint_indices.cpu())
    print(f"archive={archive}", flush=True)
    print(f"main_nodes={main_indices.tolist()}", flush=True)
    convert_dataset(archive, output, args, main_indices)
    print(f"completed: {output}", flush=True)


if __name__ == "__main__":
    main()
