"""Scratch-only, compiled 20-epoch anchor-clean curriculum entry point."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import isaa.train as training
import isaa.face_pair_training as pair
from curriculum import CurriculumDataset, SCHEDULE, clean_probability
from isaa.models.face_pair_fusion import FACE_PAIR_HAND_MOTION_VARIANT
from train_best_face_pair_hand_motion_100 import training_arguments


def validate_protocol(args):
    required = {"model_variant": FACE_PAIR_HAND_MOTION_VARIANT, "epochs": 20,
                "max_samples": 4096, "batch_size": 32, "test_batch_size": 32,
                "seed": 17, "warmup_epochs": 1, "lr_scheduler": "warmup_cosine",
                "lr": 0.025, "warmup_start_lr": 0.001, "min_lr": 0.0001,
                "momentum": 0.9, "nesterov": True, "weight_decay": 0.0004,
                "grad_accum_steps": 1, "drop_last": True, "max_grad_norm": 5.0,
                "pair_head_loss_weight": 0.5, "pair_fusion_loss_weight": 1.0,
                "pair_calibration_fraction": 0.1, "pair_calibration_start_epoch": 1,
                "precompression_anchor_dropout_probability": 0.025,
                "precompression_anchor_clean_sample_probability": 0.5,
                "compile": True, "compile_mode": "default"}
    for key, value in required.items():
        if getattr(args, key) != value:
            raise ValueError(f"Curriculum protocol requires {key}={value!r}")
    if args.init_checkpoint or args.resume_checkpoint or "--no-compile" in sys.argv[1:]:
        raise ValueError("Curriculum training requires scratch initialization and compile")
    args.anchor_clean_curriculum = dict(SCHEDULE)
    return args


def main():
    sys.argv[1:] = training_arguments(sys.argv[1:])
    original_parse = training.parse_args
    original_epoch = pair.run_epoch

    def parse_args():
        return validate_protocol(original_parse())

    def run_epoch(compiled, loader, device, args, records, epoch, phase, *rest, **kwargs):
        if phase == "train":
            dataset = loader.dataset.dataset
            if not isinstance(dataset, CurriculumDataset):
                raise RuntimeError("Expected curriculum dataset")
            dataset.set_epoch(epoch)
            print(f"ANCHOR_CURRICULUM epoch={epoch} clean_probability={clean_probability(epoch):.3f}", flush=True)
        return original_epoch(compiled, loader, device, args, records, epoch, phase, *rest, **kwargs)

    training.parse_args = parse_args
    pair.RTMWZipDataset = CurriculumDataset
    pair.run_epoch = run_epoch
    training.main()


if __name__ == "__main__":
    main()
