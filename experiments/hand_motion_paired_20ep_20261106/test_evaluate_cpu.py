"""CPU checks for the fixed 20-epoch pairing protocol."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from isaa.models.face_pair_fusion import FacePairHandMotionFusion
from evaluate import VARIANT, load_checkpoint, validate_checkpoints


def args(candidate=False):
    return {"model_variant": VARIANT, "num_classes": 5, "compile": True, "compile_mode": "default",
        "init_checkpoint": None, "resume_checkpoint": None, "feature_mode": "raw", "npy_dir": None,
        "window_size": 100, "temporal_compression": "confidence_weighted", "lr_scheduler": "warmup_cosine",
        "epochs": 20, "warmup_epochs": 1, "lr": .025, "warmup_start_lr": .001, "min_lr": .0001,
        "momentum": .9, "weight_decay": .0004, "grad_accum_steps": 1, "batch_size": 32,
        "max_samples": 4096, "seed": 17, "split": "xsub60", "save_dir": str(candidate),
        "precompression_anchor_dropout_probability": .05 if candidate else 0,
        "precompression_anchor_clean_sample_probability": .75 if candidate else 0}


def manifest():
    result = {"official_validation_used_for_routing": False}
    for key in ("fit", "calibration", "validation"):
        members = [key + "1", key + "2"]
        result[key] = {"members": members, "count": len(members),
                       "sha256": hashlib.sha256("\n".join(members).encode()).hexdigest()}
    return result


class ProtocolTests(unittest.TestCase):
    def test_checkpoint_and_protocol_guards(self):
        control, candidate = {"args": args(), "epoch": 20}, {"args": args(True), "epoch": 20}
        m = manifest()
        self.assertEqual(validate_checkpoints(control, candidate, [m, m]), ["validation1", "validation2"])
        candidate["epoch"] = 8
        with self.assertRaisesRegex(ValueError, "epoch-20"):
            validate_checkpoints(control, candidate, [m, m])
        candidate["epoch"] = 20
        candidate["args"]["weight_decay"] = .001
        with self.assertRaisesRegex(ValueError, "weight_decay"):
            validate_checkpoints(control, candidate, [m, m])
        candidate["args"]["weight_decay"] = .0004
        candidate["args"]["precompression_anchor_clean_sample_probability"] = .9
        with self.assertRaisesRegex(ValueError, "clean75"):
            validate_checkpoints(control, candidate, [m, m])

    def test_strict_last_checkpoint_and_calibration(self):
        model = FacePairHandMotionFusion(5).eval()
        model.routing_class_counts.fill_(10)
        model.routing_hand_face_correct.fill_(3)
        model.routing_body_face_correct.fill_(7)
        policy = {"enabled": True, "required": True, "backend": "inductor", "suppress_errors": False,
                  "fail_on_recompile_limit_hit": True, "whole_model_eager_fallback": False,
                  "triton_cudagraphs": False, "mode": "default"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            checkpoint = {"args": args(), "model": model.state_dict(), "architecture": model.ARCHITECTURE,
                          "epoch": 20, "compile": policy}
            torch.save(checkpoint, path)
            _, _, weights = load_checkpoint(path, torch.device("cpu"), False)
            torch.testing.assert_close(weights, torch.full((5,), .3))
            with self.assertRaisesRegex(ValueError, "last.pt"):
                load_checkpoint(Path(tmp) / "best.pt", torch.device("cpu"), False)
            checkpoint["compile"]["whole_model_eager_fallback"] = True
            torch.save(checkpoint, path)
            with self.assertRaises(Exception):
                load_checkpoint(path, torch.device("cpu"), False)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
