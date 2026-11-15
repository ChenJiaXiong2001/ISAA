"""CPU-only tests of checkpoint, split, fusion and subject-summary checks."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F

from isaa.models.face_pair_fusion import FacePairHandMotionFusion
from evaluate import VARIANT, load_checkpoint, validate_checkpoints, _soft_logits
from summarize import summarize


def args(variant):
    return {"model_variant": variant, "num_classes": 5, "compile": True, "compile_mode": "default",
        "init_checkpoint": None, "resume_checkpoint": None, "split": "xsub60", "window_size": 100,
        "max_persons": 2, "temporal_compression": "confidence_weighted", "max_samples": 4096,
        "tf32": True, "feature_mode": "raw", "epochs": 8, "batch_size": 32, "warmup_epochs": 1,
        "lr": 0.025, "min_lr": 0.0001, "max_grad_norm": 5.0, "pair_head_loss_weight": 0.5,
        "pair_fusion_loss_weight": 1.0, "pair_calibration_fraction": 0.1, "pair_calibration_start_epoch": 1, "seed": 17}


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_strict_loading_and_gamma1(self):
        with tempfile.TemporaryDirectory() as tmp:
            for variant, cls in ((VARIANT, FacePairHandMotionFusion),):
                model = cls(5).eval()
                model.routing_class_counts.fill_(10)
                model.routing_hand_face_correct.fill_(3)
                model.routing_body_face_correct.fill_(7)
                path = Path(tmp) / (variant + ".pt")
                checkpoint = {"args": args(variant), "model": model.state_dict(), "architecture": model.ARCHITECTURE, "epoch": 8}
                torch.save(checkpoint, path)
                _, loaded, weights = load_checkpoint(path, torch.device("cpu"), compile_for_inference=False)
                torch.testing.assert_close(weights, torch.full((5,), 0.3))
                output = {"hand_face_logits": torch.randn(2, 5), "body_face_logits": torch.randn(2, 5)}
                expected = 0.3 * F.softmax(output["hand_face_logits"], -1) + 0.7 * F.softmax(output["body_face_logits"], -1)
                torch.testing.assert_close(_soft_logits(output, weights).exp(), expected)
                checkpoint["architecture"] = "wrong"
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, "architecture"):
                    load_checkpoint(path, torch.device("cpu"), compile_for_inference=False)
                self.assertEqual(loaded.ARCHITECTURE, model.ARCHITECTURE)

    def test_checkpoint_and_manifest_matching(self):
        control = {"args": args(VARIANT), "epoch": 8}
        candidate = {"args": args(VARIANT), "epoch": 8}
        manifest = {k: {"members": [f"{k}1", f"{k}2"]} for k in ("fit", "calibration", "validation")}
        self.assertEqual(validate_checkpoints(control, candidate, [manifest, manifest]), ["validation1", "validation2"])
        candidate["args"]["window_size"] = 50
        with self.assertRaisesRegex(ValueError, "window_size"):
            validate_checkpoints(control, candidate, [manifest, manifest])
        candidate["args"]["window_size"] = 100
        bad = dict(manifest, calibration={"members": ["different"]})
        with self.assertRaisesRegex(ValueError, "calibration"):
            validate_checkpoints(control, candidate, [manifest, bad])
        overlapping = dict(manifest, calibration={"members": ["fit1"]})
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_checkpoints(control, candidate, [overlapping, overlapping])
        candidate["epoch"] = 7
        with self.assertRaisesRegex(ValueError, "epoch-8"):
            validate_checkpoints(control, candidate, [manifest, manifest])

    def test_strata_subject_bootstrap_and_label_guard(self):
        members = ["S001C001P001R001A001", "S001C001P002R001A001"]
        rows = []
        for name in ("clean", "point_0.05", "point_0.1", "group_0.05", "group_0.1"):
            for i, member in enumerate(members):
                rows.append({"condition": name, "sample": member, "true_class_index": "0",
                             "control_correct": "0", "candidate_correct": str(i)})
        result = summarize(rows, {members[0]}, "remaining")
        self.assertEqual((result["samples"], result["subjects"]), (1, 1))
        self.assertEqual(result["conditions"]["clean"]["candidate_minus_control"]["difference_pp"], 100.)
        rows[3]["true_class_index"] = "1"
        with self.assertRaisesRegex(ValueError, "Labels differ"):
            summarize(rows, {members[0]}, "all")


if __name__ == "__main__":
    unittest.main()
