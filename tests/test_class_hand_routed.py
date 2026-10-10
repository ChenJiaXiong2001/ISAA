"""Contracts for real hand skipping, supervised exits and train-only estimation."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from isaa.class_hand_training import auxiliary_loss
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.class_hand_routed import CLASS_HAND_VARIANT, BodyLocalClassHandRoutedFusion
from isaa.utils.hand_requirements import build_hand_requirements, split_routing_calibration


def table(scores):
    return {"num_classes": len(scores), "classes": [
        {"class_index": i, "need_score": value} for i, value in enumerate(scores)]}


class ClassHandRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(2, torch.get_num_threads()))

    def setUp(self):
        torch.manual_seed(2)
        self.model = BodyLocalClassHandRoutedFusion(3).eval()
        self.x = torch.randn(3, 3, 8, 133, 2) * 0.01
        self.x[:, 2] = 1

    def test_all_path_preserves_baseline_logits(self):
        baseline = BodyLocalHandCTRWideRelativeFusion(3).eval()
        self.model.initialize_from_baseline({"architecture": baseline.ARCHITECTURE,
                                             "model": baseline.state_dict()})
        with torch.inference_mode():
            expected = baseline(self.x)
            actual = self.model(self.x, hand_mode="all")
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_none_never_executes_hand_network(self):
        with patch.object(self.model, "_run_hand", side_effect=AssertionError("hand executed")):
            with torch.inference_mode():
                result = self.model(self.x, hand_mode="none", return_routing=True)
        self.assertFalse(result["hand_called"].any())
        torch.testing.assert_close(result["logits"], result["no_hand_logits"])

    def test_mixed_batch_runs_only_selected_people_and_body_once(self):
        selected = torch.tensor([False, True, False])
        decision = (selected, torch.zeros(3), torch.ones(3))
        with patch.object(self.model, "decide_hands", return_value=decision), \
             patch.object(self.model, "_run_hand", wraps=self.model._run_hand) as hands, \
             patch.object(self.model, "_body_features", wraps=self.model._body_features) as body:
            with torch.inference_mode():
                result = self.model(self.x, return_routing=True)
        self.assertEqual(body.call_count, 1)
        self.assertEqual(hands.call_count, 2)
        self.assertTrue(all(call.args[0].shape[0] == 2 for call in hands.call_args_list))
        torch.testing.assert_close(result["logits"][[0, 2]], result["no_hand_logits"][[0, 2]])
        with torch.inference_mode():
            full = self.model(self.x, hand_mode="all")
        torch.testing.assert_close(result["logits"][[1]], full[[1]], rtol=1e-5, atol=1e-6)

    def test_requirement_uses_probability_mixture_and_uncertainty_fallback(self):
        self.model.set_hand_requirements(table([0., 1., 1.]))
        self.model.confidence_threshold = 0.
        # Argmax class 0 has score zero; the two other classes jointly need hands.
        logits = torch.tensor([[0.40, 0.35, 0.25]]).log()
        called, score, _ = self.model.decide_hands(logits, torch.tensor([True]))
        self.assertTrue(called.item())
        self.assertAlmostEqual(score.item(), .60, places=5)
        self.model.set_hand_requirements(table([0., 0., 0.]))
        self.model.confidence_threshold = .8
        called, _, _ = self.model.decide_hands(torch.zeros(1, 3), torch.tensor([True]))
        self.assertTrue(called.item())

    def test_missing_hand_nodes_are_skipped(self):
        self.x[:, 2, :, 91:133] = 0
        with patch.object(self.model, "_run_hand", side_effect=AssertionError("missing hands executed")):
            with torch.inference_mode():
                result = self.model(self.x, return_routing=True)
        self.assertFalse(result["hand_called"].any())

    def test_auxiliary_heads_receive_direct_supervision_and_frozen_bn_stays_fixed(self):
        self.model.freeze_backbone()
        self.model.train()
        before = {name: value.clone() for name, value in self.model.named_buffers()}
        result = self.model(self.x, hand_mode="all", return_auxiliary=True)
        auxiliary_loss(result, torch.tensor([0, 1, 2])).backward()
        self.assertGreater(float(self.model.body_classifier.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.no_hand_classifier.weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.classifier.weight.grad)
        for name, value in self.model.named_buffers():
            torch.testing.assert_close(value, before[name])

    def test_eight_channel_and_empty_frames_are_finite(self):
        x = torch.zeros(2, 8, 8, 133, 1)
        with torch.inference_mode():
            result = self.model(x, torch.zeros(2, 8, dtype=torch.bool), return_routing=True)
        self.assertTrue(torch.isfinite(result["logits"]).all())
        self.assertFalse(result["hand_called"].any())

    def test_unready_table_calls_all_available_hands(self):
        called, _, _ = self.model.decide_hands(torch.zeros(2, 3), torch.tensor([True, False]))
        self.assertEqual(called.tolist(), [True, False])

    def test_table_rejects_duplicate_classes_and_nan(self):
        for bad in [table([float("nan"), 0., 1.]),
                    {"num_classes": 3, "classes": [{"class_index": 0, "need_score": 1.}] * 3}]:
            with self.assertRaises(ValueError):
                self.model.set_hand_requirements(bad)


class HandRequirementTests(unittest.TestCase):
    def test_benefit_harm_and_unsupported_class_are_separated(self):
        full = torch.tensor([[5., 0., 0.], [5., 0., 0.]])
        no_hand = torch.tensor([[0., 5., 0.], [0., 5., 0.]])
        result = build_hand_requirements(full, no_hand, torch.tensor([0, 1]), min_samples=1)
        rows = result["classes"]
        self.assertGreater(rows[0]["need_score"], .9)
        self.assertLess(rows[1]["need_score"], .1)
        self.assertEqual(rows[2]["need_score"], 1.)
        self.assertEqual(rows[0]["helped_samples"], 1)
        self.assertEqual(rows[1]["harmed_samples"], 1)

    def test_calibration_split_is_subject_disjoint_and_deterministic(self):
        members = [f"S001C001P{p:03}R001A{a:03}.npz" for p in range(1, 5) for a in range(1, 4)]
        fit, cal, subjects = split_routing_calibration(members, .25, 2)
        self.assertFalse(set(fit) & set(cal))
        self.assertEqual(set(fit) | set(cal), set(range(len(members))))
        self.assertTrue(all(int(members[i][9:12]) in subjects for i in cal))
        self.assertEqual((fit, cal, subjects), split_routing_calibration(members, .25, 2))

    def test_variant_argument_contract(self):
        from isaa.train import parse_args
        with patch.object(sys, "argv", ["main.py", "--model-variant", CLASS_HAND_VARIANT]):
            args = parse_args()
        self.assertEqual((args.node_count, args.feature_mode, args.main_only), (133, "raw", False))
        with patch.object(sys, "argv", ["main.py", "--model-variant", CLASS_HAND_VARIANT,
                                         "--body-loss-weight", "0"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_complete_training_requirement_and_evaluation_workflow(self):
        import io
        import zipfile
        import numpy as np
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "samples.zip"
            rng = np.random.default_rng(4)
            with zipfile.ZipFile(archive, "w") as z:
                for subject in (1, 2, 4, 5, 3):
                    for action in (1, 2, 3):
                        payload = io.BytesIO()
                        np.savez(payload, keypoints=rng.normal(0, .01, (8, 1, 133, 2)).astype("float32"),
                                 scores=np.ones((8, 1, 133), dtype="float32"))
                        z.writestr(f"S001C001P{subject:03}R001A{action:03}_rgb.npz", payload.getvalue())
            baseline = BodyLocalHandCTRWideRelativeFusion(3)
            init = directory / "baseline.pt"
            torch.save({"model": baseline.state_dict(), "architecture": baseline.ARCHITECTURE}, init)
            command = [sys.executable, str(root / "main.py"), "--model-variant", CLASS_HAND_VARIANT,
                       "--init-checkpoint", str(init), "--archive", str(archive), "--num-classes", "3",
                       "--epochs", "1", "--window-size", "8", "--batch-size", "3",
                       "--test-batch-size", "3", "--num-workers", "0", "--device", "cpu",
                       "--no-compile", "--no-drop-last", "--save-dir", str(directory / "runs")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout[-5000:] + result.stderr[-5000:])
            run = next((directory / "runs").iterdir())
            requirements = json.loads((run / "hand_requirement.json").read_text())
            self.assertFalse(requirements["metadata"]["official_validation_used_for_requirements"])
            self.assertEqual(len(requirements["classes"]), 3)
            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            self.assertTrue(checkpoint["model"]["requirements_ready"].item())
            self.assertEqual(checkpoint["val_metrics"]["samples"], 3)
            self.assertEqual(json.loads((run / "status.json").read_text())["status"], "completed")
            evaluation = subprocess.run([sys.executable, str(root / "tools/evaluate_best.py"),
                                         "--checkpoint", str(run / "best.pt"), "--device", "cpu",
                                         "--num-workers", "0", "--hand-mode", "none",
                                         "--output-dir", str(directory / "evaluation")],
                                        cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout[-3000:] + evaluation.stderr[-3000:])
            metrics = json.loads((directory / "evaluation/summary.json").read_text())
            self.assertEqual(metrics["hand_call_rate"], 0.)


if __name__ == "__main__":
    unittest.main()
