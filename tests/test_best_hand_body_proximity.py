"""Check preservation of the best model and the additional interaction path."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.best_hand_body_proximity import BEST_PROXIMITY_VARIANT, BestHandBodyProximityFusion
from isaa.train import parse_args
from tests import test_hand_body_proximity as proximity_tests
from train_best_hand_body_proximity import training_arguments


class BestProximityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_baseline_checkpoint_preserves_all_parameters_inputs_and_logits(self):
        baseline = BodyLocalHandCTRWideRelativeFusion(7).eval()
        model = BestHandBodyProximityFusion(7, interaction_config={"width": 8}).eval()
        model.initialize_from_baseline({"architecture": baseline.ARCHITECTURE, "model": baseline.state_dict()})
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(model.HAND_INPUT_CHANNELS, 6)
        self.assertEqual(model.CHANNELS, baseline.CHANNELS)
        self.assertEqual(model.HAND_CHANNELS, baseline.HAND_CHANNELS)
        self.assertEqual(model.LOCAL_CHANNELS, baseline.LOCAL_CHANNELS)
        before, after = [], []
        hooks = [baseline.hand_ctr_blocks[0].register_forward_pre_hook(lambda _, args: before.append(args[0])),
                 model.hand_ctr_blocks[0].register_forward_pre_hook(lambda _, args: after.append(args[0]))]
        x = torch.cat((proximity_tests.ProximityTests.sample(), torch.zeros_like(proximity_tests.ProximityTests.sample())), -1)
        with torch.no_grad():
            expected = baseline(x)
            actual, details = model(x, return_interaction=True)
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for old, new in zip(before, after):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            self.assertEqual(new.shape[1], 6)
        self.assertGreater(after[0][:, 3:].abs().sum(), 0)
        self.assertGreater(details["edges"].shape[0], 0)
        self.assertEqual(details["added_features"].count_nonzero(), 0)
        cache = torch.cat((x, torch.randn(2, 5, 5, 133, 2)), 1)
        with torch.no_grad():
            torch.testing.assert_close(baseline(cache), model(cache), rtol=0, atol=0)
            empty = torch.full_like(x, float("nan"))
            torch.testing.assert_close(baseline(empty), model(empty), rtol=0, atol=0)

    def test_new_residual_learns_without_replacing_baseline(self):
        model = BestHandBodyProximityFusion(7, interaction_config={"width": 8})
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(proximity_tests.ProximityTests.sample())
            torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
            optimizer.step()
        self.assertGreater(model.interaction_projection.weight.abs().sum(), 0)
        self.assertGreater(sum(p.grad.abs().sum() for p in model.interaction.edge_encoder.parameters()), 0)
        self.assertGreater(model.hand_ctr_blocks[0].gcn.branches[0].feature_proj.weight.grad.abs().sum(), 0)
        model.eval()
        with torch.no_grad():
            _, details = model(proximity_tests.ProximityTests.sample(), return_interaction=True)
        self.assertGreater(details["added_features"].abs().sum(), 0)

    def test_preset_keeps_original_backbone(self):
        flags = training_arguments(["--dry-run", "--no-compile"])
        self.assertNotIn("--gcn-config", flags)
        with patch.object(sys, "argv", ["main.py"] + flags):
            args = parse_args()
        self.assertEqual(args.model_variant, BEST_PROXIMITY_VARIANT)
        self.assertEqual((args.node_count, args.feature_mode), (133, "raw"))
        self.assertIsNone(args.init_checkpoint)
        self.assertEqual(Path(args.save_dir), Path("outputs/best_hand_body_proximity_ntu60_xsub"))

    def test_training_from_scratch_evaluation_and_reload(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "samples.zip"
            x = proximity_tests.ProximityTests.sample()[0, :, :, :, 0]
            with zipfile.ZipFile(archive, "w") as output:
                for subject in (1, 3):
                    for action in (1, 2, 3):
                        payload = io.BytesIO()
                        np.savez(payload, keypoints=x[:2].permute(1, 2, 0).numpy()[:, None],
                                 scores=x[2].numpy()[:, None])
                        output.writestr(f"S001C001P{subject:03}R001A{action:03}_rgb.npz", payload.getvalue())
            command = [sys.executable, str(root / "train_best_hand_body_proximity.py"),
                       "--archive", str(archive), "--num-classes", "3",
                       "--epochs", "1", "--window-size", "5", "--batch-size", "3", "--test-batch-size", "3",
                       "--max-persons", "1", "--num-workers", "0", "--device", "cpu",
                       "--no-compile", "--no-drop-last", "--save-dir", str(directory / "runs")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            run = next((directory / "runs").iterdir())
            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["architecture"], BestHandBodyProximityFusion.ARCHITECTURE)
            self.assertEqual(checkpoint["model_config"]["hand_input_channels"], 6)
            self.assertIsNone(checkpoint["args"]["init_checkpoint"])
            evaluation = subprocess.run([sys.executable, str(root / "tools/evaluate_best.py"),
                                         "--checkpoint", str(run / "best.pt"), "--device", "cpu", "--num-workers", "0",
                                         "--output-dir", str(directory / "evaluation")],
                                        cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout[-3000:] + evaluation.stderr[-3000:])
            summary = json.loads((directory / "evaluation/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["samples"], 3)
            self.assertTrue(summary["same_top1_correct_count"])
            resume = subprocess.run([sys.executable, str(root / "train_best_hand_body_proximity.py"),
                                     "--init-checkpoint", str(run / "best.pt"), "--num-classes", "3", "--dry-run",
                                     "--device", "cpu", "--window-size", "5", "--num-workers", "0", "--no-compile",
                                     "--save-dir", str(directory / "resume")],
                                    cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(resume.returncode, 0, resume.stdout[-3000:] + resume.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
