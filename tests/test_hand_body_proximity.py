"""Behavioral checks for sparse proximity routing and additive model wiring."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

from isaa.models.body_local_full_gcn import FULL_GCN_VARIANT
from isaa.models.hand_body_proximity import (
    PROXIMITY_VARIANT, HandBodyProximityFusion, SparseHandBodyInteraction, TorsoCoordinates,
)
from isaa.train import parse_args
from train_hand_body_proximity import training_arguments


class ProximityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    @staticmethod
    def sample(time=5):
        x = torch.zeros(2, 3, time, 133, 1)
        x[:, 0] = 10
        x[:, 1] = 10
        x[:, 2] = 1
        x[:, :2, :, 5, 0] = torch.tensor([-.5, 0.])[None, :, None]
        x[:, :2, :, 6, 0] = torch.tensor([.5, 0.])[None, :, None]
        x[:, :2, :, 11, 0] = torch.tensor([-.5, 1.])[None, :, None]
        x[:, :2, :, 12, 0] = torch.tensor([.5, 1.])[None, :, None]
        # Left hand and left foot near each other; right hand is far away.
        x[:, 0, :, 91:112] = 0.0
        x[:, 1, :, 91:112] = 1.5
        x[:, 0, :, 17:20] = .1
        x[:, 1, :, 17:20] = 1.5
        x[:, 0, :, 112:] = -10
        return x

    @staticmethod
    def model():
        config = {k: {"channels": [8, 16], "strides": [1, 2]} for k in ("body", "hand", "face")}
        return HandBodyProximityFusion(7, backbone_config=config, interaction_config={"width": 8})

    def test_common_coordinates_and_absolute_torso(self):
        normalize = TorsoCoordinates()
        x = self.sample()
        before = normalize(x)
        moved = x.clone()
        moved[:, :2] = moved[:, :2]*3+7
        after = normalize(moved)
        torch.testing.assert_close(before["relative"], after["relative"])
        self.assertFalse(torch.allclose(before["torso_context"], after["torso_context"]))
        torch.testing.assert_close(before["scale"], torch.ones(2))
        x[:, 2, 2, [5, 6, 11, 12]] = 0
        missing = normalize(x)
        torch.testing.assert_close(missing["relative"][:, 2, 17], before["relative"][:, 2, 17])
        x[:, 2, :, [5, 6, 11, 12]] = 0
        self.assertFalse(normalize(x)["valid"].any())

    def test_all_hand_nodes_near_foot_and_no_far_edges(self):
        geometry = TorsoCoordinates()(self.sample())
        module = SparseHandBodyInteraction(width=8)
        encoded_rows = []
        hook = module.edge_encoder.register_forward_pre_hook(lambda _, args: encoded_rows.append(args[0].shape[0]))
        pooled, details = module(geometry, True)
        hook.remove()
        edges = details["edges"]
        self.assertEqual(set(edges[:, 2].tolist()), set(range(21)))
        self.assertEqual(set(edges[:, 3].tolist()), {17, 18, 19})
        self.assertEqual(encoded_rows, [2*5*21*3])
        self.assertTrue((details["edge_distance"] < .4).all())
        self.assertEqual(pooled[:, 1].count_nonzero(), 0)
        self.assertEqual(details["region_gate"][..., :6].count_nonzero(), 0)
        self.assertGreater(details["region_gate"][:, :, 0, 6].sum(), 0)

    def test_hysteresis_reset_and_duplicate_wrists(self):
        module = SparseHandBodyInteraction(width=8, radius_on=.4, radius_off=.5)
        q = torch.zeros(1, 5, 133, 2)
        valid = torch.zeros(1, 5, 133, dtype=torch.bool)
        valid[..., [91, 17, 9]] = True
        q[0, :, 17, 0] = torch.tensor([.45, .35, .45, .6, .45])
        edges = module.build_edges(q, valid)
        self.assertEqual(edges[:, 1].tolist(), [1, 2])
        self.assertEqual(edges[:, 3].tolist(), [17, 17])
        # A separate call starts disconnected, even at a distance within off.
        self.assertEqual(module.build_edges(q[:, 2:3], valid[:, 2:3]).shape[0], 0)

    def test_missing_padding_and_empty_interaction_are_zero(self):
        module = SparseHandBodyInteraction(width=8)
        x = self.sample()
        x[:, 2, :, 91:] = 0
        pooled, details = module(TorsoCoordinates()(x), True)
        self.assertEqual(details["edges"].shape, (0, 4))
        self.assertEqual(pooled.count_nonzero(), 0)
        x = self.sample()
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[:, 3:] = False
        geometry = TorsoCoordinates()(x, mask)
        _, details = module(geometry, True)
        self.assertTrue((details["edges"][:, 1] < 3).all())
        self.assertEqual(details["region_gate"][:, 3:].count_nonzero(), 0)
        x[:, :, :, 91] = float("nan")
        geometry = TorsoCoordinates()(x, mask)
        self.assertTrue(torch.isfinite(geometry["relative"]).all())
        self.assertFalse(geometry["valid"][..., 91].any())

    def test_forward_gradients_masked_person_and_checkpoint(self):
        model = self.model()
        x = self.sample()
        logits, details = model(x, return_interaction=True)
        self.assertEqual(logits.shape, (2, 7))
        self.assertEqual(details["relative_coordinates"].shape, (2, 5, 133, 2))
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for component in (model.interaction.edge_encoder, model.interaction.gate, model.torso_projection):
            self.assertGreater(sum(p.grad.abs().sum() for p in component.parameters()), 0)
        model.eval()
        with torch.no_grad():
            expected = model(x)
            padded = torch.cat((x, torch.full_like(x, float("nan"))), -1)
            torch.testing.assert_close(expected, model(padded))
            self.assertTrue(torch.isfinite(model(torch.full_like(x, float("nan")))).all())
            self.assertEqual(model(x[:, :, :1]).shape, (2, 7))
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "backbone": model.backbone_config,
                    "interaction": model.interaction_config}, buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        restored = HandBodyProximityFusion(7, backbone_config=state["backbone"], interaction_config=state["interaction"]).eval()
        restored.load_state_dict(state["model"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(x))

    def test_entry_configuration_and_original_default(self):
        with patch.object(sys, "argv", ["main.py"]):
            self.assertEqual(parse_args().model_variant, FULL_GCN_VARIANT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"interaction.json"
            path.write_text(json.dumps({"radius_on": .3, "radius_off": .4}))
            flags = ["main.py", "--model-variant", PROXIMITY_VARIANT, "--interaction-config", str(path)]
            with patch.object(sys, "argv", flags):
                args = parse_args()
            self.assertEqual((args.node_count, args.feature_mode, args.main_only), (133, "raw", False))
            self.assertEqual(args.interaction_config["radius_on"], .3)
            json.dumps(vars(args))
        flags = training_arguments(["--dry-run", "--no-compile"])
        with patch.object(sys, "argv", ["train_hand_body_proximity.py"]+flags):
            self.assertEqual(parse_args().model_variant, PROXIMITY_VARIANT)
        with self.assertRaises(ValueError):
            HandBodyProximityFusion(interaction_config={"radius_on": .5, "radius_off": .4})

    def test_autocast_forward_backward(self):
        model = self.model()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(self.sample())
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_training_evaluation_and_checkpoint_initialization(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "samples.zip"
            x = self.sample(time=5)[0, :, :, :, 0]
            with zipfile.ZipFile(archive, "w") as output:
                for subject in (1, 3):
                    for action in (1, 2, 3):
                        payload = io.BytesIO()
                        np.savez(payload, keypoints=x[:2].permute(1, 2, 0).numpy()[:, None],
                                 scores=x[2].numpy()[:, None])
                        output.writestr(f"S001C001P{subject:03}R001A{action:03}_rgb.npz", payload.getvalue())
            config = directory / "gcn.json"
            config.write_text(json.dumps(self.model().backbone_config))
            interaction = directory / "interaction.json"
            interaction.write_text(json.dumps({"width": 8, "radius_on": .3, "radius_off": .45}))
            command = [sys.executable, str(root / "train_hand_body_proximity.py"),
                       "--archive", str(archive), "--num-classes", "3", "--epochs", "1",
                       "--window-size", "5", "--batch-size", "3", "--test-batch-size", "3",
                       "--max-persons", "1", "--num-workers", "0", "--device", "cpu",
                       "--no-compile", "--no-drop-last", "--gcn-config", str(config),
                       "--interaction-config", str(interaction), "--save-dir", str(directory / "runs")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            run = next((directory / "runs").iterdir())
            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["architecture"], HandBodyProximityFusion.ARCHITECTURE)
            self.assertEqual(checkpoint["model_config"]["interaction_config"]["radius_on"], .3)
            self.assertEqual(checkpoint["val_metrics"]["samples"], 3)
            evaluation = subprocess.run([sys.executable, str(root / "tools/evaluate_best.py"),
                                         "--checkpoint", str(run / "best.pt"), "--device", "cpu",
                                         "--num-workers", "0", "--output-dir", str(directory / "evaluation")],
                                        cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout[-3000:] + evaluation.stderr[-3000:])
            metrics = json.loads((directory / "evaluation/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["samples"], 3)
            self.assertTrue(metrics["same_top1_correct_count"])
            initialize = subprocess.run([sys.executable, str(root / "train_hand_body_proximity.py"),
                                        "--init-checkpoint", str(run / "best.pt"), "--num-classes", "3",
                                        "--dry-run", "--device", "cpu", "--window-size", "5",
                                        "--num-workers", "0", "--no-compile",
                                        "--save-dir", str(directory / "initialized")],
                                       cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(initialize.returncode, 0, initialize.stdout[-3000:] + initialize.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
