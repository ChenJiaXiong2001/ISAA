"""Routing, gradient health, preservation and scratch train/eval integration."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.best_action_hand_body import (
    ACTION_PROXIMITY_VARIANT, BestActionHandBodyFusion, ConditionalHandBodyInteraction, validated_config,
)
from isaa.models.hand_body_proximity import TorsoCoordinates
from isaa.train import parse_args
from tests import test_hand_body_proximity as proximity_tests
from train_best_action_hand_body import training_arguments


class ActionInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def model(self):
        return BestActionHandBodyFusion(60, interaction_config={"width": 8})

    def test_all_original_parameters_inputs_and_warmup_predictions_preserved(self):
        torch.manual_seed(12)
        baseline = BodyLocalHandCTRWideRelativeFusion(60).eval()
        torch.manual_seed(12)
        model = self.model().eval()
        for key, tensor in baseline.state_dict().items():
            torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(hasattr(model, "torso_projection"))
        self.assertEqual(model.HAND_INPUT_CHANNELS, 6)
        self.assertEqual([model.set_training_epoch(e) for e in (1, 5, 6, 15, 65)], [0., 0., .1, 1., 1.])
        model.set_training_epoch(1)
        before, after = [], []
        hooks = [baseline.hand_ctr_blocks[0].register_forward_pre_hook(lambda _, args: before.append(args[0])),
                 model.hand_ctr_blocks[0].register_forward_pre_hook(lambda _, args: after.append(args[0]))]
        x = proximity_tests.ProximityTests.sample()
        with torch.no_grad():
            expected = baseline(x)
            actual, details = model(x, return_interaction=True)
            cache = torch.cat((x, torch.randn(2, 5, 5, 133, 1)), 1)
            torch.testing.assert_close(baseline(cache), model(cache), rtol=0, atol=0)
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(len(before), len(after))
        for old, new in zip(before, after):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            self.assertGreater(new[:, 3:].abs().sum(), 0)
        self.assertEqual(details["edges"].shape[0], 0)

    def test_candidate_groups_reading_disabled_and_uncertain_shoes_combined(self):
        model = self.model().eval()
        logits = torch.full((3, 60), -40.)
        logits[0, 10] = 40  # reading, A011: no new route
        logits[1, 15:17] = 0  # wear/remove shoes, A016/A017
        logits[2, 27] = 40  # phone A028: head, not torso
        logits.requires_grad_()
        mass, gate, enabled = model.route_candidates(logits)
        self.assertFalse(mass.requires_grad)
        self.assertFalse(enabled[0].any())
        torch.testing.assert_close(mass[1, 4:], torch.ones(4))
        self.assertEqual(enabled[1].tolist(), [False] * 4 + [True] * 4)
        self.assertEqual(enabled[2].tolist(), [True] + [False] * 7)
        self.assertEqual(gate[0].count_nonzero(), 0)
        with patch.object(BodyLocalHandCTRWideRelativeFusion, "forward", return_value=logits.detach()):
            x = proximity_tests.ProximityTests.sample()
            output, details = model(torch.cat((x, x[:1])), return_interaction=True)
        torch.testing.assert_close(output[0], logits[0].detach(), rtol=0, atol=0)
        self.assertGreater(details["correction"][1].abs().sum(), 0)
        self.assertEqual(details["correction"][:, 10].count_nonzero(), 0)

    def test_region_screening_all_21_nodes_and_far_right_hand_skipped(self):
        geometry = TorsoCoordinates()(proximity_tests.ProximityTests.sample())
        interaction = ConditionalHandBodyInteraction(8, .4, .48)
        enabled = torch.zeros(2, 8, dtype=torch.bool)
        enabled[0, 6] = True
        rows = []
        hook = interaction.edge_encoder.register_forward_pre_hook(lambda _, args: rows.append(len(args[0])))
        pooled, details = interaction(geometry, enabled)
        hook.remove()
        edges = details["edges"]
        self.assertEqual(set(edges[:, 0].tolist()), {0})
        self.assertEqual(set(edges[:, 2].tolist()), set(range(21)))
        self.assertEqual(set(edges[:, 3].tolist()), {17, 18, 19})
        self.assertEqual(rows, [5 * 21 * 3])
        self.assertEqual(pooled[1].count_nonzero(), 0)
        self.assertEqual(pooled[:, 1].count_nonzero(), 0)
        self.assertTrue((details["edge_distance"] < .4).all())
        enabled.zero_()
        with patch("torch.cdist", side_effect=AssertionError("Inactive region computed distances")):
            pooled, details = interaction(geometry, enabled)
        self.assertEqual(pooled.count_nonzero(), 0)

    def test_hysteresis_missing_frames_duplicates_and_no_neighbors(self):
        x = proximity_tests.ProximityTests.sample(time=4)[:1]
        x[:, 0, :, 17:20, 0] = torch.tensor([.3, .44, .5, .44])[None, :, None]
        geometry = TorsoCoordinates()(x)
        module = ConditionalHandBodyInteraction(8, .4, .48)
        enabled = torch.zeros(1, 8, dtype=torch.bool)
        enabled[:, 6] = True
        edges = module.build_edges(geometry["relative"], geometry["valid"], enabled)
        self.assertEqual(set(edges[:, 1].tolist()), {0, 1})
        mask = torch.tensor([[True, False, True, True]])
        geometry = TorsoCoordinates()(x, mask)
        edges = module.build_edges(geometry["relative"], geometry["valid"], enabled)
        self.assertEqual(set(edges[:, 1].tolist()), {0})
        x[:, :2, :, 9] = x[:, :2, :, 91]
        geometry = TorsoCoordinates()(x)
        enabled[:, 2] = True
        edges = module.build_edges(geometry["relative"], geometry["valid"], enabled)
        self.assertFalse(((edges[:, 2] == 0) & (edges[:, 3] == 9)).any())
        geometry = TorsoCoordinates()(torch.full_like(x, float("nan")))
        pooled, details = module(geometry, enabled)
        self.assertEqual(pooled.count_nonzero(), 0)
        self.assertEqual(details["edges"].numel(), 0)

    def test_joint_loss_gradient_reaches_branch_baseline_and_changes_features(self):
        torch.manual_seed(5)
        model = self.model().train()
        model.set_training_epoch(6)
        optimizer = torch.optim.SGD(model.optimizer_groups(.0004), lr=.02, momentum=.9)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], .00001)
        initial = model.interaction.node_encoder[0].weight.detach().clone()
        norms = []
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            logits, details = model(proximity_tests.ProximityTests.sample(), return_interaction=True)
            loss, parts = model.classification_loss(logits, details, torch.tensor([15, 16]))
            self.assertGreater(parts["interaction_aux_loss"], 0)
            loss.backward()
            for prefix in ("node_encoder", "edge_encoder", "temporal", "torso_encoder", "gate"):
                grads = [p.grad for name, p in model.interaction.named_parameters() if name.startswith(prefix)]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads), prefix)
                self.assertGreater(sum(g.abs().sum() for g in grads), 0, prefix)
            self.assertGreater(model.classifier.weight.grad.abs().sum(), 0)
            self.assertGreater(model.interaction_head.weight.grad.abs().sum(), 0)
            norms.append(model.interaction.node_encoder[0].weight.grad.norm().item())
            optimizer.step()
        self.assertTrue(all(v > 0 for v in norms))
        self.assertFalse(torch.equal(initial, model.interaction.node_encoder[0].weight))
        self.assertGreater(details["correction"].norm(), 0)

    def test_masked_people_do_not_dilute_interaction_and_empty_addition_is_zero(self):
        model = self.model().eval()
        forced = torch.full((2, 60), -40.)
        forced[:, 15] = 40
        x = proximity_tests.ProximityTests.sample()
        with torch.no_grad(), patch.object(BodyLocalHandCTRWideRelativeFusion, "forward", return_value=forced):
            single = model(x, return_routing=True)
            double = model(torch.cat((x, torch.zeros_like(x)), -1), return_routing=True)
            torch.testing.assert_close(single["correction"], double["correction"])
            masked = model(x, torch.zeros(2, 5, dtype=torch.bool), return_routing=True)
            empty = model(torch.full_like(x, float("nan")), return_routing=True)
        self.assertEqual(masked["correction"].count_nonzero(), 0)
        self.assertEqual(empty["correction"].count_nonzero(), 0)

    def test_autocast_interaction_forward_and_backward(self):
        model = self.model().train()
        model.set_training_epoch(6)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, details = model(proximity_tests.ProximityTests.sample(), return_interaction=True)
            loss, _ = model.classification_loss(logits, details, torch.tensor([15, 16]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(model.interaction.edge_encoder[0].weight.grad.abs().sum(), 0)

    def test_preset_scratch_rejection_and_config_validation(self):
        with patch.object(sys, "argv", ["main.py", *training_arguments([])]):
            args = parse_args()
        self.assertEqual(args.model_variant, ACTION_PROXIMITY_VARIANT)
        self.assertIsNone(args.init_checkpoint)
        self.assertTrue(args.compile)
        self.assertEqual(args.node_count, 133)
        self.assertEqual(args.interaction_config["region_actions"]["left_foot"], [16, 17])
        for flag in ("--init-checkpoint", "--init-checkpoint=old.pt"):
            with self.assertRaises(ValueError):
                training_arguments([flag, "old.pt"])
        with patch.object(sys, "argv", ["main.py", "--model-variant", ACTION_PROXIMITY_VARIANT,
                                        "--init-checkpoint", "old.pt"]), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parse_args()
        for config in ({"radius_on": -1}, {"candidate_threshold": 0}, {"region_actions": {"foot": [16]}},
                       {"region_actions": {"left_foot": [0]}}, {"width": True}, {"temperature": float("nan")},
                       {"region_actions": {"left_foot": [61]}}, {"auxiliary_loss_weight": 0}):
            with self.assertRaises(ValueError):
                validated_config(config, 60)

    def test_compile_is_required_in_preset_and_generic_entry(self):
        for overrides in ([], ["--compile"], ["--compile-mode", "default"]):
            with patch.object(sys, "argv", ["main.py", *training_arguments(overrides)]):
                self.assertTrue(parse_args().compile)
        for overrides in (["--no-compile"], ["--compile", "--no-compile"],
                          ["--no-compile", "--compile"]):
            with self.assertRaisesRegex(ValueError, "requires torch.compile"):
                training_arguments(overrides)
        for overrides in (["--no-compile"], ["--compile", "--no-compile"],
                          ["--no-compile", "--compile"]):
            with patch.object(sys, "argv", ["main.py", "--model-variant", ACTION_PROXIMITY_VARIANT,
                                            *overrides]), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_save_load_preserves_schedule_and_predictions(self):
        model = self.model().eval()
        model.set_training_epoch(6)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = self.model().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=False), strict=True)
        self.assertAlmostEqual(restored.interaction_strength.item(), .1)
        with torch.no_grad():
            expected = model(proximity_tests.ProximityTests.sample())
            torch.testing.assert_close(expected, restored(proximity_tests.ProximityTests.sample()))

    def test_compile_eager_matches_when_runtime_supports_it(self):
        model = self.model().eval()
        try:
            compiled = torch.compile(model, backend="eager")
        except RuntimeError as exc:
            if "Windows not yet supported" in str(exc):
                self.skipTest("Installed PyTorch 2.1 does not support torch.compile on Windows")
            raise
        with torch.no_grad():
            x = proximity_tests.ProximityTests.sample()
            torch.testing.assert_close(model(x), compiled(x))

    def test_active_interaction_neural_layers_are_captured_and_backward_is_finite(self):
        graphs = []

        def capture_backend(graph, inputs):
            graphs.append(str(graph.graph))
            return graph.forward

        model = self.model().train()
        model.set_training_epoch(6)
        try:
            compiled = torch.compile(model, backend=capture_backend)
        except RuntimeError as exc:
            if "Windows not yet supported" in str(exc):
                self.skipTest("Installed PyTorch 2.1 does not support torch.compile on Windows")
            raise
        logits, details = compiled(proximity_tests.ProximityTests.sample(), return_interaction=True)
        loss, _ = model.classification_loss(logits, details, torch.tensor([15, 16]))
        loss.backward()
        self.assertGreater(details["edges"].shape[0], 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.interaction.node_encoder[0].weight.grad.abs().sum(), 0)
        captured = "\n".join(graphs).lower()
        for name in ("node_encoder", "edge_encoder", "torso_encoder", "temporal", "interaction_head"):
            self.assertIn(name, captured)

    def test_scratch_training_stages_checkpoint_evaluation_and_ablation(self):
        try:
            torch.compile(torch.nn.Identity())
        except RuntimeError as exc:
            if "Windows not yet supported" in str(exc):
                self.skipTest("Compiled training requires a supported runtime; no eager fallback")
            raise
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "samples.zip"
            sample = proximity_tests.ProximityTests.sample()[0, :, :, :, 0]
            with zipfile.ZipFile(archive, "w") as output:
                for subject in (1, 3):
                    for action in (11, 16, 17):
                        payload = io.BytesIO()
                        np.savez(payload, keypoints=sample[:2].permute(1, 2, 0).numpy()[:, None],
                                 scores=sample[2].numpy()[:, None])
                        output.writestr(f"S001C001P{subject:03}R001A{action:03}_rgb.npz", payload.getvalue())
            config = directory / "config.json"
            config.write_text(json.dumps({"width": 8, "baseline_epochs": 1, "ramp_epochs": 2}), encoding="utf-8")
            command = [sys.executable, str(root / "train_best_action_hand_body.py"), "--archive", str(archive),
                       "--epochs", "3", "--window-size", "5", "--batch-size", "3", "--test-batch-size", "3",
                       "--max-persons", "1", "--num-workers", "0", "--device", "cpu", "--no-drop-last",
                       "--interaction-config", str(config), "--save-dir", str(directory / "runs")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr)
            run = next((directory / "runs").iterdir())
            with (run / "epochs.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["stage"] for r in rows], ["original_best_warmup", "action_interaction_ramp", "action_interaction_joint"])
            self.assertEqual(float(rows[0]["train_interaction_gradient_l2"]), 0)
            self.assertGreater(float(rows[1]["train_interaction_gradient_l2"]), 0)
            self.assertGreater(float(rows[2]["train_interaction_aux_loss"]), 0)
            checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["architecture"], BestActionHandBodyFusion.ARCHITECTURE)
            self.assertIsNone(checkpoint["args"]["init_checkpoint"])
            self.assertTrue(checkpoint["args"]["compile"])
            saved_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(saved_config["compile"]["enabled"])
            self.assertEqual(checkpoint["model_config"]["hand_input_channels"], 6)
            self.assertEqual(checkpoint["model"]["interaction_strength"].item(), 1)
            for mode in ("adaptive", "none"):
                evaluation = subprocess.run([sys.executable, str(root / "tools/evaluate_best.py"),
                            "--checkpoint", str(run / "last.pt"), "--device", "cpu", "--num-workers", "0",
                            "--interaction-mode", mode, "--output-dir", str(directory / mode)],
                            cwd=root, capture_output=True, text=True, timeout=120)
                self.assertEqual(evaluation.returncode, 0, evaluation.stdout[-3000:] + evaluation.stderr[-3000:])
                summary = json.loads((directory / mode / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["samples"], 3)
                self.assertEqual(summary["interaction_mode"], mode)
                if mode == "adaptive":
                    self.assertTrue(summary["same_top1_correct_count"])
                else:
                    self.assertEqual(summary["interaction_diagnostics"]["correction_l2_sum"], 0)
                    self.assertEqual(summary["interaction_diagnostics"]["encoded_edges"], 0)


if __name__ == "__main__":
    unittest.main()
