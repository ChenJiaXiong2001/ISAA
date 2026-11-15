"""Input dependency, grouping, selection learning, and strict entry regressions."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from isaa.train import parse_args
from isaa.data.input_joint_dataset import InputJointDataset
from isaa.input_joint_training import build_model, compile_model, export_checkpoint, load_checkpoint, gate_weight
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.input_joint_selection import (
    INPUT_SELECTION_VARIANT, InputJointSelectionFusion, JointStochasticGates,
    BODY, OPTIONAL_BODY, PROTECTED_BODY, selection_spec, validate_spec,
)
from train_best_input_selection_100 import training_arguments


class InputJointSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def parse(self, flags):
        with patch.object(sys, "argv", ["main.py", *flags]):
            return parse_args()

    def test_both_entries_require_compile_and_scratch(self):
        direct = ["--model-variant", INPUT_SELECTION_VARIANT]
        for flags in (training_arguments([]), direct, [f"--model-variant={INPUT_SELECTION_VARIANT}"]):
            args = self.parse(flags)
            self.assertTrue(args.compile)
            self.assertEqual((args.window_size, args.epochs, args.temporal_compression), (100, 100, "confidence_weighted"))
            self.assertEqual(args.lr_scheduler, "warmup_cosine")
        for flags in (["--no-compile"], ["--compile", "--no-compile"], ["--no-compile", "--compile"]):
            with self.assertRaisesRegex(ValueError, "requires torch.compile"):
                training_arguments(flags)
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                self.parse(direct + flags)
        for flag in ("--init-checkpoint", "--resume-checkpoint", "--npy-dir"):
            with self.assertRaises(ValueError):
                training_arguments([flag, "unused"])
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                self.parse(direct + [flag, "unused"])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            self.parse(direct + ["--stage", "compact"])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            self.parse(direct + ["--num-classes", "0"])
        self.assertEqual(gate_weight(5, args), 0.)
        self.assertAlmostEqual(gate_weight(25, args), args.sparsity_weight)

    def test_compile_invoked_and_failures_propagate(self):
        model = InputJointSelectionFusion(7)
        sentinel = object()
        with torch._dynamo.config.patch(suppress_errors=True, fail_on_recompile_limit_hit=False):
            with patch("torch.compile", return_value=sentinel) as compiler:
                self.assertIs(compile_model(model, "default"), sentinel)
                compiler.assert_called_once_with(model, mode="default")
            self.assertFalse(torch._dynamo.config.suppress_errors)
            self.assertTrue(torch._dynamo.config.fail_on_recompile_limit_hit)
            with patch("torch.compile", side_effect=RuntimeError("compiler failed")):
                with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                    compile_model(model, "default")

    def test_stg_probability_cost_and_classification_gradient(self):
        gates = JointStochasticGates()
        gates.mu.data.zero_()
        self.assertAlmostEqual(gates.expected_optional_count().item(), 27.)
        gates.expected_optional_count().backward()
        self.assertTrue((gates.mu.grad > 0).all())
        model = InputJointSelectionFusion(7).eval()
        model.gates.mu.data.fill_(.6)
        x = torch.randn(2, 3, 5, 129, 2) * .1
        x[:, 2] = 1.
        logits, _ = model(x, return_selection=True)
        logits.square().sum().backward()
        self.assertTrue(torch.isfinite(model.gates.mu.grad).all())
        self.assertGreater(model.gates.mu.grad.abs().sum().item(), 0.)

    def test_new_neural_computation_is_captured(self):
        graphs = []

        def capture_backend(graph, inputs):
            graphs.append(str(graph.graph).lower())
            return graph.forward

        torch._dynamo.reset()
        try:
            model = InputJointSelectionFusion(7).train()
            model.gates.mu.data.fill_(.6)
            x = torch.randn(2, 3, 5, 129, 2) * .1
            x[:, 2] = 1.
            with torch._dynamo.config.patch(suppress_errors=False, fail_on_recompile_limit_hit=True,
                                            allow_unspec_int_on_nn_module=True):
                compiled = torch.compile(model, backend=capture_backend)
                logits, penalty = compiled(x, return_selection=True)
                (logits.square().mean() + .1 * penalty).backward()
            captured = "\n".join(graphs)
            for name in ("gates", "body_blocks", "hand_ctr_blocks", "hand_to_local",
                         "body_projection", "local_projection", "classifier", "erf"):
                self.assertIn(name, captured)
            self.assertTrue(torch.isfinite(model.gates.mu.grad).all())
            self.assertGreater(model.gates.mu.grad.abs().sum().item(), 0.)
        finally:
            torch._dynamo.reset()

    def test_export_pair_closure_and_dependency_counts(self):
        model = InputJointSelectionFusion(7)
        model.gates.mu.data.fill_(-2.)
        model.gates.mu.data[len(OPTIONAL_BODY) + 3] = .8
        spec = model.export_selection()
        self.assertEqual(spec["hand_local_indices"], [0, 4])
        self.assertTrue({95, 116}.issubset(spec["input_indices"]))
        self.assertEqual(spec["raw_input_count"], 77)
        self.assertTrue(set(PROTECTED_BODY).issubset(spec["body_indices"]))
        self.assertEqual(validate_spec(spec), spec)
        spec["input_indices"].remove(116)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_spec(spec)

    def test_all_open_matches_original_baseline(self):
        original = BodyLocalHandCTRWideRelativeFusion(7).eval()
        model = InputJointSelectionFusion(7).eval()
        state = original.state_dict()
        # Corresponding weights only for this parity test; production never
        # loads original weights and compact training only reads a manifest.
        excluded = {"body_indices", "face_indices"}
        target = model.state_dict()
        common = {k: v for k, v in state.items() if k in target and k not in excluded}
        model.load_state_dict(common, strict=False)
        model.gates.mu.data.fill_(2.)
        raw = torch.randn(2, 3, 7, 133, 2) * .05
        raw[:, 2] = 1.
        raw[:, :, :, :, 1] = 0.
        raw[:, :, 3, 9] = 0.
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[0, -1] = False
        with torch.no_grad():
            expected = original(raw, mask)
            actual = model(raw.index_select(3, model.input_indices), mask)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_closed_inputs_cannot_influence_logits(self):
        model = InputJointSelectionFusion(7).eval()
        model.gates.mu.data.fill_(-5.)
        raw = torch.randn(2, 3, 5, 129, 2) * .1
        raw[:, 2] = 1.
        changed = raw.clone()
        retained = set(model.export_selection()["input_indices"])
        dropped = [i for i, j in enumerate(model.selection["input_indices"]) if j not in retained]
        changed[:, :, :, dropped] = float("nan")
        with torch.no_grad():
            torch.testing.assert_close(model(raw), model(changed), rtol=0, atol=0)

    def test_compact_graph_bn_missing_data_and_scratch(self):
        spec = selection_spec(sorted(PROTECTED_BODY), [0, 4, 8])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subset.json"
            path.write_text(json.dumps(spec))
            args = self.parse(training_arguments(["--stage", "compact", "--subset", str(path)]))
            with patch("torch.load", side_effect=AssertionError("No weight loading allowed")):
                model = build_model(args)
        self.assertFalse(hasattr(model, "gates"))
        self.assertEqual(model.body_input_norm.bn.num_features, 24)
        self.assertEqual(model.hand_ctr_blocks[0].gcn.static_topology.shape[-1], 3)
        self.assertEqual(model.body_blocks[0].gcn.static_topology.shape[-1], 8)
        x = torch.randn(2, 3, 5, spec["raw_input_count"], 2) * .1
        x[:, 2] = 1.
        x[0, :, :, :, 1] = float("nan")
        y, penalty = model(x, return_selection=True)
        (y.square().mean() + penalty).backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with self.assertRaisesRegex(ValueError, "no full-input fallback"):
            model(torch.zeros(2, 3, 5, 133, 2))

    def test_checkpoint_export_rebuild_and_no_overwrite(self):
        model = InputJointSelectionFusion(7)
        model.gates.mu.data.fill_(-1.)
        config = dict(num_classes=7, stage="select", subset=None, gate_sigma=.5, gate_initial_mu=1.)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "model.pt", Path(tmp) / "subset.json"
            torch.save(dict(architecture=model.ARCHITECTURE, model=model.state_dict(), model_config=config, epoch=3), source)
            spec = export_checkpoint(source, output, .5)
            self.assertEqual(spec["raw_input_count"], 75)
            self.assertFalse(spec["weights_transferred"])
            with self.assertRaises(FileExistsError):
                export_checkpoint(source, output, .5)
            restored, _ = load_checkpoint(source)
            self.assertEqual(restored.export_selection(), validate_spec(spec))

    def test_dataset_independent_of_deleted_coordinates_scores_and_frame_validity(self):
        spec = selection_spec(sorted(PROTECTED_BODY), [0, 4])
        ids = spec["input_indices"]
        rng = np.random.default_rng(1)
        points = rng.random((9, 2, 133, 2), dtype=np.float32)
        scores = np.full((9, 2, 133), .8, dtype=np.float32)
        points[0, :, ids, :] = np.nan
        dropped = sorted(set(range(133)) - set(ids))
        changed_points, changed_scores = points.copy(), scores.copy()
        changed_points[:, :, dropped] = np.nan
        changed_scores[:, :, dropped] = 1000.  # Must not trigger selected-score sigmoid normalization.
        with tempfile.TemporaryDirectory() as tmp:
            def dataset(name, pts, scr):
                archive = Path(tmp) / name
                buffer = io.BytesIO()
                np.savez(buffer, keypoints=pts, scores=scr)
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr("S001C001P001R001A001_rgb.npz", buffer.getvalue())
                return InputJointDataset(archive, input_indices=ids, split="train", split_protocol="xsub60",
                                         window_size=5, num_classes=60, layout="rtmw_133", max_persons=2,
                                         temporal_compression="confidence_weighted")
            first = dataset("a.zip", points, scores)
            second = dataset("b.zip", changed_points, changed_scores)
            compact = dataset("c.zip", points[:, :, ids], scores[:, :, ids])
            for a, b, c in zip(first[0], second[0], compact[0]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(a, c, rtol=0, atol=0)
            for ds in (first, second, compact):
                ds._archive.close()
                ds._archive = None


if __name__ == "__main__":
    unittest.main()
