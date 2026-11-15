"""Independent pair supervision, label-free class routing and strict entry tests."""
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_RAW_XY_VARIANT, FACE_PAIR_VARIANT,
    FacePairFusion, FacePairHandFaceGeometryFusion, FacePairRawXYFusion,
)
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.face_pair_training import (COMPILE_SCOPE, compile_model, compile_scope, numerical_stability_config,
                                     pair_loss, pair_loss_components, routing_report, run_epoch)
from isaa.train import parse_args
from isaa.utils.batch_cosine import BatchWarmupCosine
from train_best_face_pair_100 import training_arguments
from train_best_face_pair_raw_xy_100 import training_arguments as raw_arguments
from train_best_face_pair_handface_geometry_100 import training_arguments as geometry_arguments


class FacePairFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def parse(flags):
        with patch.object(sys, "argv", ["main.py", *flags]):
            return parse_args()

    @staticmethod
    def sample():
        torch.manual_seed(11)
        x = torch.randn(2, 3, 8, 133, 2) * 0.02 + 0.4
        x[:, 2] = 1
        x[:, 1, :, 11:13] += 0.2
        return x

    def test_default_preset_keeps_baseline_inputs_and_mandatory_compile(self):
        args = self.parse(training_arguments([]))
        self.assertEqual(args.model_variant, FACE_PAIR_VARIANT)
        self.assertEqual((args.window_size, args.temporal_compression), (100, "confidence_weighted"))
        self.assertEqual((args.feature_mode, args.node_count, args.max_persons), ("raw", 133, 2))
        self.assertEqual((args.epochs, args.batch_size, args.test_batch_size), (100, 32, 32))
        self.assertEqual(args.lr_scheduler, "warmup_cosine")
        self.assertEqual((args.lr, args.max_grad_norm), (0.025, 5.0))
        self.assertTrue(args.compile)
        self.assertEqual(args.compile_mode, "default")
        self.assertEqual(args.pair_calibration_fraction, 0.1)
        self.assertEqual(args.pair_head_loss_weight, 0.5)
        self.assertEqual(args.pair_fusion_loss_weight, 1.0)
        self.assertEqual(args.pair_calibration_start_epoch, 5)
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.resume_checkpoint)
        stability = numerical_stability_config(args)
        self.assertEqual((stability["base_lr"], stability["max_grad_norm"]), (.025, 5.))
        self.assertFalse(stability["loss_weight_normalization"])
        self.assertEqual(stability["loss_weights"], {"fusion": 1., "hand_face": .5,
                                                      "body_face": .5, "consistency": 0.})
        self.assertIn("logsumexp", stability["fusion_domain"])

    def test_raw_xy_variant_is_separate_and_preserves_source_hand_face_coordinates(self):
        model = FacePairRawXYFusion(4)
        self.assertEqual(model.HAND_FACE_COORDINATE_MODE, "torso_relative_plus_raw_xy")
        self.assertEqual(model.model_config["variant"], FACE_PAIR_RAW_XY_VARIANT)
        self.assertNotEqual(model.ARCHITECTURE, FacePairFusion.ARCHITECTURE)
        self.assertEqual(model.hand_ctr_blocks[0].gcn.branches[0].feature_proj.in_channels, 8)
        self.assertEqual(model.face_st_blocks[0].spatial.in_channels, 5)
        x = self.sample()
        with torch.no_grad():
            # A global translation changes raw-XY branch inputs but leaves the
            # torso-relative legacy model's local inputs unchanged.
            shifted = x.clone()
            shifted[:, 0:2] += 0.35
            raw_a = x[:, :3].permute(0, 4, 1, 2, 3).reshape(-1, 3, 8, 133)
            raw_b = shifted[:, :3].permute(0, 4, 1, 2, 3).reshape(-1, 3, 8, 133)
            torso = torch.tensor((5, 6, 11, 12))
            center_a = raw_a[:, :2].index_select(-1, torso).mean(-1, keepdim=True)
            center_b = raw_b[:, :2].index_select(-1, torso).mean(-1, keepdim=True)
            self.assertTrue(torch.allclose(raw_a[:, :2] - center_a, raw_b[:, :2] - center_b,
                                           atol=1e-5, rtol=1e-5))
            self.assertFalse(torch.allclose(raw_a[:, :2], raw_b[:, :2]))

    def test_raw_xy_entry_keeps_compile_and_scratch_constraints(self):
        args = self.parse(raw_arguments([]))
        self.assertEqual(args.model_variant, FACE_PAIR_RAW_XY_VARIANT)
        self.assertTrue(args.compile)
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.resume_checkpoint)
        with self.assertRaisesRegex(ValueError, "torch.compile"):
            raw_arguments(["--no-compile"])

    def test_hand_face_geometry_variant_adds_only_hf_input_channels(self):
        model = FacePairHandFaceGeometryFusion(4).eval()
        self.assertEqual(model.model_config["variant"], FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT)
        self.assertNotEqual(model.ARCHITECTURE, FacePairFusion.ARCHITECTURE)
        self.assertEqual(model.HAND_INPUT_CHANNELS, 12)
        self.assertEqual(model.hand_ctr_blocks[0].gcn.branches[0].feature_proj.in_channels, 12)
        self.assertEqual(model.face_st_blocks[0].spatial.in_channels, 3)
        x = self.sample()
        with torch.no_grad():
            output = model(x, torch.ones(2, 8, dtype=torch.bool))
        self.assertEqual(tuple(output["hand_face_logits"].shape), (2, 4))
        self.assertEqual(tuple(output["body_face_logits"].shape), (2, 4))
        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_hand_face_geometry_entry_keeps_compile_and_scratch_constraints(self):
        args = self.parse(geometry_arguments([]))
        self.assertEqual(args.model_variant, FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT)
        self.assertTrue(args.compile)
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.resume_checkpoint)
        with self.assertRaisesRegex(ValueError, "torch.compile"):
            geometry_arguments(["--no-compile"])
        scope = compile_scope(args)
        self.assertIn("explicit hand-face geometry", scope["scope"])
        self.assertEqual(scope["inherited_eager_boundary"], "face LocalSTBlock.forward (unchanged baseline boundary)")

    def test_generic_defaults_match_preset_and_clipping_cannot_be_disabled(self):
        flags = training_arguments([])
        for flag in ("--lr", "--max-grad-norm"):
            position = flags.index(flag)
            del flags[position:position + 2]
        args = self.parse(flags)
        self.assertEqual((args.lr, args.max_grad_norm), (0.025, 5.0))
        for value in ("0", "-1", "nan", "inf", "-inf"):
            for base in (flags, training_arguments([])):
                with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    self.parse(base + [f"--max-grad-norm={value}"])
        args = self.parse(training_arguments(["--lr", "0.01", "--max-grad-norm", "2"]))
        self.assertEqual((args.lr, args.max_grad_norm), (0.01, 2.0))

    def test_compile_cannot_be_disabled_in_either_entry_or_argument_order(self):
        for flags in (["--no-compile"], ["--no-compile", "--compile"],
                      ["--compile", "--no-compile"]):
            with self.assertRaisesRegex(ValueError, "requires torch.compile"):
                training_arguments(flags)
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                self.parse(training_arguments([]) + flags)

    def test_scratch_and_raw_zip_protocol_are_enforced_in_both_entries(self):
        for flag in ("--init-checkpoint", "--resume-checkpoint", "--npy-dir"):
            with self.assertRaisesRegex(ValueError, "random initialization"):
                training_arguments([flag, "unused"])
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                self.parse(training_arguments([]) + [flag, "unused"])
        for flags in (["--window-size", "64"], ["--feature-mode", "isaa"],
                      ["--temporal-compression", "none"], ["--lr-scheduler", "step"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                self.parse(training_arguments([]) + flags)
        for flags in (["--pair-calibration-fraction", "0"],
                      ["--pair-head-loss-weight", "0"],
                      ["--pair-fusion-loss-weight", "nan"],
                      ["--pair-calibration-start-epoch", "0"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                self.parse(training_arguments([]) + flags)

    def test_class_routing_uses_accuracy_counts_and_handles_ties_unseen_classes(self):
        model = FacePairFusion(4)
        model.set_routing_counts([8, 3, 4, 0], [6, 7, 4, 0], [10, 10, 8, 0])
        torch.testing.assert_close(model.hand_face_weight, torch.tensor([1., 0., .5, .5]))
        hf = torch.tensor([[4., 1., 2., 0.], [0., 1., 3., -1.]])
        bf = torch.tensor([[0., 3., 1., -1.], [2., 0., 1., 2.]])
        score = model.hand_face_weight * hf.softmax(-1) + (1 - model.hand_face_weight) * bf.softmax(-1)
        expected = score / score.sum(-1, keepdim=True)
        torch.testing.assert_close(model.fuse_logits(hf, bf).exp(), expected)
        # A classifier's common logit offset cannot alter selected probabilities.
        torch.testing.assert_close(model.fuse_logits(hf + 100, bf - 100), expected.log(), atol=1e-5, rtol=1e-5)
        restored = FacePairFusion(4)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.fuse_logits(hf, bf), expected.log())
        for counts in (([11, 0, 0, 0], [0] * 4, [10] * 4),
                       ([1.5, 0, 0, 0], [0] * 4, [10] * 4)):
            with self.assertRaises(ValueError):
                model.set_routing_counts(*counts)

    def test_separate_heads_and_encoders_receive_gradients(self):
        model = FacePairFusion(4).train()
        output = model(self.sample(), torch.ones(2, 8, dtype=torch.bool))
        labels = torch.tensor([0, 3])
        loss = F.cross_entropy(output["logits"], labels)
        loss = loss + 0.5 * (F.cross_entropy(output["hand_face_logits"], labels)
                             + F.cross_entropy(output["body_face_logits"], labels))
        loss.backward()
        for module in (model.hand_face_head, model.body_face_head, model.hand_ctr_blocks,
                       model.face_st_blocks, model.body_blocks):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_direct_supervision_keeps_unselected_head_learning(self):
        model = FacePairFusion(4)
        labels = torch.tensor([0, 3])
        for hf_counts, bf_counts in (([2] * 4, [0] * 4), ([0] * 4, [2] * 4)):
            model.set_routing_counts(hf_counts, bf_counts, [2] * 4)
            hf = torch.tensor([[2., 0., 1., -1.], [1., -1., 0., 2.]], requires_grad=True)
            bf = torch.tensor([[0., 2., -1., 1.], [-1., 0., 2., 1.]], requires_grad=True)
            output = {"logits": model.fuse_logits(hf, bf),
                      "hand_face_logits": hf, "body_face_logits": bf}
            pair_loss(output, labels, head_weight=.5, fusion_weight=1.).backward()
            for head in (hf, bf):
                self.assertTrue(torch.isfinite(head.grad).all())
                self.assertGreater(float(head.grad.abs().sum()), 0)

    def test_three_loss_weights_remain_unnormalized(self):
        output = {"logits": torch.tensor([[2., -1.]]),
                  "hand_face_logits": torch.tensor([[1., 0.]]),
                  "body_face_logits": torch.tensor([[0., 1.]])}
        labels = torch.tensor([0])
        terms = pair_loss_components(output, labels)
        expected = 3 * terms["fusion_loss"] + 2 * (terms["hand_face_loss"] + terms["body_face_loss"])
        torch.testing.assert_close(pair_loss(output, labels, head_weight=2, fusion_weight=3), expected)

    def test_optional_consistency_loss_is_finite_and_default_is_unchanged(self):
        output = {"logits": torch.tensor([[2., -1.]]),
                  "hand_face_logits": torch.tensor([[1., 0.]]),
                  "body_face_logits": torch.tensor([[0., 1.]])}
        labels = torch.tensor([0])
        baseline = pair_loss(output, labels, head_weight=.5, fusion_weight=1.)
        with_consistency = pair_loss(output, labels, head_weight=.5, fusion_weight=1.,
                                     consistency_weight=.05)
        self.assertTrue(torch.isfinite(with_consistency))
        self.assertGreater(float(with_consistency), float(baseline))
        torch.testing.assert_close(
            pair_loss(output, labels, head_weight=.5, fusion_weight=1., consistency_weight=0.),
            baseline,
        )

    @staticmethod
    def epoch_fixture(directory, failure=None):
        class InvalidGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                return gradient * float("nan")

        class ToyPair(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, features, frame_mask):
                logits = self.linear(features)
                if failure == "loss":
                    logits = logits * float("nan")
                elif failure == "gradient":
                    logits = InvalidGradient.apply(logits)
                return {"logits": logits.log_softmax(-1),
                        "hand_face_logits": logits, "body_face_logits": logits + 0.1}

        model = ToyPair()
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[1., -1.], [-1., 1.]]))
            model.linear.bias.zero_()
        features = torch.tensor([[1000., -500.], [500., 1000.]])
        labels, mask = torch.tensor([1, 0]), torch.ones(2, 1, dtype=torch.bool)
        loader = [(features, labels, mask), (features * 0.5, labels, mask)]
        rows = []
        records = SimpleNamespace(directory=Path(directory), state={}, batch=rows.append,
                                  finish=lambda status: None)
        args = SimpleNamespace(num_classes=2, max_grad_norm=5., pair_head_loss_weight=.5,
                               pair_fusion_loss_weight=1., log_interval=100., epochs=2,
                               compile_mode="default", lr=.025)
        optimizer = torch.optim.SGD(model.parameters(), lr=.025, momentum=.9, nesterov=True)
        scheduler = BatchWarmupCosine(epochs=2, batches_per_epoch=2, warmup_epochs=0,
                                     base_lr=.025, warmup_start_lr=.001, min_lr=.0001)
        return model, loader, args, records, rows, optimizer, scheduler

    def test_batch_records_loss_components_logit_ranges_and_preclip_gradient(self):
        with TemporaryDirectory() as directory:
            model, loader, args, records, rows, optimizer, scheduler = self.epoch_fixture(directory)
            with patch("sys.stdout", io.StringIO()) as console:
                result = run_epoch(model, loader, torch.device("cpu"), args, records, 1,
                                   "train", optimizer, scheduler)
            self.assertEqual(scheduler.completed_updates, 2)
            self.assertEqual(len(rows), 2)
            self.assertGreater(rows[0]["gradient_norm"], args.max_grad_norm)
            for row in rows:
                for key in ("fusion_loss", "hand_face_loss", "body_face_loss",
                            "hand_face_logits_absmax", "body_face_logits_absmax", "gradient_norm"):
                    self.assertTrue(torch.isfinite(torch.tensor(row[key])))
                self.assertEqual(row["max_grad_norm"], 5.)
                self.assertAlmostEqual(row["loss"], row["fusion_loss"]
                                       + .5 * (row["hand_face_loss"] + row["body_face_loss"]), places=3)
            clipped = torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
            self.assertLessEqual(float(clipped.norm()), 5.00001)
            self.assertIn("grad_preclip=", console.getvalue())
            self.assertIn("CE_fused=", console.getvalue())
            self.assertEqual(result["samples"], 4)
            self.assertFalse((Path(directory) / "failure.pt").exists())

    def test_nonfinite_loss_or_gradient_saves_batch_and_aborts_before_update(self):
        for failure in ("loss", "gradient"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                model, loader, args, records, rows, optimizer, scheduler = self.epoch_fixture(directory, failure)
                with patch.object(optimizer, "step", wraps=optimizer.step) as update:
                    with patch("sys.stdout", io.StringIO()), self.assertRaisesRegex(RuntimeError, "snapshot saved"):
                        run_epoch(model, loader, torch.device("cpu"), args, records, 7,
                                  "train", optimizer, scheduler)
                update.assert_not_called()
                self.assertEqual((scheduler.completed_updates, len(rows)), (0, 0))
                snapshot = torch.load(Path(directory) / "failure.pt", weights_only=False)
                self.assertEqual((snapshot["epoch"], snapshot["step"], snapshot["phase"]), (7, 1, "train"))
                self.assertEqual(snapshot["completed_updates"], 0)
                self.assertEqual(snapshot["max_grad_norm"], 5.)
                self.assertEqual(snapshot["compile"]["required"], True)
                self.assertFalse(snapshot["numerical_stability"]["loss_weight_normalization"])
                for key, expected in zip(("x", "labels", "frame_mask"), loader[0]):
                    torch.testing.assert_close(snapshot[key], expected)
                    self.assertEqual(snapshot[key].device.type, "cpu")
                self.assertEqual(set(snapshot["output"]), {"logits", "hand_face_logits", "body_face_logits"})
                self.assertEqual(set(snapshot["loss_components"]), {"fusion_loss", "hand_face_loss", "body_face_loss"})
                self.assertEqual(snapshot["model"].keys(), model.state_dict().keys())
                self.assertIn("param_groups", snapshot["optimizer"])
                self.assertEqual(records.state["failure_batch"]["snapshot"], "failure.pt")
                if failure == "gradient":
                    self.assertTrue(any(not torch.isfinite(value).all() for value in snapshot["gradients"].values()))

    def test_compile_wrapper_requires_inductor_and_propagates_errors(self):
        model, compiled = object(), object()
        if not hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile_limit_hit"):
                compile_model(model, "default")
            return
        with torch._dynamo.config.patch(suppress_errors=True, fail_on_recompile_limit_hit=False,
                                        cache_size_limit=8, accumulated_cache_size_limit=256):
            with patch("torch._inductor.list_mode_options", return_value={"triton.cudagraphs": True}) as modes:
                with patch("torch.compile", return_value=compiled) as compiler:
                    self.assertIs(compile_model(model, "max-autotune"), compiled)
                    modes.assert_called_once_with("max-autotune")
                    compiler.assert_called_once_with(model, backend="inductor", options={"triton.cudagraphs": False})
            self.assertFalse(torch._dynamo.config.suppress_errors)
            self.assertTrue(torch._dynamo.config.fail_on_recompile_limit_hit)
            with patch("torch.compile", side_effect=RuntimeError("compiler failed")):
                with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                    compile_model(model, "default")
        with patch("isaa.face_pair_training.torch._dynamo.config", SimpleNamespace(suppress_errors=True)):
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile_limit_hit"):
                compile_model(model, "default")
        with patch.object(torch, "compile"):
            del torch.compile
            with self.assertRaisesRegex(RuntimeError, "requires torch.compile"):
                compile_model(model, "default")
        self.assertTrue(COMPILE_SCOPE["required"])
        self.assertFalse(COMPILE_SCOPE["suppress_errors"])
        self.assertTrue(COMPILE_SCOPE["fail_on_recompile_limit_hit"])
        self.assertFalse(COMPILE_SCOPE["whole_model_eager_fallback"])

    def test_routing_report_records_per_action_accuracy_and_selection(self):
        model = FacePairFusion(4)
        model.set_routing_counts([8, 3, 4, 0], [6, 7, 4, 0], [10, 10, 8, 0])
        report = routing_report(model, 6)
        self.assertEqual(report["source"], "training_subject_holdout_only")
        self.assertEqual(report["epoch"], 6)
        rows = report["classes"]
        self.assertEqual([row["selected"] for row in rows], ["hand_face", "body_face", "average", "average"])
        self.assertEqual([row["action_id"] for row in rows], ["A001", "A002", "A003", "A004"])
        self.assertEqual(rows[0]["hand_face_accuracy"], .8)
        self.assertEqual(rows[1]["body_face_accuracy"], .7)
        self.assertIsNone(rows[3]["hand_face_accuracy"])
        self.assertIsNone(rows[3]["body_face_accuracy"])

    def test_backbone_and_cross_hand_features_are_preserved(self):
        torch.manual_seed(19)
        baseline = BodyLocalHandCTRWideRelativeFusion(4).eval()
        torch.manual_seed(19)
        model = FacePairFusion(4).eval()
        for key, value in model.state_dict().items():
            if key.startswith(("body_blocks.", "hand_ctr_blocks.", "face_st_blocks.",
                               "hand_to_local.", "body_input_norm.")):
                torch.testing.assert_close(value, baseline.state_dict()[key], rtol=0, atol=0)
        before, after = [], []
        hooks = [baseline.hand_ctr_blocks[0].register_forward_pre_hook(lambda _, values: before.append(values[0])),
                 model.hand_ctr_blocks[0].register_forward_pre_hook(lambda _, values: after.append(values[0]))]
        try:
            with torch.no_grad():
                baseline(self.sample())
                model(self.sample())
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(after), 2)
        for original, updated in zip(before, after):
            self.assertEqual(updated.shape[1], 6)
            torch.testing.assert_close(updated, original, atol=1e-6, rtol=1e-6)
            self.assertGreater(float(updated[:, 3:].abs().sum()), 0)

    def test_masks_padding_missing_people_and_all_missing_remain_finite(self):
        model = FacePairFusion(4).eval()
        x = self.sample()
        x[..., 1] = 0
        frame = torch.ones(2, 8, dtype=torch.bool)
        frame[:, -2:] = False
        changed = x.clone()
        changed[:, :, -2:] = float("nan")
        with torch.no_grad():
            expected, actual = model(x, frame), model(changed, frame)
            missing = model(torch.full_like(x, float("nan")), frame)
            one_person = model(x[..., :1], frame)
        for key in expected:
            self.assertEqual(expected[key].shape, (2, 4))
            torch.testing.assert_close(expected[key], actual[key])
            torch.testing.assert_close(expected[key], one_person[key], rtol=1e-4, atol=1e-5)
            self.assertTrue(torch.isfinite(missing[key]).all())


if __name__ == "__main__":
    unittest.main()
