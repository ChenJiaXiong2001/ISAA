"""Real narrow execution, label-free routing, calibration and pipeline checks."""
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from isaa.models.class_adaptive_gcn import ClassAdaptiveGCN
from isaa.models.body_local_full_gcn import BodyLocalFullGCNFusion
from isaa.models.adaptive_branch import load_adaptive_branch
from isaa.models.backbones.slimmable import indices, STAGE_OF_LAYER
from isaa.utils.adaptive_width import calibrate_profile_bn, split_width_subjects, greedy_width_search
from isaa import adaptive_width_training as training


class SyntheticActions(Dataset):
    def __init__(self):
        self.members = [f"S001C001P{s:03}R001A{c + 1:03}.pkl"
                        for s in range(1, 11) for c in range(3)]
        self.labels = [c for s in range(10) for c in range(3)]
        self.x = torch.randn(30, 3, 5, 133, 1, generator=torch.Generator().manual_seed(17)) * .01
        self.x[:, 2] = 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.x[i], self.labels[i], torch.ones(5, dtype=torch.bool)


class AdaptiveGCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        return ClassAdaptiveGCN(3, stage_maxima=(8, 16, 32), router_hidden=8).eval()

    def input(self, channels=3):
        x = torch.randn(2, channels, 5, 133, 1) * .01
        x[:, 2] = 1
        return x, torch.ones(2, 5, dtype=torch.bool)

    def test_full_parity_raw_cached_and_missing(self):
        model = self.model()
        config = {b: {"channels": [model.stage_maxima[s] for s in STAGE_OF_LAYER]}
                  for b in ("body", "hand", "face")}
        full = BodyLocalFullGCNFusion(3, backbone_config=config).eval()
        full.load_state_dict({k: model.state_dict()[k] for k in full.state_dict()})
        for channels in (3, 8):
            x, mask = self.input(channels)
            x[0, :, :, 91:] = float("nan")
            mask[1, -1] = False
            with torch.no_grad():
                torch.testing.assert_close(model.forward_fixed(x, mask), full(x, mask), rtol=0, atol=0)

    def test_real_kernel_slicing_preserves_temporal_branches(self):
        self.assertEqual(indices(8, 32, "cpu", 4).tolist(), [0, 1, 8, 9, 16, 17, 24, 25])
        model = self.model()
        x, mask = self.input()
        actual = torch.nn.functional.conv2d
        widths = []
        def observe(features, weights, *args, **kwargs):
            widths.append((features.shape[1], weights.shape[0], weights.shape[1]))
            return actual(features, weights, *args, **kwargs)
        with patch("isaa.models.backbones.slimmable.F.conv2d", side_effect=observe):
            model.forward_fixed(x, mask, model.min_profile)
        self.assertTrue(widths)
        self.assertLessEqual(max(row[0] for row in widths), 8)
        self.assertLessEqual(max(row[1] for row in widths), 24)  # ST: three partitions x eight.
        self.assertIn((8, 2, 8), widths)  # Every CTR temporal branch has two retained channels.

    def test_soft_gates_router_and_mixed_hard_gradients(self):
        model = self.model()
        x, mask = self.input()
        labels = torch.tensor([0, 2])
        output = model.forward_fixed(x, mask, labels=labels)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss = loss + torch.nn.functional.cross_entropy(model.route_logits(x, mask), labels)
        loss.backward()
        self.assertGreater(float(model.width_logits.grad.abs().sum()), 0)
        self.assertGreater(float(model.router[0].weight.grad.abs().sum()), 0)
        model.zero_grad(set_to_none=True)
        model.set_profiles([model.min_profile, model.max_profile, model.min_profile], model.max_profile)
        with patch.object(model, "route_logits", return_value=torch.tensor([[0., 10., 0.], [10., 0., 0.]])):
            output = model(x, mask, return_routing=True)
            output["logits"].sum().backward()
        self.assertEqual(output["widths"].tolist(), [list(model.max_profile), list(model.min_profile)])
        self.assertTrue(torch.isfinite(model.body_encoder.blocks[0].gcn1.convs[0].conv3.weight.grad).all())

    def test_uncalibrated_union_falls_back_once_with_order_preserved(self):
        model = self.model()
        a = (8, 8, 32) * 3
        b = (8, 16, 8) * 3
        shared = model.min_profile
        model.set_profiles([a, b, shared], shared)
        bank = model.bn_snapshot()
        model.normalization_stats = {model.profile_key(p): bank for p in (a, b, shared)}
        x = torch.zeros(4, 3, 5, 133, 1)
        x[:, 0, 0, 0, 0] = torch.arange(4.)
        seen = []
        def execute(values, mask, profile, **kwargs):
            ids = values[:, 0, 0, 0, 0]
            seen.extend(ids.tolist())
            return ids[:, None].expand(-1, 3)
        router = torch.tensor([[10., 0., 0.], [0., 0., 10.], [1., 1., -5.], [0., 0., 0.]])
        with patch.object(model, "route_logits", return_value=router), patch.object(model, "execute", side_effect=execute):
            output = model(x, return_routing=True)
        self.assertEqual(sorted(seen), [0., 1., 2., 3.])
        self.assertEqual(output["logits"][:, 0].tolist(), [0., 1., 2., 3.])
        self.assertEqual(output["fallback"].tolist(), [False, False, True, True])
        self.assertEqual(output["widths"][2:].tolist(), [list(shared), list(shared)])

    def test_bn_calibration_restores_live_state_and_mixed_backward(self):
        model = self.model()
        x, mask = self.input()
        original = model.bn_snapshot()
        fit = [(x, torch.tensor([0, 1]), mask)]
        banks = {model.profile_key(p): calibrate_profile_bn(model, fit, p, "cpu")
                 for p in (model.min_profile, model.max_profile)}
        for name, values in original.items():
            for key, value in values.items():
                torch.testing.assert_close(value, model.bn_snapshot()[name][key])
        model.normalization_stats = banks
        model.set_profiles([model.min_profile, model.max_profile, model.min_profile], model.max_profile)
        with patch.object(model, "route_logits", return_value=torch.tensor([[10., 0., 0.], [0., 10., 0.]])):
            model(x, mask).sum().backward()  # Switching banks must not invalidate autograd buffers.
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())

    def test_subject_partitions_and_bounded_search(self):
        data = SyntheticActions()
        fit, search, selection, info = split_width_subjects(data.members, .3, 1)
        self.assertEqual(sorted(fit + search + selection), list(range(30)))
        self.assertFalse(set(info["search_subjects"]) & set(info["selection_subjects"]))
        maximum = (8, 16, 32) * 3
        def predict(profile):
            self.model().validate_profile(profile)
            reduction = sum(maximum) - sum(profile)
            return torch.tensor([[2. - reduction * .01, 0.]]).repeat(10, 1), torch.zeros(10, dtype=torch.long)
        result = greedy_width_search(predict, maximum, max_steps=5, cost_weight=1.,
                                     loss_tolerance=.03, maxima=(8, 16, 32))
        self.assertLessEqual(result["selected"]["loss"], result["reference"]["loss"] + .03)
        self.assertLess(result["selected"]["cost_proxy"], 1.)

    def test_compile_fixed_output_and_gradients(self):
        model = self.model()
        self.assertTrue(training.parse_args(["--dry-run"]).compile)
        x, mask = self.input()
        eager = model.execute(x, mask, model.min_profile)
        eager.sum().backward()
        expected = model.classifier.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        model.enable_compile(True, backend="eager", limit=2)
        compiled = model.execute(x, mask, model.min_profile)
        compiled.sum().backward()
        torch.testing.assert_close(compiled, eager)
        torch.testing.assert_close(model.classifier.weight.grad, expected)
        self.assertEqual(len(model._compiled), 1)

    def test_inductor_options_disable_capture_even_for_reduce_overhead(self):
        model = self.model()
        model.enable_compile(True, mode="reduce-overhead")
        seen = []
        def compile_path(path, **options):
            seen.append(options)
            return path
        x, mask = self.input()
        with patch("torch.compile", side_effect=compile_path):
            model.execute(x, mask, model.min_profile).sum().backward()
        self.assertEqual(len(seen), 30)
        self.assertTrue(all(not options["options"]["triton.cudagraphs"] for options in seen))
        self.assertEqual(len(model._compiled_blocks), 30)

    def test_compiled_sandwich_accumulates_all_gradients(self):
        eager = self.model().train()
        compiled = self.model().train()
        compiled.load_state_dict(eager.state_dict())
        compiled.enable_compile(True, backend="eager")
        x, mask = self.input()
        labels = torch.tensor([0, 2])
        def run(model):
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            optimizer.zero_grad(set_to_none=True)
            for microbatch in range(2):
                for profile, use_compile, update in ((model.max_profile, True, True),
                                                      (model.min_profile, True, False),
                                                      ((8, 16, 16) * 3, False, False)):
                    logits = model.execute(x, mask, profile, compiled=use_compile, update_bn=update)
                    (torch.nn.functional.cross_entropy(logits, labels) / 6).backward()
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            optimizer.step()
            return gradients
        expected, actual = run(eager), run(compiled)
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], msg=name)
        for name, value in eager.state_dict().items():
            torch.testing.assert_close(value, compiled.state_dict()[name], msg=name)

    def test_end_to_end_stages_checkpoints_and_branch_reuse(self):
        data = SyntheticActions()
        def datasets(args):
            fit, search, selection, subjects = split_width_subjects(data.members, args.calibration_fraction, args.seed)
            return data, data, fit, search, selection, subjects, {}
        with tempfile.TemporaryDirectory() as folder:
            args = ["--device", "cpu", "--no-compile", "--no-teacher", "--console-only",
                    "--stage-maxima", "8", "16", "32", "--router-hidden", "8", "--num-classes", "3",
                    "--num-workers", "0", "--window-size", "5", "--max-persons", "1",
                    "--batch-size", "32", "--test-batch-size", "32", "--no-drop-last",
                    "--supernet-epochs", "1", "--gate-epochs", "1", "--route-epochs", "1",
                    "--search-steps", "1", "--bn-calibration-batches", "1", "--min-class-samples", "1",
                    "--calibration-fraction", ".3", "--save-dir", folder]
            with patch.object(training, "make_datasets", side_effect=datasets), patch.object(training, "RTMWZipDataset", return_value=data):
                training.main(args)
            directory = next(Path(folder).iterdir())
            restored = training.load_adaptive_checkpoint(directory / "best.pt", compile=False)
            self.assertTrue(bool(restored.profiles_ready))
            checkpoint = torch.load(directory / "best.pt", weights_only=False)
            self.assertEqual(checkpoint["phase"], "final")
            self.assertIsNotNone(checkpoint["val_accuracy"])
            for filename in ("class_widths.csv", "routing_calibration.json", "evaluation.json", "searched.pt"):
                self.assertTrue((directory / filename).is_file())
            self.assertEqual(len(json.loads((directory / "class_widths.json").read_text())["classes"]), 3)
            branch = load_adaptive_branch(directory / "branches/body_gcn.pt")
            x = torch.randn(2, 3, 5, 22)
            mask = torch.ones(2, 1, 5, 22, dtype=torch.bool)
            profile = tuple(restored.class_widths[0, :3].tolist())
            restored.apply_profile_bn(tuple(restored.class_widths[0].tolist()))
            with torch.no_grad():
                from isaa.models.backbones.slimmable import forward_slim
                expected = forward_slim(restored.body_encoder, x, mask, profile, kind="ctr")
                actual = branch(x, mask, class_index=0)
            torch.testing.assert_close(actual[0], expected[0])
            branch.calibrate_bn([(x, mask)], widths=(8, 8, 8))
            branch.save(directory / "finetuned_branch.pt")
            reloaded_branch = load_adaptive_branch(directory / "finetuned_branch.pt")
            torch.testing.assert_close(branch(x, mask, widths=(8, 8, 8))[0],
                                       reloaded_branch(x, mask, widths=(8, 8, 8))[0])
            # Detailed evaluator must handle a searched checkpoint without official metrics.
            from tools import evaluate_best
            evaluation_dir = directory / "detailed"
            argv = ["evaluate_best.py", "--checkpoint", str(directory / "searched.pt"),
                    "--device", "cpu", "--no-compile", "--num-workers", "0",
                    "--output-dir", str(evaluation_dir)]
            with patch.object(sys, "argv", argv), patch.object(evaluate_best, "RTMWZipDataset", return_value=data), redirect_stdout(io.StringIO()):
                evaluate_best.main()
            detail = json.loads((evaluation_dir / "summary.json").read_text())
            self.assertIsNone(detail["recorded_top1_percent"])
            self.assertIn("fallback", detail["mean_routing"])
            # A search-only continuation uses the identical subject partition.
            with patch.object(training, "make_datasets", side_effect=datasets):
                training.main(args + ["--stage", "search", "--init-checkpoint", str(directory / "best.pt"), "--search-steps", "0"])


if __name__ == "__main__":
    unittest.main()
