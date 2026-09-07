"""Behavioral checks for local skeleton structure and main-node coordination."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from isaa.data.transforms import build_skeleton_feature_channels
from isaa.models.rtmw_local_ctr import FixedSkeletonConv, MainNodeCTR, RTMWLocalCTR, _downsample_mask
from isaa.train import collate_rtmw, parse_args, run_epoch


class RTMWLocalCTRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)
        self.model = RTMWLocalCTR(num_classes=6, channels=(8, 8))
        self.x = torch.randn(2, 3, 5, 133, 2)
        self.x[:, 2] = 1

    def test_fixed_local_and_only_32_node_dynamic_graphs(self):
        self.assertTrue(self.model.fine_enabled)
        self.assertEqual(self.model.main_joint_indices.unique().numel(), 32)
        torch.testing.assert_close(self.model.joint_to_main[self.model.main_joint_indices], torch.arange(32))
        self.assertEqual(len(self.model.auxiliary.layers), 2)
        for layer in self.model.auxiliary.layers:
            self.assertIsInstance(layer, FixedSkeletonConv)
            self.assertEqual(layer.adjacency.shape, (3, 133, 133))
            self.assertNotIn("adjacency", dict(layer.named_parameters()))
            self.assertEqual(layer.out_channels, 16)
        for block in self.model.blocks:
            self.assertEqual(block.gcn.static_topology.shape, (3, 32, 32))
            self.assertEqual(len(block.gcn.branches), 3)
            self.assertFalse(any(isinstance(module, FixedSkeletonConv) for module in block.modules()))

    def test_non_neighbor_main_nodes_can_exchange_information(self):
        branch = MainNodeCTR(1, 1).eval()
        adjacency = torch.eye(32)
        with torch.no_grad():
            branch.feature_proj.weight.fill_(1)
            branch.feature_proj.bias.zero_()
            branch.theta.weight.zero_()
            branch.theta.bias.zero_()
            branch.phi.weight.zero_()
            branch.phi.weight[0, 0] = 1
            branch.phi.bias.zero_()
            branch.relation_proj.weight.zero_()
            branch.relation_proj.weight[0, 0] = -1
            branch.relation_proj.bias.zero_()
        x = torch.zeros(1, 1, 2, 32)
        x[:, :, :, 20] = 1
        mask = torch.ones(1, 1, 2, 32, dtype=torch.bool)
        fixed = branch(x, mask, adjacency, 0)
        self.assertEqual(fixed[0, 0, 0, 3].item(), 0)
        learned = branch(x, mask, adjacency, 1)
        self.assertGreater(learned[0, 0, 0, 3].item(), 0)
        learned.sum().backward()
        self.assertGreater(branch.relation_proj.weight.grad.abs().sum().item(), 0)

    def test_local_edges_exclude_distant_nodes(self):
        local = self.model.auxiliary.layers[0]
        with torch.no_grad():
            local.projection.weight.fill_(1)
        x = torch.zeros(1, 3, 1, 133)
        x[:, 0, :, 7] = 1
        out = local(x)
        self.assertGreater(out[:, :, :, 5].abs().sum().item(), 0)
        self.assertEqual(out[:, :, :, 16].abs().sum().item(), 0)

    def test_stage_switch_uses_detail_only_after_warmup(self):
        self.model.set_fine_enabled(False)
        self.model.eval()
        changed = self.x.clone()
        detail = torch.ones(133, dtype=torch.bool)
        detail[self.model.main_joint_indices] = False
        changed[:, :2, :, detail] += 3
        calls = []
        hooks = [layer.register_forward_hook(lambda *args: calls.append(1))
                 for layer in self.model.auxiliary.layers]
        coarse = self.model(self.x)
        torch.testing.assert_close(self.model(changed), coarse)
        self.assertEqual(calls, [])
        head = self.model.classifier
        self.model.set_fine_enabled(True)
        fine = self.model(self.x)
        different = self.model(changed)
        for hook in hooks:
            hook.remove()
        self.assertIs(head, self.model.classifier)
        self.assertEqual(len(calls), 4)
        self.assertFalse(torch.allclose(fine, different))

    def test_training_both_stages_and_fixed_graph_stays_fixed(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        original = [layer.adjacency.clone() for layer in self.model.auxiliary.layers]
        for fine in (False, True):
            self.model.set_fine_enabled(fine)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(self.model(self.x), torch.tensor([0, 4]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            grad = self.model.auxiliary.layers[0].projection.weight.grad
            self.assertEqual(grad is not None, fine)
            if fine:
                self.assertGreater(grad.abs().sum().item(), 0)
                self.assertGreater(self.model.auxiliary_projection.weight.grad.abs().sum().item(), 0)
            self.assertIsNotNone(self.model.blocks[0].gcn.static_topology.grad)
            optimizer.step()
        for expected, layer in zip(original, self.model.auxiliary.layers):
            torch.testing.assert_close(expected, layer.adjacency)

    def test_masks_empty_tracks_padding_and_running_stats(self):
        for fine in (False, True):
            for training in (False, True):
                with self.subTest(fine=fine, training=training):
                    model = copy.deepcopy(self.model).train(training)
                    model.set_fine_enabled(fine)
                    padded_model = copy.deepcopy(model)
                    expected = model(self.x)
                    padded = torch.cat((self.x, torch.randn_like(self.x)), dim=2)
                    padded = torch.cat((padded, torch.zeros_like(padded[..., :1])), dim=-1)
                    frames = torch.tensor([[True] * 5 + [False] * 5] * 2)
                    torch.testing.assert_close(padded_model(padded, frames), expected, rtol=1e-4, atol=1e-5)
                    for name, tensor in model.named_buffers():
                        torch.testing.assert_close(dict(padded_model.named_buffers())[name], tensor,
                                                   rtol=1e-4, atol=1e-5)
        self.model.train()
        expected_buffers = {k: v.clone() for k, v in self.model.named_buffers()}
        out = self.model(torch.zeros_like(self.x))
        torch.testing.assert_close(out, self.model.classifier.bias.expand(2, -1))
        for name, tensor in self.model.named_buffers():
            torch.testing.assert_close(tensor, expected_buffers[name])

    def test_no_main_observations_during_coarse_stage(self):
        self.model.set_fine_enabled(False)
        x = self.x.clone()
        x[:, 2, :, self.model.main_joint_indices] = 0
        torch.testing.assert_close(self.model(x), self.model.classifier.bias.expand(2, -1))

    def test_temporal_order_matters(self):
        self.model.eval()
        self.model.set_fine_enabled(True)
        self.assertFalse(torch.allclose(self.model(self.x), self.model(self.x.flip(2)), atol=1e-7, rtol=1e-6))

    def test_node_outputs_and_checkpoint_stage_roundtrip(self):
        self.model.eval()
        for fine in (False, True):
            self.model.set_fine_enabled(fine)
            out = self.model(self.x, return_node_features=True)
            self.assertEqual(out["node_features"].shape, (2, 2, 8, 5, 32))
            torch.testing.assert_close(out["node_indices"], self.model.main_joint_indices)
            torch.testing.assert_close(out["time_indices"], torch.arange(5))
            if fine:
                self.assertEqual(out["auxiliary_node_features"].shape, (2, 2, 16, 5, 133))
                self.assertEqual(out["auxiliary_node_mask"].shape, (2, 2, 1, 5, 133))
                torch.testing.assert_close(out["auxiliary_node_indices"], torch.arange(133))
            else:
                self.assertIsNone(out["auxiliary_node_features"])
            restored = RTMWLocalCTR(num_classes=6, channels=(8, 8)).eval()
            restored.load_state_dict(self.model.state_dict())
            self.assertEqual(restored.fine_enabled, fine)
            torch.testing.assert_close(restored(self.x), out["logits"])

    def test_backbone_and_time_convolutions_only_see_32_nodes(self):
        model = RTMWLocalCTR(num_classes=6, channels=(8,) * 10).eval()
        seen = []
        hooks = [block.temporal.register_forward_pre_hook(
            lambda module, args: seen.append((args[0].size(2), args[0].size(3)))) for block in model.blocks]
        out = model(self.x, return_node_features=True)
        for hook in hooks:
            hook.remove()
        self.assertEqual(seen, [(5, 32)] * 5 + [(3, 32)] * 3 + [(2, 32)] * 2)
        self.assertEqual(out["node_features"].shape, (2, 2, 8, 2, 32))
        self.assertEqual(out["auxiliary_node_features"].shape, (2, 2, 16, 5, 133))
        torch.testing.assert_close(out["time_indices"], torch.tensor([0, 4]))
        self.assertFalse(model.blocks[0].use_residual)
        for block in model.blocks:
            temporal = block.temporal
            self.assertEqual([layer.conv.dilation for layer in temporal.dilated], [(1, 1), (2, 1)])
            self.assertEqual([layer.conv.kernel_size for layer in temporal.dilated], [(5, 1), (5, 1)])
            self.assertTrue(all(layer.conv.out_channels == 2 for layer in temporal.projections))

    def test_default_channels_and_single_frame(self):
        model = RTMWLocalCTR(num_classes=6).eval()
        self.assertEqual([block.gcn.norm.bn.num_features for block in model.blocks],
                         [64, 64, 64, 64, 128, 128, 128, 256, 256, 256])
        with torch.no_grad():
            out = model(self.x[:1, :, :1, :, :1], return_node_features=True)
        self.assertEqual(out["node_features"].shape, (1, 1, 256, 1, 32))
        self.assertTrue(torch.isfinite(out["logits"]).all())

    def test_strided_masks_padding_and_missing_odd_frames(self):
        mask = torch.tensor([False, True, False, False, True]).view(1, 1, 5, 1)
        torch.testing.assert_close(_downsample_mask(mask, 2).flatten(), torch.tensor([True, False, True]))
        for training in (False, True):
            model = RTMWLocalCTR(num_classes=6, channels=(8,) * 10).train(training)
            padded_model = copy.deepcopy(model)
            expected = model(self.x)
            padded = torch.cat((self.x, torch.randn_like(self.x)), dim=2)
            frames = torch.tensor([[True] * 5 + [False] * 5] * 2)
            torch.testing.assert_close(padded_model(padded, frames), expected, rtol=2e-4, atol=2e-5)

    def test_auxiliary_switch_leaves_main_features_unchanged(self):
        self.model.eval()
        enabled = self.model(self.x, return_node_features=True)
        self.model.set_fine_enabled(False)
        disabled = self.model(self.x, return_node_features=True)
        torch.testing.assert_close(enabled["node_features"], disabled["node_features"])
        self.assertFalse(torch.allclose(enabled["logits"], disabled["logits"]))

    def test_entry_auxiliary_options_and_legacy_alias(self):
        for flag in ("--aux-start-epoch", "--fine-start-epoch"):
            with patch("sys.argv", ["main.py", flag, "5", "--auxiliary-channels", "8"]):
                args = parse_args()
            self.assertEqual(args.fine_start_epoch, 5)
            self.assertEqual(args.auxiliary_channels, 8)

    def test_relative_coordinate_translation_and_scaling(self):
        raw = torch.randn(3, 5, 133, 2)
        raw[2] = 1
        moved = raw.clone()
        moved[:2] = moved[:2] * 3 + 20
        expected = build_skeleton_feature_channels(raw, layout="rtmw_133", score_index=2)
        actual = build_skeleton_feature_channels(moved, layout="rtmw_133", score_index=2)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_training_entry_metrics_and_two_stages(self):
        features = torch.randn(3, 5, 4, 133, 1)
        features[:, 4] = 1
        labels = torch.tensor([0, 3, 5])
        mask = torch.ones(3, 4, dtype=torch.bool)
        loader = DataLoader(TensorDataset(features, labels, mask), batch_size=2, collate_fn=collate_rtmw)
        optimizer = torch.optim.Adam(self.model.parameters())
        with patch("isaa.train.write_progress_lines"), \
                patch("isaa.train.finish_progress_lines"):
            for fine in (False, True):
                self.model.set_fine_enabled(fine)
                run_epoch(self.model, loader, torch.device("cpu"), optimizer)
                actual = run_epoch(self.model, loader, torch.device("cpu"))
                with torch.no_grad():
                    logits = self.model(features[:, [0, 1, 4]], mask)
                    expected = nn.functional.cross_entropy(logits, labels)
                self.assertAlmostEqual(actual["loss"], expected.item(), places=5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_backward(self):
        self.model.cuda()
        for fine in (False, True):
            self.model.set_fine_enabled(fine)
            loss = self.model(self.x.cuda()).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()


if __name__ == "__main__":
    unittest.main()
