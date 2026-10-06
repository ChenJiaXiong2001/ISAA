"""Behavioral checks for local skeleton structure and main-node coordination."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from isaa.data.transforms import build_skeleton_feature_channels
from isaa.models.rtmw_local_ctr import (
    AuxiliarySkeleton, FaceTokenCompression, FixedSkeletonConv, MainNodeCTR, RegionalDetailPool,
    RTMWLocalCTR, _downsample_mask,
)
from isaa.train import collate_rtmw, parse_args, run_epoch
from isaa.utils.experiment import RunRecords


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
            self.assertEqual(layer.adjacency.shape, (3, 71, 71))
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
        x = torch.zeros(1, 3, 1, 71)
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
                self.assertGreater(self.model.auxiliary_to_main.weight.grad.abs().sum().item(), 0)
                self.assertGreater(self.model.auxiliary.temporal.weight.grad.abs().sum().item(), 0)
                self.assertGreater(self.model.regional_pool.score.weight.grad.abs().sum().item(), 0)
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
                self.assertEqual(out["auxiliary_token_features"].shape, (2, 2, 16, 5, 71))
                self.assertEqual(out["auxiliary_token_mask"].shape, (2, 2, 1, 5, 71))
                expanded = out["auxiliary_token_features"].index_select(-1, out["auxiliary_original_to_token"])
                torch.testing.assert_close(out["auxiliary_node_features"], expanded)
            else:
                self.assertIsNone(out["auxiliary_node_features"])
                self.assertIsNone(out["auxiliary_token_features"])
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
                         [48, 48, 48, 48, 96, 96, 96, 192, 192, 192])
        with torch.no_grad():
            out = model(self.x[:1, :, :1, :, :1], return_node_features=True)
        self.assertEqual(out["node_features"].shape, (1, 1, 192, 1, 32))
        self.assertTrue(torch.isfinite(out["logits"]).all())

    def test_width_presets_preserve_graphs_auxiliary_and_strides(self):
        compact = RTMWLocalCTR(num_classes=6)
        standard = RTMWLocalCTR(num_classes=6, backbone_width="standard")
        self.assertEqual(standard.channels, RTMWLocalCTR.STANDARD_CHANNELS)
        self.assertLess(sum(p.numel() for p in compact.parameters()),
                        sum(p.numel() for p in standard.parameters()))
        self.assertNotEqual(compact.experiment_name, standard.experiment_name)
        for name in ("main_joint_indices", "joint_to_main", "joint_graph", "main_graph"):
            torch.testing.assert_close(getattr(compact, name), getattr(standard, name))
        self.assertEqual([block.stride for block in compact.blocks],
                         [block.stride for block in standard.blocks])
        self.assertEqual([tuple(p.shape) for p in compact.auxiliary.parameters()],
                         [tuple(p.shape) for p in standard.auxiliary.parameters()])
        self.assertEqual(compact.auxiliary_to_main.out_channels, 48)
        self.assertEqual(standard.auxiliary_to_main.out_channels, 64)

    def test_compact_training_and_checkpoint_roundtrip(self):
        model = RTMWLocalCTR(num_classes=6).train()
        loss = nn.functional.cross_entropy(model(self.x), torch.tensor([0, 4]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.auxiliary_to_main.weight.grad.abs().sum().item(), 0)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        model.eval()
        restored = RTMWLocalCTR(num_classes=6, backbone_width="compact").eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(restored(self.x), model(self.x))

    def test_entry_width_selection(self):
        with patch("sys.argv", ["main.py"]):
            args = parse_args()
            self.assertEqual(args.backbone_width, "standard")
            self.assertEqual(args.model_variant, "body-local")
            self.assertFalse(args.main_only)
            self.assertEqual(args.archive, "data/ntu60_skeletons_rtmw.zip")
            self.assertEqual(args.split, "xsub60")
            self.assertEqual((args.batch_size, args.test_batch_size, args.epochs), (32, 32, 65))
            self.assertEqual((args.lr, args.weight_decay, args.momentum), (0.1, 0.0004, 0.9))
            self.assertEqual((args.warmup_epochs, args.lr_steps), (5, [35, 55]))
            self.assertTrue(args.nesterov)
        with patch("sys.argv", ["main.py", "--backbone-width", "standard"]):
            self.assertEqual(parse_args().backbone_width, "standard")
        with patch("sys.argv", ["main.py", "--model-variant", "isaa"]):
            legacy = parse_args()
            self.assertTrue(legacy.main_only)
            self.assertEqual((legacy.feature_mode, legacy.node_count), ("isaa", 32))
        with self.assertRaises(ValueError):
            RTMWLocalCTR(backbone_width="unknown")

    def test_main_only_has_no_auxiliary_and_accepts_selected_nodes(self):
        model = RTMWLocalCTR(num_classes=6, channels=(8, 8), main_only=True).eval()
        self.assertFalse(model.fine_enabled)
        for name in ("auxiliary", "face_compression", "regional_pool", "auxiliary_to_main", "auxiliary_scale"):
            self.assertFalse(hasattr(model, name))
        selected = self.x.index_select(3, model.main_joint_indices)
        torch.testing.assert_close(model(selected), model(self.x))
        changed = self.x.clone()
        detail = torch.ones(133, dtype=torch.bool)
        detail[model.main_joint_indices] = False
        changed[:, :, :, detail] = float("nan")
        torch.testing.assert_close(model(changed), model(selected))
        self.assertIsNone(model(selected, return_node_features=True)["auxiliary_node_features"])
        with self.assertRaises(ValueError):
            model.set_fine_enabled(True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, nesterov=True)
        model.train()
        loss = nn.functional.cross_entropy(model(selected), torch.tensor([0, 4]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        restored = RTMWLocalCTR(num_classes=6, channels=(8, 8), main_only=True).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(selected), model.eval()(selected))

    def test_worker_selection_and_batch_records_match_epoch_metrics(self):
        model = RTMWLocalCTR(num_classes=6, channels=(8,), main_only=True)
        features = torch.randn(3, 5, 4, 133, 1)
        features[:, 4] = 1
        labels = torch.tensor([0, 3, 5])
        masks = torch.ones(3, 4, dtype=torch.bool)
        from functools import partial
        collate = partial(collate_rtmw, main_indices=model.main_joint_indices)
        loader = DataLoader(TensorDataset(features, labels, masks), batch_size=2, collate_fn=collate)
        selected, _, _ = next(iter(loader))
        self.assertEqual(selected.shape, (2, 3, 4, 32, 1))
        torch.testing.assert_close(selected, features[:2, [0, 1, 4]].index_select(3, model.main_joint_indices))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            records = RunRecords(path, {}, path)
            try:
                with patch("isaa.train.write_progress_lines"), patch("isaa.train.finish_progress_lines"):
                    metrics = run_epoch(model, loader, torch.device("cpu"), records=records, epoch=2)
            finally:
                records.close()
            rows = [json.loads(line) for line in (path / "batches.jsonl").read_text().splitlines()]
            self.assertEqual([row["samples"] for row in rows], [2, 1])
            self.assertEqual([row["global_step"] for row in rows], [3, 4])
            for key in ("loss", "top1", "top5"):
                expected = sum(row[key] * row["samples"] for row in rows) / 3
                self.assertAlmostEqual(metrics[key], expected, places=6)

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

    def test_auxiliary_switch_changes_main_features_through_early_fusion(self):
        self.model.eval()
        enabled = self.model(self.x, return_node_features=True)
        self.model.set_fine_enabled(False)
        disabled = self.model(self.x, return_node_features=True)
        self.assertFalse(torch.allclose(enabled["node_features"], disabled["node_features"]))
        self.assertFalse(torch.allclose(enabled["logits"], disabled["logits"]))

    def test_sparse_fixed_graph_matches_dense_forward_and_gradients(self):
        graph = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                              [[0., .25, .75], [0., 0., 0.], [0., 1., 0.]],
                              [[0., 0., 0.], [.5, 0., .5], [1., 0., 0.]]])
        sparse = FixedSkeletonConv(2, 4, graph)
        dense = copy.deepcopy(sparse)
        x = torch.randn(2, 2, 5, 3, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual = sparse(x)
        projected = dense.projection(reference_x).reshape(2, 3, 4, 5, 3)
        expected = torch.einsum("bpctv,puv->bctu", projected, graph)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
        torch.testing.assert_close(sparse.projection.weight.grad, dense.projection.weight.grad)
        state = sparse.state_dict()
        state["adjacency"] = torch.zeros_like(graph)
        state["adjacency"][0, 0, 2] = 2
        sparse.load_state_dict(state)
        self.assertEqual(sparse.edge_weights.numel(), 1)
        projected = sparse.projection(x).reshape(2, 3, 4, 5, 3)
        expected = torch.einsum("bpctv,puv->bctu", projected, state["adjacency"])
        torch.testing.assert_close(sparse(x), expected)

    def test_region_pool_uses_valid_counts_and_does_not_mix_regions(self):
        pool = RegionalDetailPool(1, torch.tensor([0, 0, 0, 1]), 3)
        with torch.no_grad():
            pool.score.weight.zero_()
        x = torch.tensor([[[[2., 999., 6., 10.], [4., 8., 999., 20.]]]], requires_grad=True)
        mask = torch.tensor([[[[True, False, True, True], [True, True, False, False]]]])
        values, valid = pool(x, mask)
        expected = torch.tensor([[[[4., 10., 0.], [6., 0., 0.]],
                                  [[4., 10., 0.], [6., 0., 0.]]]])
        torch.testing.assert_close(values, expected)
        torch.testing.assert_close(valid, torch.tensor([[[[True, True, False], [True, False, False]]]]))
        changed = x.detach().clone()
        changed[..., 3] += 100
        torch.testing.assert_close(pool(changed, mask)[0][..., 0], values[..., 0])
        values.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad[mask] > 0).all())
        self.assertTrue((x.grad[~mask] == 0).all())
        self.assertTrue(torch.isfinite(pool.score.weight.grad).all())

    def test_region_attention_learns_and_empty_regions_stay_finite(self):
        pool = RegionalDetailPool(1, torch.tensor([0, 0, 1]), 2)
        x = torch.tensor([[[[1., 3., 999.]]]], requires_grad=True)
        mask = torch.tensor([[[[True, True, False]]]])
        with torch.no_grad():
            pool.score.weight.fill_(1)
        values, valid = pool(x, mask)
        self.assertGreater(values[0, 1, 0, 0].item(), values[0, 0, 0, 0].item())
        self.assertEqual(values[..., 1].abs().sum().item(), 0)
        values.sum().backward()
        self.assertGreater(pool.score.weight.grad.abs().sum().item(), 0)
        empty, empty_mask = pool(x, torch.zeros_like(mask))
        self.assertEqual(empty.abs().sum().item(), 0)
        self.assertFalse(empty_mask.any())
        empty.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_auxiliary_temporal_order_and_residual(self):
        auxiliary = AuxiliarySkeleton(torch.eye(2).repeat(3, 1, 1), 2).eval()
        with torch.no_grad():
            for layer in auxiliary.layers:
                layer.projection.weight.fill_(.1)
            auxiliary.temporal.weight.zero_()
            auxiliary.temporal.weight[:, :, 1, 0] = 1
            auxiliary.temporal_mix.weight.fill_(.5)
        x = torch.rand(1, 3, 6, 2)
        mask = torch.ones(1, 1, 6, 2, dtype=torch.bool)
        self.assertEqual(auxiliary.temporal.groups, 2)
        actual = auxiliary(x, mask)
        reversed_back = auxiliary(x.flip(2), mask).flip(2)
        self.assertFalse(torch.allclose(actual, reversed_back))
        with torch.no_grad():
            auxiliary.temporal.weight.zero_()
        reference = x
        for layer, norm in zip(auxiliary.layers, auxiliary.norms):
            reference = torch.relu(norm(layer(reference), mask))
        torch.testing.assert_close(auxiliary(x, mask), reference)

    def test_missing_center_still_receives_valid_regional_details(self):
        self.model.eval()
        x = self.x.clone()
        x[:, 2, :, self.model.main_joint_indices] = 0
        output = self.model(x, return_node_features=True)
        self.assertTrue(output["regional_mask"].any())
        self.assertTrue(output["node_mask"].any())
        self.assertGreater(output["node_features"].abs().sum().item(), 0)
        self.model.set_fine_enabled(False)
        torch.testing.assert_close(self.model(x), self.model.classifier.bias.expand(2, -1))

    def test_detail_fuses_once_after_first_block_with_temporal_regions(self):
        self.model.eval()
        captured = {}
        hooks = [
            self.model.blocks[0].register_forward_hook(
                lambda module, args, out: captured.update(before=out[0].detach().clone())),
            self.model.blocks[1].register_forward_pre_hook(
                lambda module, args: captured.update(after=args[0].detach().clone())),
        ]
        out = self.model(self.x, return_node_features=True)
        for hook in hooks:
            hook.remove()
        self.assertEqual(out["regional_features"].shape, (2, 2, 32, 5, 32))
        self.assertEqual(out["regional_mask"].shape, (2, 2, 1, 5, 32))
        regional = out["regional_features"].reshape(4, 32, 5, 32)
        expected = captured["before"] + self.model.auxiliary_scale * self.model.auxiliary_to_main(regional)
        torch.testing.assert_close(captured["after"], expected)

    def test_face_compression_raw_means_masks_and_gradients(self):
        compress = self.model.face_compression
        x = self.x[..., 0].clone().requires_grad_()
        mask = torch.ones(2, 1, 5, 133, dtype=torch.bool)
        mask[0, :, 1, 23:91] = False
        mask[1, :, :, 23] = False
        compact, valid = compress(x, mask)
        self.assertEqual(compact.shape, (2, 3, 5, 71))
        torch.testing.assert_close(compact[..., :65], x.index_select(-1, compress.nonface_indices))
        for index, group in enumerate(FaceTokenCompression.FACE_GROUPS):
            weights = mask[..., list(group)]
            expected = x[..., list(group)].masked_fill(~weights, 0).sum(-1) / weights.sum(-1).clamp_min(1)
            torch.testing.assert_close(compact[..., 65 + index], expected)
        self.assertFalse(valid[0, :, 1, 65:].any())
        self.assertEqual(compact[0, :, 1, 65:].abs().sum().item(), 0)
        compact.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad[mask.expand_as(x)] > 0).all())
        self.assertTrue((x.grad[~mask.expand_as(x)] == 0).all())

    def test_compacted_graph_mapping_and_nonface_edges(self):
        compress = self.model.face_compression
        self.assertEqual(compress.original_to_token.unique().numel(), 71)
        graph = compress.adjacency
        torch.testing.assert_close(graph[0], torch.eye(71))
        for part in (1, 2):
            self.assertEqual(graph[part].diagonal().abs().sum().item(), 0)
            degree = graph[part].sum(-1)
            torch.testing.assert_close(degree[degree > 0], torch.ones_like(degree[degree > 0]))
        expected = self.model.joint_graph.index_select(1, compress.nonface_indices).index_select(2, compress.nonface_indices)
        torch.testing.assert_close(graph[:, :65, :65], expected)
        torch.testing.assert_close(compress.token_owners[:65], self.model.joint_to_main[compress.nonface_indices])
        self.assertTrue((compress.token_owners[65:] == self.model.joint_to_main[23]).all())
        # Original chain contracts to five inter-token links in each direction.
        self.assertEqual(torch.count_nonzero(graph[1, 65:, 65:]).item(), 5)
        self.assertEqual(torch.count_nonzero(graph[2, 65:, 65:]).item(), 5)

    def test_face_compression_precedes_every_auxiliary_projection(self):
        self.model.eval()
        seen = []
        def capture(name):
            def hook(module, args):
                seen.append((name, args[0].shape[-1]))
            return hook
        hooks = [layer.projection.register_forward_pre_hook(capture("projection"))
                 for layer in self.model.auxiliary.layers]
        hooks += [norm.register_forward_pre_hook(capture("graph_norm"))
                  for norm in self.model.auxiliary.norms]
        hooks += [self.model.auxiliary.temporal.register_forward_pre_hook(capture("body_time")),
                  self.model.auxiliary.face_temporal.register_forward_pre_hook(capture("face_time")),
                  self.model.regional_pool.register_forward_pre_hook(capture("regions"))]
        with torch.no_grad():
            logits = self.model(self.x)
            exported = self.model(self.x, return_node_features=True)
        for hook in hooks:
            hook.remove()
        expected = [("projection", 71), ("graph_norm", 71), ("projection", 71),
                    ("graph_norm", 71), ("body_time", 65), ("face_time", 6), ("regions", 71)]
        self.assertEqual(seen, expected * 2)
        torch.testing.assert_close(logits, exported["logits"])

    def test_missing_face_does_not_update_face_statistics(self):
        model = copy.deepcopy(self.model).train()
        x = self.x.clone()
        x[:, 2, :, 23:91] = 0
        before = {name: value.clone() for name, value in model.auxiliary.face_norm.named_buffers()}
        out = model(x, return_node_features=True)
        self.assertFalse(out["auxiliary_token_mask"][..., 65:].any())
        self.assertEqual(out["auxiliary_token_features"][..., 65:].abs().sum().item(), 0)
        for name, value in model.auxiliary.face_norm.named_buffers():
            torch.testing.assert_close(value, before[name])
        out["logits"].square().sum().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

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
