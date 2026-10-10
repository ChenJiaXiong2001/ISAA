"""Verify full backbones, best-model feature parity, and reusable checkpoints."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from isaa.models.backbones import CTRGCNBackbone, STGCNBackbone
from isaa.models.body_local_full_gcn import FULL_GCN_VARIANT, BodyLocalFullGCNFusion
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.original_ctrgcn import OriginalCTRGCN
from isaa.models.official_stgcn import ConvTemporalGraphical
from isaa.train import parse_args


class FullGCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def input(self, channels=3, time=9):
        x = torch.randn(2, channels, time, 133, 2) * .01
        x[:, 2] = 1
        return x

    def test_full_structure_and_gradients(self):
        model = BodyLocalFullGCNFusion(7)
        self.assertEqual(model.CHANNELS, OriginalCTRGCN.STANDARD_CHANNELS)
        self.assertEqual(model.HAND_CHANNELS, model.CHANNELS)
        self.assertEqual(model.LOCAL_CHANNELS, model.CHANNELS)
        self.assertEqual(len(model.body_encoder.blocks), 10)
        self.assertEqual(len(model.hand_encoder.blocks), 10)
        self.assertEqual(len(model.face_encoder.st_gcn_networks), 10)
        self.assertFalse(hasattr(model, "local_blocks"))
        self.assertGreater(model.hand_encoder.A[1:, 0, 1:].count_nonzero(), 0)
        self.assertGreater(model.face_encoder.A[1:].count_nonzero(), 0)
        for encoder in (model.body_encoder, model.hand_encoder):
            self.assertEqual(encoder.strides, (1, 1, 1, 1, 2, 1, 1, 2, 1, 1))
            for block in encoder.blocks:
                self.assertEqual(len(block.gcn1.convs), 3)
                self.assertEqual(block.tcn1.num_branches, 4)
                self.assertTrue(block.gcn1.adaptive)
        for block in model.face_encoder.st_gcn_networks:
            self.assertIsInstance(block.gcn, ConvTemporalGraphical)
            self.assertEqual(block.gcn.conv.kernel_size, (1, 1))
            self.assertEqual(block.tcn[2].kernel_size, (9, 1))
        logits = model(self.input())
        self.assertEqual(logits.shape, (2, 7))
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.hand_encoder.blocks[0].gcn1.convs[0].conv3.weight.grad.abs().sum(), 0)
        self.assertGreater(model.face_encoder.edge_importance[0].grad.abs().sum(), 0)

    def test_best_model_raw_features_are_preserved(self):
        old = BodyLocalHandCTRWideRelativeFusion(7).eval()
        new = BodyLocalFullGCNFusion(7).eval()
        x = self.input()
        old_seen, new_seen = [], []
        old_hook = old.hand_ctr_blocks[0].register_forward_pre_hook(lambda _, args: old_seen.append(args[0]))
        new_hook = new.hand_encoder.data_bn.register_forward_pre_hook(lambda _, args: new_seen.append(args[0]))
        with torch.no_grad():
            old(x)
            new(x)
        old_hook.remove()
        new_hook.remove()
        self.assertEqual(len(new_seen), 2)
        for hand, flat in zip(old_seen, new_seen):
            b, c, t, v = hand.shape
            torch.testing.assert_close(hand.permute(0, 3, 1, 2).reshape(b, v * c, t), flat)

    def test_missing_nodes_odd_frames_and_checkpoint(self):
        model = BodyLocalFullGCNFusion(7).eval()
        x = self.input(8)
        x[..., 1] = float("nan")
        x[:, 2, :, 91:] = 0
        frame_mask = torch.ones(2, 9, dtype=torch.bool)
        frame_mask[:, -2:] = False
        with torch.no_grad():
            expected = model(x, frame_mask)
            self.assertTrue(torch.isfinite(expected).all())
            empty = model(torch.full_like(x, float("nan")), frame_mask)
            self.assertTrue(torch.isfinite(empty).all())
            one = model(x[:, :, :1], frame_mask[:, :1])
            self.assertEqual(one.shape, (2, 7))
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "config": model.backbone_config}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = BodyLocalFullGCNFusion(7, backbone_config=checkpoint["config"]).eval()
        restored.load_state_dict(checkpoint["model"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(x, frame_mask))

    def test_custom_backbones_and_all_valid_reference_parity(self):
        a = torch.eye(6).repeat(3, 1, 1)
        x = torch.randn(2, 6, 9, 6)
        mask = torch.ones(2, 1, 9, 6, dtype=torch.bool)
        ctr = CTRGCNBackbone(a, in_channels=6, channels=(8, 16), strides=(1, 2)).eval()
        st = STGCNBackbone(6, a, channels=(8, 16), strides=(1, 2)).eval()
        for encoder in (ctr, st):
            with torch.no_grad():
                reference, time = encoder(x.unsqueeze(-1))
                masked, out_mask = encoder.forward_masked(x, mask)
            torch.testing.assert_close(reference, masked)
            self.assertEqual(time, 5)
            self.assertEqual(out_mask.shape, (2, 1, 5, 6))
        config = {"hand": {"channels": [8] * 10}, "face": {"channels": [16] * 10, "dropout": .2}}
        model = BodyLocalFullGCNFusion(7, backbone_config=config).eval()
        self.assertEqual(model.hand_to_local.out_channels, 16)
        with torch.no_grad():
            self.assertEqual(model(self.input()).shape, (2, 7))
        with self.assertRaises(ValueError):
            CTRGCNBackbone(a, channels=(8, 10), strides=(1, 2))

    def test_entry_config_is_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcn.json"
            path.write_text(json.dumps({"face": {"dropout": .2}}))
            with patch.object(sys, "argv", ["main.py", "--gcn-config", str(path)]):
                args = parse_args()
            self.assertEqual(args.model_variant, FULL_GCN_VARIANT)
            self.assertEqual((args.node_count, args.feature_mode, args.main_only), (133, "raw", False))
            json.dumps(vars(args))


if __name__ == "__main__":
    unittest.main()
