import contextlib
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
from zipfile import ZipFile

import numpy as np
import torch

from isaa.data.skelet_dataset import (SkeleTRTMWDataset, SkeleTPoseDataset, preprocess_keypoints,
                                      uniform_indices, bone_parents)
from isaa.models.skelet import SkeleTGCN
from isaa.models.skelet.backbone import SkeleT_GCN
from isaa.models.skelet.init_mapping_matrices import mapping_matrices
from isaa.models.skelet.instance_pooling import InstancePoolingModule
from isaa.models.skelet.model import backbone_config
from isaa.skelet_training import parse_args, compile_model, main, make_loader, RepeatDataset
from tools.fuse_skelet_scores import fuse


class SkeleTTests(unittest.TestCase):
    def test_semantic_partition(self):
        m = mapping_matrices["J65toJ27"]
        self.assertEqual(tuple(m.shape), (65, 27))
        torch.testing.assert_close(m.sum(1) > 0, torch.ones(65, dtype=torch.bool))
        torch.testing.assert_close(m.sum(0), torch.ones(27))
        self.assertEqual(m[9, 9], 0.5)  # Body and detailed left wrist.
        self.assertEqual(m[23, 9], 0.5)
        torch.testing.assert_close(m[24:28, 17], torch.full((4,), 0.25))
        self.assertEqual(mapping_matrices["J27toJ11"][9, 4], torch.tensor(0.1667))

    def test_stage_shapes_and_gradients(self):
        for variant in ("dgstgcn", "stgcn++", "ctrgcn", "ctrgcn-official-config"):
            with self.subTest(variant=variant):
                model = SkeleTGCN(7, variant)
                shapes = []
                handles = [b.register_forward_hook(lambda _, __, y: shapes.append(tuple(y.shape)))
                           for b in model.backbone.gcn]
                x = torch.randn(2, 3, 16, 65, 2)
                y = model(x)
                y.square().sum().backward()
                for handle in handles:
                    handle.remove()
                self.assertEqual(tuple(y.shape), (2, 7))
                self.assertEqual([s[-1] for s in shapes], [65]*4 + [27]*3 + [11]*3)
                self.assertEqual([s[2] for s in shapes], [16]*4 + [8]*3 + [4]*3)
                self.assertEqual([s[1] for s in shapes], [64]*4 + [128]*3 + [256]*3)
                for index in (4, 7):
                    grad = model.backbone.gcn[index].gcn.M.grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
                importance = model.backbone.gcn[1].gcn
                torch.testing.assert_close(importance.M * importance.WM,
                                           torch.eye(65)[None])

    def test_face_selection_and_padding(self):
        model = SkeleTGCN(7).eval()
        x = torch.randn(2, 3, 8, 133, 2)
        x[..., 1] = 0
        with torch.no_grad():
            expected = model(x)
            selected = model(x.index_select(3, model.expressive_indices))
            x[:, :, :, 23:91] += 999
            changed = model(x)
        torch.testing.assert_close(expected, selected)
        torch.testing.assert_close(expected, changed)
        self.assertTrue(torch.isfinite(changed).all())

    def test_instance_pooling_permutation_and_gradient(self):
        pool = InstancePoolingModule(3, 16, 65, drop_path=0, pe="VC").eval()
        x = torch.randn(2, 3, 65, 3, 8)
        a, b = pool(x), pool(x[:, [2, 0, 1]])
        torch.testing.assert_close(a, b)
        a.square().mean().backward()
        self.assertGreater(pool.point_encoder[0].weight.grad.abs().sum().item(), 0)
        model = SkeleTGCN(7, instance_pooling=True)
        self.assertEqual(tuple(model(torch.randn(2, 3, 8, 65, 2)).shape), (2, 7))

    def test_real_ctr_and_published_config_are_distinct(self):
        self.assertEqual(backbone_config("ctrgcn")["gcn_type"], "unit_ctrgcn")
        self.assertEqual(backbone_config("ctrgcn-official-config")["gcn_type"], "unit_gcn")
        with self.assertRaisesRegex(ValueError, "random initialization"):
            SkeleT_GCN(**backbone_config("dgstgcn"), pretrained="best.pt")

    def test_streams_normalization_and_full_sequence_motion(self):
        raw = np.zeros((3, 4, 133, 2), np.float32)
        raw[0, :, :, 0] = 1440
        raw[1, :, :, 0] = 810
        raw[2, :, :, 0] = 1
        joint = preprocess_keypoints(raw)
        np.testing.assert_allclose(joint[0, ..., :2], 0.5)
        np.testing.assert_array_equal(joint[1], 0)
        raw[0, :, 92, 0] += np.arange(4)*96
        raw[2, :, 92, 0] = [0.4, 0.6, 0.8, 1.0]
        joint = preprocess_keypoints(raw)
        bone = preprocess_keypoints(raw, "b")
        np.testing.assert_allclose(bone, joint-joint[:, :, bone_parents()])
        motion = preprocess_keypoints(raw, "jm")
        np.testing.assert_allclose(motion[:, :-1], np.diff(joint, axis=1))
        np.testing.assert_array_equal(motion[:, -1], 0)

    def test_uniform_sampling_short_medium_long_and_rng(self):
        for frames in (3, 16, 25, 1000):
            np.random.seed(123)
            state = np.random.get_state()
            a = uniform_indices(frames, 16, training=False, num_clips=10)
            after = np.random.get_state()
            np.testing.assert_array_equal(state[1], after[1])
            np.testing.assert_array_equal(a, uniform_indices(frames, 16, training=False, num_clips=10))
            self.assertEqual(a.shape, (10, 16))
            self.assertTrue(((a >= 0) & (a < frames)).all())
        long = uniform_indices(1000, 16, training=False)[0]
        self.assertLess(long[0], 63)
        self.assertGreater(long[-1], 936)

    def test_zip_xsub_xview_and_ntu60_filter(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"poses.zip"
            points = np.full((6, 1, 133, 2), 100, np.float32)
            scores = np.ones((6, 1, 133), np.float32)
            with ZipFile(path, "w") as z:
                for name in ("S001C002P001R001A001_rgb.npz", "S001C001P003R001A001_rgb.npz",
                             "S018C002P001R001A061_rgb.npz"):
                    b = io.BytesIO()
                    np.savez(b, keypoints=points, scores=scores)
                    z.writestr(name, b.getvalue())
            for protocol in ("xsub60", "xview60"):
                train = SkeleTRTMWDataset(path, protocol=protocol, clip_len=8)
                val = SkeleTRTMWDataset(path, protocol=protocol, training=False, clip_len=8)
                self.assertEqual(len(train), 1)
                self.assertEqual(len(val), 1)
                x, label = train[0]
                self.assertEqual(tuple(x.shape), (3, 8, 65, 2))
                self.assertEqual(label.item(), 0)
                self.assertTrue(torch.isfinite(x).all())
                torch.testing.assert_close(x[..., 1], torch.zeros_like(x[..., 1]))
                if train._archive is not None:
                    train._archive.close()
                    train._archive = None
                if val._archive is not None:
                    val._archive.close()
                    val._archive = None

    def test_official_annotations_match_zip_multiclip(self):
        with tempfile.TemporaryDirectory() as d:
            points = np.full((1, 6, 133, 2), 100, np.float32)
            scores = np.ones((1, 6, 133), np.float32)
            name = "S001C001P003R001A001_rgb"
            annotation = dict(frame_dir=name, keypoint=points, keypoint_score=scores,
                              label=0, img_shape=(1080, 1920))
            ann_file = Path(d) / "poses.pkl"
            with ann_file.open("wb") as f:
                pickle.dump(dict(split={"xsub_val": [name]}, annotations=[annotation]), f)
            archive = Path(d) / "poses.zip"
            with ZipFile(archive, "w") as z:
                b = io.BytesIO()
                np.savez(b, keypoints=points.transpose(1, 0, 2, 3),
                         scores=scores.transpose(1, 0, 2))
                z.writestr(name + ".npz", b.getvalue())
            for stream in ("j", "b", "jm", "bm"):
                kwargs = dict(training=False, stream=stream, clip_len=8, num_clips=2)
                pose = SkeleTPoseDataset(ann_file, **kwargs)
                zip_ds = SkeleTRTMWDataset(archive, **kwargs)
                actual, label = pose[0]
                expected, zip_label = zip_ds[0]
                self.assertEqual(tuple(actual.shape), (2, 3, 8, 65, 2))
                torch.testing.assert_close(actual, expected)
                self.assertEqual(label.item(), zip_label.item())
                zip_ds._archive.close()
                zip_ds._archive = None

    def test_fusion_aligns_ids_and_rejects_wrong_labels(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / f"{stream}.npz" for stream in ("j", "b", "jm", "bm")]
            scores = np.asarray([[0.9, 0.1], [0.2, 0.8]])
            for i, path in enumerate(paths):
                order = [1, 0] if i % 2 else [0, 1]
                np.savez(path, probabilities=scores[order], labels=np.asarray([0, 1])[order],
                         sample_names=np.asarray(["sample-a", "sample-b"])[order])
            combined, labels, names, metrics = fuse(paths)
            np.testing.assert_allclose(combined, scores)
            self.assertEqual(metrics["top1"], 1.0)
            np.testing.assert_array_equal(names, ["sample-a", "sample-b"])
            np.savez(paths[-1], probabilities=scores, labels=[1, 0], sample_names=names)
            with self.assertRaisesRegex(ValueError, "labels differ"):
                fuse(paths)

    def test_compile_contract_and_saved_config(self):
        self.assertEqual(parse_args([]).backbone, "ctrgcn")
        self.assertTrue(parse_args([]).compile)
        self.assertTrue(parse_args(["--compile"]).compile)
        for args in (["--no-compile"], ["--no-compile", "--compile"], ["--compile", "--no-compile"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)
        with patch("torch.compile", side_effect=RuntimeError("unsupported")):
            with self.assertRaisesRegex(RuntimeError, "no eager fallback"):
                compile_model(torch.nn.Identity())
        with tempfile.TemporaryDirectory() as d:
            # Mock verifies records only, never substitutes for real CUDA proof.
            with patch("torch.compile", side_effect=lambda m, **kw: m) as compiler, contextlib.redirect_stdout(io.StringIO()):
                main(["--dry-run", "--device", "cpu", "--window-size", "8", "--save-dir", d])
            self.assertEqual(compiler.call_args.kwargs["backend"], "inductor")
            self.assertTrue(compiler.call_args.kwargs["fullgraph"])
            config = json.loads(next(Path(d).glob("*/config.json")).read_text())
            self.assertTrue(config["args"]["compile"])
            self.assertTrue(config["compile"]["enabled"])
            self.assertTrue(config["compile"]["required"])
            self.assertEqual(config["backbone"]["gcn_type"], "unit_ctrgcn")
            self.assertEqual(config["backbone"]["data_bn_type"], "MVC")
            self.assertEqual(config["initialization"], "random; no pretrained weights")

    def test_official_preset_requires_four_bn_batches_and_compile(self):
        path = str(Path(__file__).resolve().parents[1] / "configs/skelet_ctrgcn_official.json")
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--config", path])
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            args = parse_args(["--config", path])
            self.assertEqual(args.backbone, "ctrgcn-official-config")
            self.assertEqual(args.batch_size * args.expected_world_size, 128)
            self.assertEqual(args.num_workers, 2)
            self.assertFalse(args.persistent_workers)
            self.assertTrue(args.test_best)
            self.assertEqual(args.checkpoint_keep, 5)
            self.assertEqual((args.epochs, args.repeat, args.num_clips), (24, 5, 10))
            self.assertTrue(args.compile)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--config", path, "--no-compile", "--compile"])
        # The public distributed sampler shuffles with epoch+seed and pads
        # sample count across ranks while retaining equal per-rank batches.
        dataset = RepeatDataset(list(range(27)), 5)
        args.num_workers = 0
        with patch("isaa.skelet_training.dist.is_initialized", return_value=True), \
             patch("isaa.skelet_training.dist.get_world_size", return_value=4):
            samplers = []
            for rank in range(4):
                with patch("isaa.skelet_training.dist.get_rank", return_value=rank):
                    samplers.append(make_loader(args, dataset, True).sampler)
            shards = [list(s) for s in samplers]
            self.assertEqual([len(s) for s in shards], [34] * 4)
            self.assertEqual(set(sum(shards, [])), set(range(135)))
            samplers[0].set_epoch(1)
            self.assertNotEqual(list(samplers[0]), shards[0])


if __name__ == "__main__":
    unittest.main()
