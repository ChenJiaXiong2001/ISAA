"""CPU checks for checkpoint guards, sample alignment and fusion reconstruction."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import os
from zipfile import ZipFile

import numpy as np
import torch

from experiments.training_route_mismatch_20261106 import paired_evaluate as e
from experiments.training_route_mismatch_20261106.soft_route import gamma1_log_fusion


def checkpoint(architecture):
    args = {"model_variant": e.FACE_PAIR_VARIANT, "compile": True, "compile_mode": "default",
            "init_checkpoint": None, "resume_checkpoint": None, "feature_mode": "raw", "npy_dir": None,
            "window_size": 100, "temporal_compression": "confidence_weighted", "lr_scheduler": "warmup_cosine",
            "epochs": 20, "warmup_epochs": 1, "lr": .025, "warmup_start_lr": .001, "min_lr": .0001,
            "momentum": .9, "weight_decay": .0004, "grad_accum_steps": 1, "batch_size": 32,
            "max_samples": 4096, "seed": 17, "pair_calibration_start_epoch": 1}
    policy = {"enabled": True, "required": True, "backend": "inductor", "suppress_errors": False,
              "fail_on_recompile_limit_hit": True, "whole_model_eager_fallback": False,
              "triton_cudagraphs": False, "mode": "default"}
    return {"architecture": architecture, "epoch": 20, "args": args, "compile": policy}


def manifest():
    value = {"official_validation_used_for_routing": False}
    for section in ("fit", "calibration", "validation"):
        names = [section + "1", section + "2"]
        value[section] = {"members": names, "count": len(names), "sha256": e.members_digest(names)}
    return value


class PairedEvaluatorTests(unittest.TestCase):
    def test_checkpoint_guards(self):
        left, right = checkpoint(e.FacePairFusion.ARCHITECTURE), checkpoint(e.SoftRouteFacePair.ARCHITECTURE)
        m = manifest()
        e.validate_checkpoints(left, right, [m, m])
        right["architecture"] = left["architecture"]
        with self.assertRaisesRegex(ValueError, "original hard control"):
            e.validate_checkpoints(left, right, [m, m])
        right["architecture"] = e.SoftRouteFacePair.ARCHITECTURE
        right["epoch"] = 19
        with self.assertRaisesRegex(ValueError, "epoch-20"):
            e.validate_checkpoints(left, right, [m, m])
        right["epoch"] = 20
        right["args"]["seed"] = 9
        with self.assertRaisesRegex(ValueError, "seed"):
            e.validate_checkpoints(left, right, [m, m])

    def test_manifest_and_order_guards(self):
        left, right = checkpoint(e.FacePairFusion.ARCHITECTURE), checkpoint(e.SoftRouteFacePair.ARCHITECTURE)
        m, bad = manifest(), manifest()
        bad["validation"]["members"].reverse()
        with self.assertRaisesRegex(ValueError, "validation members"):
            e.validate_checkpoints(left, right, [m, bad])
        with self.assertRaisesRegex(ValueError, "order"):
            e.validate_order(["b", "a"], [0, 1], ["a", "b"], [0, 1])
        with self.assertRaisesRegex(ValueError, "order"):
            e.validate_order(["a", "b"], [1, 0], ["a", "b"], [0, 1])

    def test_extreme_branch_logits_and_shared_weights(self):
        h = torch.tensor([[10000., -10000., 0.], [-10000., 10000., 0.]])
        b = -h
        weights = (torch.tensor([0., 1., .5]), torch.tensor([1., 0., .5]))
        count_h, count_b = torch.tensor([1, 0, 1]), torch.tensor([0, 1, 1])
        outputs = [{"hand_face_logits": h, "body_face_logits": b},
                   {"hand_face_logits": b, "body_face_logits": h}]
        outputs[1]["logits"] = gamma1_log_fusion(b, h, count_h, count_b, 1., 1.)
        own, common = e.paired_predictions(outputs, weights)
        arrays = [torch.stack((x["hand_face_logits"], x["body_face_logits"]), 1).numpy() for x in outputs]
        rows = [{"control_prediction": own[0][i].item(), "candidate_prediction": own[1][i].item(),
                 "shared_candidate_prediction": common[1][i].item()} for i in range(2)]
        e.verify_cached_predictions(arrays, rows, weights)
        rows[0]["candidate_prediction"] = (rows[0]["candidate_prediction"] + 1) % 3
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            e.verify_cached_predictions(arrays, rows, weights)

    def test_source_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp)/"a.zip", Path(tmp)/"b.zip"
            for path, contents in ((left, b"a"), (right, b"b")):
                with ZipFile(path, "w") as archive:
                    for name in ("isaa/model.py", "train_best_face_pair_100.py", "train_best_weighted_100_cosine.py"):
                        archive.writestr(name, contents)
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                e.verify_sources(left, right)

    def test_single_subject_has_no_bootstrap_interval(self):
        rows = [{"sample": "S001C001P025R001A001.skeleton", "true_class_index": 0,
                 "control_prediction": 1, "candidate_prediction": 0}]
        result = e.condition_stats(rows, "candidate_prediction")
        self.assertIsNone(result["subject_ci95_pp"])
        self.assertEqual(result["corrected"], 1)
        self.assertEqual(result["equal_subject_difference_pp"], 100.)

    def test_candidate_output_mismatch_is_rejected(self):
        out = {"hand_face_logits": torch.tensor([[1., 0.]]),
               "body_face_logits": torch.tensor([[0., 1.]]),
               "logits": torch.tensor([[0., 0.]])}
        with self.assertRaisesRegex(RuntimeError, "does not reconstruct"):
            e.paired_predictions((out, out), (torch.tensor([1., 0.]), torch.tensor([1., 0.])))

    def test_common_weight_result_cannot_be_swapped(self):
        h = torch.tensor([[3., -3.]])
        b = -h
        weights = (torch.tensor([1., 1.]), torch.tensor([0., 0.]))
        outputs = [{"hand_face_logits": h, "body_face_logits": b},
                   {"hand_face_logits": h, "body_face_logits": b}]
        outputs[1]["logits"] = gamma1_log_fusion(h, b, torch.tensor([0, 0]), torch.tensor([1, 1]), 1., 1.)
        own, common = e.paired_predictions(outputs, weights)
        self.assertNotEqual(own[1].item(), common[1].item())
        arrays = [torch.stack((h, b), 1).numpy()] * 2
        rows = [{"control_prediction": own[0].item(), "candidate_prediction": own[1].item(),
                 "shared_candidate_prediction": own[1].item()}]
        with self.assertRaisesRegex(ValueError, "shared-weight"):
            e.verify_cached_predictions(arrays, rows, weights)

    @unittest.skipUnless(os.environ.get("ISAA_EVAL_ARCHIVE"), "requires real archive")
    def test_real_data_loaders_cpu(self):
        archive = Path(os.environ["ISAA_EVAL_ARCHIVE"])
        kwargs = {"split": "val", "split_protocol": "xsub60", "window_size": 100,
                  "num_joints": 133, "num_classes": 60, "layout": "rtmw_133", "max_persons": 2,
                  "feature_mode": "raw", "max_samples": 2, "temporal_compression": "confidence_weighted"}
        clean = e.RTMWZipDataset(archive, **kwargs)
        masked = e.RTMWZipDataset(archive, augment=True, augmentation_config={"joint_dropout_probability": .05}, **kwargs)
        e.validate_order(clean.members, clean.labels, masked.members, masked.labels)
        standard = e.DataLoader(e.NamedSamples(masked, [0, 1]), batch_size=2, collate_fn=e.named_standard_collate, num_workers=0)
        names, (features, labels, frames) = next(iter(standard))
        self.assertEqual(names, list(clean.members))
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(labels.tolist(), [int(x) for x in clean.labels])
        paired = e.PairedPrecompression(clean, clean.members, .05, 1729)
        raw = e.DataLoader(e.NamedSamples(paired, [0, 1]), batch_size=2, num_workers=0)
        names2, (clear, corrupt, labels2, frames2, _, _) = next(iter(raw))
        self.assertEqual(list(names2), list(clean.members))
        self.assertEqual(clear.shape, corrupt.shape)
        self.assertEqual(labels.tolist(), labels2.tolist())
        self.assertEqual(frames.shape, frames2.shape)
        reordered = e.DataLoader(e.NamedSamples(masked, [1, 0]), batch_size=2, collate_fn=e.named_standard_collate, num_workers=0)
        wrong_names, (_, wrong_labels, _) = next(iter(reordered))
        with self.assertRaisesRegex(ValueError, "order"):
            e.validate_order(wrong_names, wrong_labels.tolist(), clean.members, clean.labels)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
