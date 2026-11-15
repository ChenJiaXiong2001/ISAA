import argparse
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import isaa.train  # Configure CUDA library paths before importing torch.
import torch

from curriculum import CurriculumDataset, clean_probability
from train import validate_protocol


class WorkerProbe(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 1

    def __getitem__(self, index):
        self.dataset._refresh_late_augmentation()
        return self.dataset.augmentation_config["precompression_anchor_clean_sample_probability"]


class CurriculumTests(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual([clean_probability(i) for i in (1, 10, 11, 15, 20)],
                         [0.5, 0.5, 0.55, 0.75, 1.0])
        with self.assertRaises(ValueError):
            clean_probability(21)

    def test_shared_epoch_refresh_and_nontraining(self):
        dataset = CurriculumDataset.__new__(CurriculumDataset)
        dataset.split, dataset.augment = "train", True
        dataset._epoch_state = torch.zeros(2, dtype=torch.int64).share_memory_()
        dataset.augmentation_config = {"precompression_anchor_clean_sample_probability": 0.5}
        dataset.set_epoch(15)
        self.assertEqual(dataset.augmentation_config["precompression_anchor_clean_sample_probability"], 0.75)
        dataset.split = "val"
        dataset.set_epoch(20)
        self.assertEqual(dataset.augmentation_config["precompression_anchor_clean_sample_probability"], 0.75)

    def test_persistent_spawn_worker_observes_new_epoch(self):
        dataset = CurriculumDataset.__new__(CurriculumDataset)
        dataset.split, dataset.augment = "train", True
        dataset._epoch_state = torch.zeros(2, dtype=torch.int64).share_memory_()
        dataset.augmentation_config = {"precompression_anchor_clean_sample_probability": 0.5}
        loader = torch.utils.data.DataLoader(WorkerProbe(dataset), num_workers=1,
            persistent_workers=True, multiprocessing_context="spawn")
        dataset.set_epoch(10)
        self.assertAlmostEqual(next(iter(loader)).item(), 0.5)
        dataset.set_epoch(20)
        self.assertAlmostEqual(next(iter(loader)).item(), 1.0)

    def test_protocol_rejects_eager_and_checkpoint(self):
        args = argparse.Namespace(model_variant="best-face-pair-hand-motion", epochs=20,
            max_samples=4096, batch_size=32, test_batch_size=32, seed=17,
            warmup_epochs=1, lr_scheduler="warmup_cosine", lr=0.025,
            warmup_start_lr=0.001, min_lr=0.0001, momentum=0.9, nesterov=True,
            weight_decay=0.0004, grad_accum_steps=1, drop_last=True, max_grad_norm=5.0,
            pair_head_loss_weight=0.5, pair_fusion_loss_weight=1.0,
            pair_calibration_fraction=0.1, pair_calibration_start_epoch=1,
            precompression_anchor_dropout_probability=0.025,
            precompression_anchor_clean_sample_probability=0.5,
            compile=True, compile_mode="default", init_checkpoint=None, resume_checkpoint=None)
        self.assertEqual(validate_protocol(args).anchor_clean_curriculum["epochs"], 20)
        args.compile = False
        with self.assertRaises(ValueError):
            validate_protocol(args)
        args.compile, args.init_checkpoint = True, "old.pt"
        with self.assertRaises(ValueError):
            validate_protocol(args)


if __name__ == "__main__":
    unittest.main()
