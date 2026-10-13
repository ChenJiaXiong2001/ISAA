"""Check per-batch scheduling, logs, legacy defaults, and real CUDA execution."""
import io
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from isaa.train import parse_args, run_epoch
from isaa.utils.batch_cosine import BatchWarmupCosine
from isaa.utils.experiment import RunRecords
from train_best_weighted_100_cosine import training_arguments


class BatchCosineTests(unittest.TestCase):
    def parse(self, flags):
        with patch.object(sys, 'argv', ['main.py'] + flags):
            return parse_args()

    def test_preset_and_generic_compile_contract(self):
        args = self.parse(training_arguments([]))
        self.assertEqual((args.epochs, args.window_size), (100, 100))
        self.assertEqual(args.model_variant, 'body-local-hand-ctr-wide-relative')
        self.assertEqual(args.temporal_compression, 'confidence_weighted')
        self.assertEqual((args.lr_scheduler, args.lr_steps, args.warmup_epochs), ('warmup_cosine', [], 5))
        self.assertEqual((args.lr, args.warmup_start_lr, args.min_lr), (.1, .001, .0001))
        self.assertEqual(args.grad_accum_steps, 1)
        self.assertTrue(args.compile)
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.resume_checkpoint)
        for flags in (['--no-compile'], ['--compile', '--no-compile'], ['--no-compile', '--compile']):
            with self.assertRaisesRegex(ValueError, 'requires torch.compile'):
                training_arguments(flags)
            with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                self.parse(training_arguments([]) + flags)
        for flags in (['--init-checkpoint', 'x'], ['--resume-checkpoint', 'x'], ['--lr-steps', '35', '55'],
                      ['--lr-decay', '.1']):
            with self.assertRaises(ValueError):
                training_arguments(flags)
        for flags in (['--grad-accum-steps', '2'], ['--lr-scheduler', 'step']):
            with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                training_arguments(flags)
        for flags in (['--grad-accum-steps', '2'], ['--lr-steps', '35'], ['--resume-checkpoint', 'x'],
                      ['--min-lr', 'nan'], ['--warmup-epochs', '100']):
            with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                self.parse(training_arguments([]) + flags)

    def test_endpoints_monotonicity_and_actual_training_budget(self):
        schedule = BatchWarmupCosine(epochs=100, batches_per_epoch=3, warmup_epochs=5,
                                    base_lr=.1, warmup_start_lr=.001, min_lr=.0001)
        self.assertEqual((schedule.total_updates, schedule.warmup_updates), (300, 15))
        self.assertEqual(schedule.lr_at(0), .001)
        self.assertAlmostEqual(schedule.lr_at(14), .1)
        self.assertAlmostEqual(schedule.lr_at(15), .1)
        self.assertEqual(schedule.lr_at(299), .0001)
        values = [schedule.lr_at(i) for i in range(15, 300)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertLess(schedule.lr_at(3 * 36), schedule.lr_at(3 * 36 - 1))
        for update in (-1, 300):
            with self.assertRaises(ValueError):
                schedule.lr_at(update)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            BatchWarmupCosine(epochs=1, batches_per_epoch=1, warmup_epochs=0,
                             base_lr=.1, warmup_start_lr=.001, min_lr=.0001)

    def test_training_loop_lr_logs_cross_epoch_and_validation(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)
            def forward(self, x, mask):
                return self.linear(x)
        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        loader = DataLoader(TensorDataset(torch.ones(3, 2), torch.zeros(3, dtype=torch.long),
                                         torch.ones(3, 1, dtype=torch.bool)), batch_size=1)
        schedule = BatchWarmupCosine(epochs=3, batches_per_epoch=len(loader), warmup_epochs=1,
                                    base_lr=.1, warmup_start_lr=.001, min_lr=.0001)
        used = []
        original_step = optimizer.step
        def observe_step():
            used.append(optimizer.param_groups[0]['lr'])
            original_step()
        with tempfile.TemporaryDirectory() as tmp:
            records = RunRecords(Path(tmp), {}, Path(tmp))
            try:
                with patch('isaa.train.write_progress_lines'), patch('isaa.train.finish_progress_lines'), \
                        patch.object(optimizer, 'step', side_effect=observe_step):
                    for epoch in range(1, 4):
                        metrics = run_epoch(model, loader, torch.device('cpu'), optimizer,
                                            records=records, epoch=epoch, lr_scheduler=schedule)
                        self.assertEqual(metrics['lr_first'], used[-3])
                        self.assertEqual(metrics['lr_last'], used[-1])
                        self.assertAlmostEqual(metrics['lr_mean'], sum(used[-3:])/3)
                        before = schedule.state_dict()
                        run_epoch(model, loader, torch.device('cpu'), records=records, epoch=epoch,
                                  learning_rate=used[-1], lr_scheduler=schedule)
                        self.assertEqual(schedule.state_dict(), before)
            finally:
                records.close()
            rows = [json.loads(s) for s in (Path(tmp)/'batches.jsonl').read_text().splitlines()]
            training = [r for r in rows if r['phase'] == 'train']
            self.assertEqual([r['lr'] for r in training], used)
            self.assertEqual([r['optimizer_update'] for r in training], list(range(9)))
            self.assertEqual(used, [schedule.lr_at(i) for i in range(9)])
            self.assertEqual(schedule.completed_updates, 9)

    @unittest.skipUnless(os.environ.get('ISAA_TEST_DEVICE') == 'cuda', 'Requires server CUDA compilation')
    def test_cuda_compiled_training(self):
        from tests.test_best_weighted_100 import BestWeighted100Tests
        self.assertTrue(torch.cuda.is_available())
        root = Path(__file__).resolve().parents[1]
        evidence = os.environ.get('ISAA_COSINE_CUDA_EVIDENCE_DIR')
        if evidence:
            Path(evidence).mkdir(parents=True, exist_ok=False)
        with (nullcontext(evidence) if evidence else tempfile.TemporaryDirectory()) as tmp:
            directory = Path(tmp)
            archive = directory/'samples.zip'
            BestWeighted100Tests.make_archive(archive)
            result = subprocess.run([sys.executable, str(root/'train_best_weighted_100_cosine.py'),
                '--archive', str(archive), '--num-classes', '2', '--epochs', '1', '--warmup-epochs', '0',
                '--batch-size', '1', '--test-batch-size', '2', '--num-workers', '0', '--device', 'cuda',
                '--save-dir', str(directory/'runs')], cwd=root, capture_output=True, text=True, timeout=1800)
            self.assertEqual(result.returncode, 0, result.stdout[-7000:] + result.stderr[-7000:])
            run = next((directory/'runs').iterdir())
            config = json.loads((run/'config.json').read_text())
            self.assertTrue(config['compile']['enabled'])
            self.assertTrue(config['compile']['required'])
            self.assertFalse(config['compile']['suppress_errors'])
            self.assertEqual(config['initialization'], 'random_from_scratch')
            self.assertEqual(config['lr_schedule']['total_updates'], config['dataset']['train']['batches'])
            self.assertEqual(config['args']['lr_steps'], [])
            checkpoint = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['lr_schedule']['completed_updates'], 2)
            self.assertTrue(checkpoint['optimizer']['state'])
            self.assertTrue(all(torch.isfinite(t).all() for t in checkpoint['model'].values()))
            rows = [json.loads(s) for s in (run/'batches.jsonl').read_text().splitlines()]
            self.assertEqual([r['lr'] for r in rows if r['phase']=='train'], [.1, .0001])
            from zipfile import ZipFile
            with ZipFile(run/'source.zip') as source:
                self.assertIn('train_best_weighted_100_cosine.py', source.namelist())
                self.assertIn('isaa/utils/batch_cosine.py', source.namelist())


if __name__ == '__main__':
    unittest.main()
