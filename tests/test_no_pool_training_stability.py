"""Prevent corrupt SGD updates and retain useful records on numerical failure."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from isaa.train import run_epoch, parse_args
from isaa.utils.experiment import RunRecords
from isaa.utils.no_pool_resume import load_resume_checkpoint
from train_best_no_pool import training_arguments


class TinyClassifier(nn.Module):
    ARCHITECTURE = 'test_no_pool'

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, mask):
        return self.linear(x)


class StabilityTests(unittest.TestCase):
    @staticmethod
    def loader():
        return DataLoader(TensorDataset(torch.tensor([[1000., 0.], [0., 1000.]]),
                                       torch.tensor([0, 1]), torch.ones(2, 1, dtype=torch.bool)), batch_size=1)

    def test_clipping_limits_update_and_accumulation_uses_one_update(self):
        model = TinyClassifier()
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        with patch('isaa.train.write_progress_lines'), patch('isaa.train.finish_progress_lines'), \
                patch.object(opt, 'step', wraps=opt.step) as step:
            run_epoch(model, self.loader(), torch.device('cpu'), opt,
                      progress_context={'max_grad_norm': 5., 'grad_accum_steps': 2})
        self.assertEqual(step.call_count, 1)
        self.assertAlmostEqual(model.linear.weight.norm().item(), .5, places=5)

    def test_nonfinite_gradient_stops_before_update_and_flushes_prior_batch(self):
        model = TinyClassifier()
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        count = 0

        def bad_second_gradient(grad):
            nonlocal count
            count += 1
            return grad if count == 1 else torch.full_like(grad, float('nan'))

        hook = model.linear.weight.register_hook(bad_second_gradient)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            records = RunRecords(directory, {}, directory)
            try:
                with patch('isaa.train.write_progress_lines'), patch('isaa.train.finish_progress_lines'), \
                        patch.object(opt, 'step', wraps=opt.step) as step, self.assertRaisesRegex(RuntimeError, 'non-finite'):
                    run_epoch(model, self.loader(), torch.device('cpu'), opt, records=records, epoch=3,
                              progress_context={'max_grad_norm': 5.})
                self.assertEqual(step.call_count, 1)
                self.assertTrue(torch.isfinite(model.linear.weight).all())
            finally:
                records.close(); hook.remove()
            rows = [json.loads(line) for line in (directory/'batches.jsonl').read_text().splitlines()]
            self.assertEqual([(r['epoch'], r['step']) for r in rows], [(3, 1)])
            self.assertGreater(rows[0]['gradient_norm'], 5)
            failure = torch.load(directory/'failure.pt', weights_only=False)
            self.assertEqual((failure['epoch'], failure['step'], failure['reason']), (3, 2, 'nonfinite_gradient_norm'))
            self.assertFalse(torch.isfinite(failure['gradients']['linear.weight']).all())

    def test_resume_restores_matching_weights_and_rejects_bad_or_baseline_state(self):
        args = SimpleNamespace(model_variant='test', num_classes=2, window_size=4, max_persons=1,
                               split='xsub60', feature_mode='raw', epochs=4)
        model = TinyClassifier()
        opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        model(torch.ones(1, 2), None).sum().backward(); opt.step()
        checkpoint = {'epoch':2, 'architecture':model.ARCHITECTURE, 'model':model.state_dict(),
                      'optimizer':opt.state_dict(), 'args':vars(args), 'val_accuracy':.5}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'
            torch.save(checkpoint, path)
            restored = TinyClassifier()
            loaded = load_resume_checkpoint(path, restored, args)
            torch.testing.assert_close(restored.linear.weight, model.linear.weight)
            restored_opt = torch.optim.SGD(restored.parameters(), lr=.1, momentum=.9)
            restored_opt.load_state_dict(loaded['optimizer'])
            torch.testing.assert_close(restored_opt.state[restored.linear.weight]['momentum_buffer'],
                                       opt.state[model.linear.weight]['momentum_buffer'])
            checkpoint['architecture'] = 'original_best'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'architecture'):
                load_resume_checkpoint(path, restored, args)
            checkpoint['architecture'] = model.ARCHITECTURE
            checkpoint['model']['linear.weight'][0, 0] = float('nan')
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                load_resume_checkpoint(path, restored, args)

    def test_no_pool_cannot_remove_guard(self):
        import sys
        for value in ('0', '-1', 'nan', 'inf'):
            with patch.object(sys, 'argv', ['main.py'] + training_arguments(['--max-grad-norm', value])), \
                    patch('sys.stderr'), self.assertRaises(SystemExit):
                parse_args()

    def test_nonfinite_loss_saves_input_and_flushes_previous_success(self):
        class BadLossClassifier(TinyClassifier):
            def forward(self, x, mask):
                if x[0, 1] > 0:
                    return torch.full((1, 2), float('nan'))
                return super().forward(x, mask)

        model = BadLossClassifier()
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            records = RunRecords(directory, {}, directory)
            try:
                with patch('isaa.train.write_progress_lines'), patch('isaa.train.finish_progress_lines'), \
                        patch.object(opt, 'step', wraps=opt.step) as step, self.assertRaisesRegex(RuntimeError, 'Non-finite loss at batch 2'):
                    run_epoch(model, self.loader(), torch.device('cpu'), opt, records=records,
                              progress_context={'max_grad_norm':5.})
                self.assertEqual(step.call_count, 1)
            finally:
                records.close()
            failure = torch.load(directory/'failure.pt', weights_only=False)
            torch.testing.assert_close(failure['x'], torch.tensor([[0.,1000.]]))
            self.assertEqual(failure['reason'], 'nonfinite_loss')
            self.assertEqual(len((directory/'batches.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
