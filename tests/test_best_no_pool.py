"""Verify ordered readouts, preserved encoders and checkpoint evaluation."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from isaa.models.best_no_pool import BestNoPoolFusion, NO_POOL_VARIANT, OrderedBranchReadout
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.train import parse_args
from train_best_no_pool import training_arguments


class BestNoPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    @staticmethod
    def sample(channels=3, persons=2):
        x = torch.randn(2, channels, 5, 133, persons) * .05
        x[:, 2] = 1
        return x

    def test_encoder_features_match_best_raw_and_cached_inputs(self):
        baseline = BodyLocalHandCTRWideRelativeFusion(7).eval()
        model = BestNoPoolFusion(7, window_size=5).eval()
        state = baseline.state_dict()
        model.load_state_dict({key: state.get(key, value) for key, value in model.state_dict().items()})
        before, after = [], []
        hooks = []
        for old, new in ((baseline.body_blocks[-1], model.body_blocks[-1]),
                         (baseline.hand_ctr_blocks[0], model.hand_ctr_blocks[0]),
                         (baseline.hand_ctr_blocks[-1], model.hand_ctr_blocks[-1]),
                         (baseline.face_st_blocks[-1], model.face_st_blocks[-1])):
            hooks.append(old.register_forward_hook(lambda _, args, out: before.append((args[0], out[0]))))
            hooks.append(new.register_forward_hook(lambda _, args, out: after.append((args[0], out[0]))))
        try:
            for channels in (3, 8):
                before.clear()
                after.clear()
                x = self.sample(channels)
                with torch.no_grad():
                    baseline(x)
                    model(x)
                self.assertEqual(len(before), len(after))
                for (old_in, old_out), (new_in, new_out) in zip(before, after):
                    torch.testing.assert_close(old_in, new_in, rtol=0, atol=0)
                    torch.testing.assert_close(old_out, new_out, rtol=0, atol=0)
        finally:
            for hook in hooks:
                hook.remove()

    def test_readout_distinguishes_joint_and_time_order(self):
        head = OrderedBranchReadout(1, 2, 2)
        with torch.no_grad():
            head.node_projection.weight.copy_(torch.tensor([[1., 2.]]))
            head.time_projection.weight.copy_(torch.tensor([[1., 3.]]))
        x = torch.tensor([[[[1., 2.], [4., 8.]]]])
        mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        self.assertNotEqual(head(x, mask).item(), head(x.flip(-1), mask).item())
        self.assertNotEqual(head(x, mask).item(), head(x.flip(-2), mask).item())
        self.assertEqual(head(x, torch.zeros_like(mask)).item(), 0)
        masked = mask.clone()
        masked[..., 1] = False
        corrupt = x.clone()
        corrupt[..., 1] = float('nan')
        torch.testing.assert_close(head(corrupt, masked), head(x, masked))

    def test_missing_slots_frames_and_nonfinite_inputs(self):
        model = BestNoPoolFusion(7, window_size=5).eval()
        x = self.sample(persons=1)
        padded = torch.cat((x, torch.zeros_like(x)), -1)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(padded))
            torch.testing.assert_close(model(x.squeeze(-1)), model(x))
            frame_mask = torch.tensor([[True, True, False, False, False]] * 2)
            corrupt = x.clone()
            corrupt[:, :, 2:] = float('nan')
            torch.testing.assert_close(model(x, frame_mask), model(corrupt, frame_mask))
            empty = torch.full_like(padded, float('nan'))
            self.assertTrue(torch.isfinite(model(empty)).all())
            cached = self.sample(8)
            cached[:, 4, 0, 91] = float('nan')
            self.assertTrue(torch.isfinite(model(cached)).all())

    def test_all_active_encoders_and_new_heads_receive_gradients(self):
        model = BestNoPoolFusion(7, window_size=5)
        torch.nn.functional.cross_entropy(model(self.sample()), torch.tensor([1, 4])).backward()
        for module in (model.body_blocks, model.hand_ctr_blocks, model.face_st_blocks,
                       model.hand_to_local, model.body_readout, model.hand_readout,
                       model.face_readout, model.body_person_projection, model.local_slot_projection,
                       model.body_projection, model.local_projection, model.classifier):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_fixed_shape_contract_and_preset(self):
        flags = training_arguments(['--dry-run', '--window-size', '5'])
        with patch.object(sys, 'argv', ['main.py'] + flags):
            args = parse_args()
        self.assertEqual(args.model_variant, NO_POOL_VARIANT)
        self.assertEqual((args.node_count, args.feature_mode), (133, 'raw'))
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.gcn_config)
        self.assertEqual(args.max_grad_norm, 5.0)
        self.assertEqual(args.compile_mode, 'default')
        self.assertEqual(args.lr, 0.025)
        self.assertIsNone(args.resume_checkpoint)
        self.assertEqual(args.save_dir, 'outputs/best_no_pool_ntu60_xsub')
        model = BestNoPoolFusion(7, window_size=5)
        for x in (torch.zeros(2, 3, 4, 133, 2), torch.zeros(2, 3, 5, 133, 3),
                  torch.zeros(2, 5, 5, 133, 2)):
            with self.assertRaises(ValueError):
                model(x)
        with self.assertRaises(ValueError):
            training_arguments(['--init-checkpoint', 'best.pt'])
        with patch.object(sys, 'argv', ['main.py', '--model-variant', NO_POOL_VARIANT]):
            generic = parse_args()
        self.assertEqual((generic.lr, generic.compile_mode, generic.max_grad_norm), (0.025, 'default', 5.))
        with patch.object(sys, 'argv', ['main.py', '--model-variant', 'body-local-hand-ctr-wide-relative']):
            baseline = parse_args()
        self.assertEqual((baseline.lr, baseline.compile_mode, baseline.max_grad_norm), (0.1, 'reduce-overhead', 0.))
        with patch.object(sys, 'argv', ['main.py'] + training_arguments(['--lr', '0.05'])):
            self.assertEqual(parse_args().lr, 0.05)

    def test_compile_required_in_both_entries(self):
        with patch.object(sys, 'argv', ['main.py'] + training_arguments([])):
            self.assertTrue(parse_args().compile)
        for flags in (['--no-compile'], ['--compile', '--no-compile'], ['--no-compile', '--compile']):
            with self.assertRaisesRegex(ValueError, 'requires torch.compile'):
                training_arguments(flags)
            with patch.object(sys, 'argv', ['main.py', '--model-variant', NO_POOL_VARIANT] + flags), \
                    patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_small_zip_training_checkpoint_evaluation_and_reload(self):
        try:
            torch.compile(torch.nn.Identity())
        except RuntimeError as exc:
            if 'Windows not yet supported' in str(exc):
                self.skipTest('Compiled training requires a supported runtime; no eager fallback')
            raise
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / 'samples.zip'
            x = self.sample(persons=1)[0, :, :, :, 0]
            with zipfile.ZipFile(archive, 'w') as output:
                for subject in (1, 3):
                    for action in (1, 2, 3):
                        payload = io.BytesIO()
                        np.savez(payload, keypoints=x[:2].permute(1, 2, 0).numpy()[:, None],
                                 scores=x[2].numpy()[:, None])
                        output.writestr(f'S001C001P{subject:03}R001A{action:03}_rgb.npz', payload.getvalue())
            command = [sys.executable, str(root / 'train_best_no_pool.py'),
                       '--archive', str(archive), '--num-classes', '3', '--epochs', '1',
                       '--window-size', '5', '--batch-size', '3', '--test-batch-size', '3',
                       '--max-persons', '1', '--num-workers', '0', '--device', os.environ.get('ISAA_TEST_DEVICE', 'cpu'), '--no-drop-last', '--save-dir', str(directory / 'runs')]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            run = next((directory / 'runs').iterdir())
            checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            self.assertTrue(checkpoint['args']['compile'])
            run_config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
            self.assertTrue(run_config['compile']['enabled'])
            self.assertTrue(run_config['compile']['required'])
            self.assertEqual(run_config['compile']['mode'], 'default')
            self.assertFalse(run_config['compile']['suppress_errors'])
            self.assertEqual(run_config['optimizer']['max_grad_norm'], 5.0)
            self.assertEqual(run_config['optimizer']['base_lr'], 0.025)
            self.assertEqual(checkpoint['architecture'], BestNoPoolFusion.ARCHITECTURE)
            config = checkpoint['model_config']['readout_config']
            self.assertEqual((config['window_size'], config['max_persons']), (5, 1))
            self.assertEqual(checkpoint['model_config']['hand_input_channels'], 6)
            model = BestNoPoolFusion(3, window_size=config['window_size'], max_persons=config['max_persons'])
            model.load_state_dict(checkpoint['model'], strict=True)
            self.assertTrue(torch.isfinite(model.eval()(x[None, :, :, :, None])).all())
            evaluation = subprocess.run([sys.executable, str(root / 'tools/evaluate_best.py'),
                '--checkpoint', str(run / 'best.pt'), '--device', 'cpu', '--num-workers', '0',
                '--output-dir', str(directory / 'evaluation')],
                cwd=root, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout[-3000:] + evaluation.stderr[-3000:])
            summary = json.loads((directory / 'evaluation/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['samples'], 3)
            self.assertTrue(summary['same_top1_correct_count'])
            continuation = command.copy()
            continuation[continuation.index('--epochs') + 1] = '2'
            continuation += ['--resume-checkpoint', str(run / 'last.pt')]
            resumed = subprocess.run(continuation, cwd=root, env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(resumed.returncode, 0, resumed.stdout[-3000:] + resumed.stderr[-3000:])
            self.assertIn('next_epoch=2', resumed.stdout)
            resumed_run = next(p for p in (directory / 'runs').iterdir() if p != run)
            saved = torch.load(resumed_run / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(saved['epoch'], 2)
            self.assertEqual(saved['args']['max_grad_norm'], 5.0)


if __name__ == '__main__':
    unittest.main()
