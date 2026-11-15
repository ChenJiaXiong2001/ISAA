"""Conditional per-clip routing, missing observations and mandatory compile."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from isaa.models.input_joint_selection import OPTIONAL_BODY
from isaa.models.input_joint_selection_v4 import ConditionalJointGates, InputJointSelectionV4Fusion, INPUT_SELECTION_V4_VARIANT
from isaa.input_joint_training_v4 import parse_args, build_model, make_optimizers, compile_model, load_checkpoint, gate_weight
from train_best_input_selection_v4_100 import training_arguments


class InputJointSelectionV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def input(batch=2, frames=5):
        x = torch.randn(batch, 3, frames, 129, 2) * .05
        x[:, 2] = 1.
        return x

    def test_entry_compile_defaults_and_no_warmstart(self):
        from isaa.train import parse_args as generic_parse
        args = parse_args([])
        self.assertTrue(args.compile)
        self.assertEqual((args.epochs, args.window_size, args.batch_size), (100, 100, 32))
        self.assertFalse(hasattr(args, 'keep_k'))
        for overrides in ([], ['--compile']):
            with patch.object(sys, 'argv', ['main.py', *training_arguments(overrides)]):
                self.assertEqual(generic_parse().model_variant, INPUT_SELECTION_V4_VARIANT)
        for flags in (['--no-compile'], ['--no-compile', '--compile'], ['--compile', '--no-compile']):
            with self.assertRaises(ValueError):
                training_arguments(flags)
            with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)
            with patch.object(sys, 'argv', ['main.py', '--model-variant='+INPUT_SELECTION_V4_VARIANT, *flags]), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                generic_parse()
        for flag in ('--init-checkpoint', '--resume-checkpoint', '--npy-dir'):
            with self.assertRaises(ValueError):
                training_arguments([flag, 'unused'])
            with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, 'unused'])
        with patch('torch.load', side_effect=AssertionError('Must train from scratch')):
            model = build_model(args)
        backbone, router = make_optimizers(model, args)
        self.assertIsInstance(backbone, torch.optim.SGD)
        self.assertIsInstance(router, torch.optim.Adam)
        ids = {id(p) for group in backbone.param_groups for p in group['params']}
        self.assertTrue(all(id(p) not in ids for p in model.gates.parameters()))

    def test_warmup_binary_transition_and_cost(self):
        g = ConditionalJointGates()
        descriptors, context = torch.randn(4, 40, 8), torch.randn(4, 8)
        initial = {k: v.detach().clone() for k, v in g.named_parameters()}
        for epoch in range(1, 6):
            g.begin_epoch(epoch, 5, 5)
            values, penalty, aux = g(descriptors, context)
            torch.testing.assert_close(values, torch.ones_like(values), rtol=0, atol=0)
            (values.sum() + gate_weight(epoch, parse_args([])) * penalty).backward()
            for name, p in g.named_parameters():
                self.assertEqual(float(p.grad.abs().sum()), 0.)
                torch.testing.assert_close(p, initial[name], rtol=0, atol=0)
            g.zero_grad(set_to_none=True)
        g.begin_epoch(6, 5, 5)
        self.assertAlmostEqual(float(g.open_mix), .8, places=6)
        g.begin_epoch(10, 5, 5)
        values, penalty, aux = g(descriptors, context)
        self.assertTrue(set(values.detach().flatten().tolist()).issubset({0., 1.}))
        torch.testing.assert_close(penalty, (aux['gate_logits'].sigmoid() * g.cost).sum(-1).mean())
        self.assertEqual(float(g.cost.sum()), 60.)
        penalty.backward()
        self.assertGreater(float(g.router[-1].bias.grad.abs().sum()), 0.)
        self.assertAlmostEqual(gate_weight(25, parse_args([])), .01)
        g.eval(); g.hard_eval.fill_(True)
        self.assertTrue(set(g(descriptors, context)[0].flatten().tolist()).issubset({0., 1.}))

    def test_input_conditioning_and_deterministic_evaluation(self):
        model = InputJointSelectionV4Fusion(7).eval()
        g = model.gates
        g.begin_epoch(10, 5, 5)
        # Explicitly trained-like weights: decision depends on node motion.
        with torch.no_grad():
            for p in g.parameters():
                p.zero_()
            g.router[0].weight[0, 4] = 1.
            g.router[-1].weight[0, 0] = 2.
            g.router[-1].bias.fill_(-.2)
        x = self.input()
        x[:, :2] = 0.
        x[:, 1, :, model.routing_torso_positions[:2]] = .5
        x[:, 1, :, model.routing_torso_positions[2:]] = -.5
        p = int(model.candidate_positions[0, 0])
        x[0, :2, :, p] = 0.
        x[1, 0, :, p] = torch.tensor([0., .5, -.5, .5, 0.])[:, None]
        mask = x[:, 2:3] > 0
        gates, _, _ = model.route(x, mask)
        again, _, _ = model.route(x, mask)
        torch.testing.assert_close(gates, again, rtol=0, atol=0)
        self.assertGreater(float((gates[0] - gates[1]).abs().max()), .1)
        g.hard_eval.fill_(True)
        hard, _, _ = model.route(x, mask)
        self.assertTrue(((hard[0] > 0) != (hard[1] > 0)).any())
        self.assertEqual(model.selection['raw_input_count'], 129)
        self.assertEqual(len(gates[0]), 40)
        self.assertEqual(tuple(model.face_positions.shape), (6, 12))
        torch.testing.assert_close(model.face_positions, model.face_indices)
        with self.assertRaisesRegex(ValueError, 'input clip'):
            model.export_selection()

    def test_missing_observations_finite_and_classification_gradient(self):
        model = InputJointSelectionV4Fusion(7).train()
        model.gates.begin_epoch(10, 5, 5)
        x = self.input()
        x[0, :, :, :, 1] = 0.
        x[1, :, :, 4] = float('nan')
        x[1, :, 2] = 0.
        frame = torch.ones(2, 5, dtype=torch.bool); frame[0, -1] = False
        logits, penalty, aux = model(x, frame, return_gate_values=True)
        self.assertEqual(logits.shape, (2, 7))
        self.assertTrue(torch.isfinite(logits).all() and torch.isfinite(penalty))
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
        grads = [p.grad for p in model.gates.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(p).all() for p in grads))
        self.assertGreater(sum(float(p.abs().sum()) for p in grads), 0.)
        model.zero_grad(set_to_none=True)
        model.gates.hard_train.fill_(True)
        model.gates.noise_scale.zero_()
        with torch.no_grad():
            model.gates.router[-1].weight.zero_()
            model.gates.router[-1].bias.fill_(-10.)
        closed_logits, _, closed_aux = model(x, frame, return_gate_values=True)
        closed_aux['gate_logits'].retain_grad()
        closed_logits.square().mean().backward()
        self.assertTrue((closed_aux['sampled_gate_values'].detach() == 0).all())
        self.assertGreater(float(model.gates.router[-1].bias.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(model.gates.router[-1].bias.grad))
        for group in (slice(0, 14), slice(14, 34), slice(34, 40)):
            self.assertGreater(float(closed_aux['gate_logits'].grad[:, group].abs().sum()), 0.)
        model.eval()
        with torch.no_grad():
            logits, penalty, _ = model(torch.full_like(x, float('nan')), frame, return_gate_values=True)
        self.assertTrue(torch.isfinite(logits).all() and torch.isfinite(penalty))

    def test_checkpoint_roundtrip(self):
        model = InputJointSelectionV4Fusion(7).eval()
        model.gates.begin_epoch(10, 5, 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dynamic.pt'
            torch.save(dict(architecture=model.ARCHITECTURE, model_config=dict(num_classes=7, gate_temperature=.67), model=model.state_dict()), path)
            restored, _ = load_checkpoint(path)
            restored.eval()
            x = self.input()
            with torch.no_grad():
                torch.testing.assert_close(restored(x), model(x))

    @unittest.skipUnless(hasattr(torch._dynamo.config, 'fail_on_recompile_limit_hit'), 'Server PyTorch required for strict compile regression')
    def test_compile_capture_and_failure_propagation(self):
        graphs = []
        def backend(graph, inputs):
            graphs.append(str(graph.graph).lower())
            return graph.forward
        torch._dynamo.reset()
        model = InputJointSelectionV4Fusion(7).train()
        model.gates.begin_epoch(10, 5, 5)
        try:
            with torch._dynamo.config.patch(suppress_errors=False, fail_on_recompile_limit_hit=True, allow_unspec_int_on_nn_module=True):
                compiled = torch.compile(model, backend=backend)
                logits, penalty = compiled(self.input(), return_selection=True)
                (logits.square().mean()+.01*penalty).backward()
            captured = ' '.join(graphs)
            for key in ('router', 'node_embedding', 'body_blocks', 'hand_ctr_blocks', 'hand_to_local', 'local_projection', 'classifier'):
                self.assertIn(key, captured)
            self.assertNotIn('face_st_blocks', captured)  # Original eager boundary.
            with patch('torch.compile', side_effect=RuntimeError('compiler failed')):
                with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
                    compile_model(model, 'default')
            self.assertFalse(torch._dynamo.config.suppress_errors)
            self.assertTrue(torch._dynamo.config.fail_on_recompile_limit_hit)
        finally:
            torch._dynamo.reset()


if __name__ == '__main__':
    unittest.main()
