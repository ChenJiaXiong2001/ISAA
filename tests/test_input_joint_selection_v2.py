"""No fixed K: bounded STG, independent optimization and compile regressions."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from isaa.models.input_joint_selection_v2 import BoundedJointGates, InputJointSelectionV2Fusion, INPUT_SELECTION_V2_VARIANT
from isaa.models.input_joint_selection import OPTIONAL_BODY, selection_spec, PROTECTED_BODY
from isaa.input_joint_training_v2 import parse_args, build_model, make_optimizers, gate_weight, compile_model, load_checkpoint, export_checkpoint
from train_best_input_selection_v2_100 import training_arguments

class InputJointSelectionV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_strict_entries_and_config(self):
        from isaa.train import parse_args as generic_parse
        args=parse_args([])
        self.assertTrue(args.compile)
        self.assertEqual((args.epochs,args.window_size,args.gate_initial_mu,args.gate_lr), (100,100,.5,.001))
        self.assertEqual(args.sparsity_weight,.01)
        self.assertFalse(hasattr(args,'keep_k'))
        with patch.object(sys,'argv',['main.py',*training_arguments([])]):
            self.assertEqual(generic_parse().model_variant,INPUT_SELECTION_V2_VARIANT)
        with patch.object(sys,'argv',['main.py','--model-variant='+INPUT_SELECTION_V2_VARIANT]):
            self.assertTrue(generic_parse().compile)
        for flags in (['--no-compile'],['--no-compile','--compile'],['--compile','--no-compile']):
            with self.assertRaises(ValueError): training_arguments(flags)
            with patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit): parse_args(flags)
            with patch.object(sys,'argv',['main.py','--model-variant',INPUT_SELECTION_V2_VARIANT,*flags]),patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):
                generic_parse()
        for flag in ('--init-checkpoint','--resume-checkpoint','--npy-dir'):
            with self.assertRaises(ValueError): training_arguments([flag,'unused'])
            with patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit): parse_args([flag,'unused'])
        for flags in (['--gate-mu-max','3'],['--gate-lr','0'],['--gate-noise-ramp-epochs','0']):
            with patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit): parse_args(flags)

    def test_exact_open_frozen_warmup_and_reset(self):
        model=InputJointSelectionV2Fusion(7)
        args=parse_args([])
        _,opt=make_optimizers(model,args)
        g=model.gates
        initial=g.mu.detach().clone()
        for epoch in range(1,6):
            self.assertFalse(g.begin_epoch(epoch,5,5,opt))
            torch.testing.assert_close(g(),torch.ones_like(g.mu),rtol=0,atol=0)
            (g().sum()*2).backward()
            torch.testing.assert_close(g.mu.grad,torch.zeros_like(g.mu),rtol=0,atol=0)
            opt.zero_grad(set_to_none=True)
        torch.testing.assert_close(g.mu,initial,rtol=0,atol=0)
        with torch.no_grad(): g.mu.fill_(.9)
        self.assertTrue(g.begin_epoch(6,5,5,opt))
        torch.testing.assert_close(g.mu,initial,rtol=0,atol=0)
        self.assertAlmostEqual(float(g.open_mix),.8,places=6)
        self.assertAlmostEqual(float(g.noise_scale),.2,places=6)
        g.begin_epoch(10,5,5,opt)
        self.assertEqual(float(g.open_mix),0)
        self.assertEqual(float(g.noise_scale),1)

    def test_bound_projection_recovery_and_node_cost_gradient(self):
        g=BoundedJointGates()
        opt=torch.optim.Adam(g.parameters(),lr=.1)
        # Sustained classification-like pressure cannot send mu to 3 again.
        for _ in range(40):
            opt.zero_grad(); (-g.mu.sum()).backward(); opt.step(); g.project_(opt)
        self.assertLessEqual(float(g.mu.max().detach()),1.)
        self.assertGreaterEqual(float(g.mu.min().detach()),-.5)
        for _ in range(10):
            opt.zero_grad(); (.01*g.expected_optional_count()).backward(); opt.step(); g.project_(opt)
        self.assertLess(float(g.mu.max().detach()),1.)
        opt.zero_grad(); (.01*g.expected_optional_count()).backward()
        torch.testing.assert_close(g.mu.grad,g.sparse_gradient(.01))
        self.assertGreater(float(g.mu.grad.min()),1e-3)
        with torch.no_grad(): g.mu.fill_(.5)
        opt.zero_grad(); g.expected_optional_count().backward()
        self.assertAlmostEqual(float(g.mu.grad[-1]/g.mu.grad[0]),2.,places=5)

    def test_penalty_matches_actual_optional_count(self):
        model=InputJointSelectionV2Fusion(7).eval()
        model.gates.begin_epoch(10,5,5)
        x=torch.randn(2,3,5,129,2)*.05; x[:,2]=1.
        logits,penalty=model(x,return_selection=True)
        torch.testing.assert_close(penalty,model.gates.expected_optional_count())
        self.assertEqual(logits.shape,(2,7))
        self.assertEqual(gate_weight(5,parse_args([])),0.)
        self.assertAlmostEqual(gate_weight(25,parse_args([])),.01)

    def test_separate_gate_optimizer_and_scratch(self):
        args=parse_args([])
        with patch('torch.load',side_effect=AssertionError('No inherited weights')):
            model=build_model(args)
        backbone,gates=make_optimizers(model,args)
        self.assertIsInstance(backbone,torch.optim.SGD)
        self.assertIsInstance(gates,torch.optim.Adam)
        self.assertEqual(gates.param_groups[0]['lr'],.001)
        self.assertEqual(gates.param_groups[0]['weight_decay'],0.)
        self.assertNotIn(id(model.gates.mu),{id(p) for group in backbone.param_groups for p in group['params']})

    def test_hard_masking_and_variable_count(self):
        model=InputJointSelectionV2Fusion(7).eval()
        model.gates.begin_epoch(10,5,5)
        g=model.gates
        with torch.no_grad():
            g.mu.fill_(.2); g.mu[0]=.8; g.mu[len(OPTIONAL_BODY)+3]=.8
        self.assertEqual(model.export_selection()['raw_input_count'],78)
        with torch.no_grad(): g.mu[1]=.8
        self.assertEqual(model.export_selection()['raw_input_count'],79)
        g.hard_eval.fill_(True)
        self.assertEqual(set(g().tolist()),{0.,1.})
        raw=torch.randn(2,3,5,129,2)*.05; raw[:,2]=1.
        retained=set(model.export_selection()['input_indices'])
        dropped=[i for i,j in enumerate(model.selection['input_indices']) if j not in retained]
        changed=raw.clone(); changed[:,:,:,dropped]=float('nan')
        with torch.no_grad(): torch.testing.assert_close(model(raw),model(changed),rtol=0,atol=0)
        g.hard_eval.zero_()
        self.assertTrue(((g()>0)&(g()<1)).all())

    def test_v2_checkpoint_roundtrip_and_v1_manifest_compatibility(self):
        model=InputJointSelectionV2Fusion(7)
        model.gates.begin_epoch(10,5,5)
        with torch.no_grad(): model.gates.mu.fill_(.2)
        cfg=dict(num_classes=7,stage='select',subset=None,gate_sigma=.5,gate_initial_mu=.5,gate_mu_min=-.5,gate_mu_max=1.)
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'model.pt'; target=Path(tmp)/'subset.json'
            torch.save(dict(architecture=model.ARCHITECTURE,model=model.state_dict(),model_config=cfg,epoch=25),cp)
            restored,_=load_checkpoint(cp)
            self.assertEqual(restored.export_selection(),model.export_selection())
            spec=export_checkpoint(cp,target,.5)
            self.assertEqual(spec['raw_input_count'],75)
            self.assertFalse(spec['weights_transferred'])
            with self.assertRaises(FileExistsError): export_checkpoint(cp,target,.5)
        compact=InputJointSelectionV2Fusion(7,stage='compact',subset=selection_spec(sorted(PROTECTED_BODY),[0,4]))
        self.assertFalse(hasattr(compact,'gates'))
        self.assertEqual(compact.hand_ctr_blocks[0].gcn.static_topology.shape[-1],2)

    def test_compile_capture_and_failure_propagation(self):
        graphs=[]
        def backend(graph,inputs):
            graphs.append(str(graph.graph).lower()); return graph.forward
        model=InputJointSelectionV2Fusion(7).train()
        model.gates.begin_epoch(10,5,5)
        x=torch.randn(2,3,5,129,2)*.05; x[:,2]=1.
        torch._dynamo.reset()
        try:
            with torch._dynamo.config.patch(suppress_errors=False,fail_on_recompile_limit_hit=True,allow_unspec_int_on_nn_module=True):
                compiled=torch.compile(model,backend=backend)
                logits,penalty=compiled(x,return_selection=True)
                (logits.square().mean()+.01*penalty).backward()
            captured=' '.join(graphs)
            for key in ('gates','lerp','erf','body_blocks','hand_ctr_blocks','classifier'):
                self.assertIn(key,captured)
            self.assertTrue(torch.isfinite(model.gates.mu.grad).all())
            self.assertGreater(float(model.gates.mu.grad.abs().sum()),0.)
            with patch('torch.compile',side_effect=RuntimeError('compiler failed')):
                with self.assertRaisesRegex(RuntimeError,'compiler failed'): compile_model(model,'default')
            self.assertFalse(torch._dynamo.config.suppress_errors)
            self.assertTrue(torch._dynamo.config.fail_on_recompile_limit_hit)
        finally: torch._dynamo.reset()

if __name__=='__main__': unittest.main()
