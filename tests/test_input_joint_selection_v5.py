import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from isaa.models.input_joint_selection_v5 import ConditionalJointGates,InputJointSelectionV5Fusion,INPUT_SELECTION_V5_VARIANT
from isaa.input_joint_training_v5 import parse_args,build_model,make_optimizers,compile_model,load_checkpoint,gate_weight
from train_best_input_selection_v5_100 import training_arguments


class InputJointSelectionV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def input():
        x=torch.randn(2,3,5,129,2)*.05
        x[:,2]=1.
        return x

    def test_entries_mandatory_compile_scratch_and_delayed_sparse_schedule(self):
        from isaa.train import parse_args as generic
        args=parse_args([])
        self.assertTrue(args.compile)
        self.assertEqual((args.epochs,args.window_size,args.batch_size),(100,100,32))
        self.assertEqual(args.gate_lr,.0001)
        self.assertFalse(hasattr(args,'keep_k'))
        with patch.object(sys,'argv',['main.py',*training_arguments([])]):
            self.assertEqual(generic().model_variant,INPUT_SELECTION_V5_VARIANT)
        for flags in (['--no-compile'],['--no-compile','--compile'],['--compile','--no-compile']):
            with self.assertRaises(ValueError): training_arguments(flags)
            with patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit): parse_args(flags)
            with patch.object(sys,'argv',['main.py','--model-variant='+INPUT_SELECTION_V5_VARIANT,*flags]),patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit): generic()
        for flag in ('--init-checkpoint','--resume-checkpoint','--npy-dir'):
            with self.assertRaises(ValueError): training_arguments([flag,'unused'])
            with patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit): parse_args([flag,'unused'])
        with patch('torch.load',side_effect=AssertionError('Must train from scratch')):
            model=build_model(args)
        backbone,router=make_optimizers(model,args)
        self.assertIsInstance(backbone,torch.optim.SGD)
        self.assertIsInstance(router,torch.optim.Adam)
        ids={id(p) for group in backbone.param_groups for p in group['params']}
        self.assertTrue(all(id(p) not in ids for p in model.gates.parameters()))
        for epoch in range(1,16): self.assertEqual(gate_weight(epoch,args),0.)
        self.assertAlmostEqual(gate_weight(16,args),.1/30)
        self.assertAlmostEqual(gate_weight(45,args),.1)

    def test_binary_no_floor_and_matched_expected_cost(self):
        g=ConditionalJointGates().train()
        d,c=torch.randn(16,40,8),torch.randn(16,8)
        g.begin_epoch(1,5)
        gates,cost,aux=g(d,c)
        torch.testing.assert_close(gates,torch.ones_like(gates))
        self.assertEqual(float(cost),1.)
        (gates.sum()+cost+aux['logit_guard_loss']).backward()
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in g.parameters()))
        g.zero_grad(set_to_none=True); g.begin_epoch(6,5)
        self.assertEqual(float(g.open_mix),0.)
        gates,cost,aux=g(d,c)
        self.assertTrue(set(gates.detach().flatten().tolist()).issubset({0.,1.}))
        torch.testing.assert_close(cost,(aux['gate_values']*g.cost).sum(-1).mean()/60.)
        # A lowered gate must lower the readout, not cancel against denominator.
        x=torch.ones(1,2,1,4); observed=torch.ones(1,1,1,4,dtype=torch.bool)
        small=InputJointSelectionV5Fusion._pool_gated_nodes(x*.2,observed,torch.full((1,1,1,4),.2))[0]
        torch.testing.assert_close(small,torch.full_like(small,.2))

    def test_saturation_guard_exploration_and_reopening_gradient(self):
        g=ConditionalJointGates().train(); g.begin_epoch(6,5)
        with torch.no_grad(): g.router[-1].bias.fill_(-40.)
        d,c=torch.randn(8,40,8),torch.randn(8,8)
        gates,cost,aux=g(d,c)
        self.assertGreaterEqual(float(aux['gate_values'].min()),.05)
        self.assertGreaterEqual(float(aux['gate_logits'].min()),-4.00001)
        self.assertGreater(float(aux['logit_guard_loss']),0.)
        (-aux['gate_values'].mean()).backward()
        self.assertLess(float(g.router[-1].bias.grad),-.001)
        g.eval();g.hard_eval.fill_(True)
        self.assertEqual(float(g(d,c)[0].sum()),0.)  # No enforced minimum K.
        g.router[-1].bias.data.fill_(40.)
        self.assertEqual(float(g(d,c)[0].sum()),320.)  # No forced pruning either.

    def test_expected_cost_agrees_with_monte_carlo_binary_sampling(self):
        torch.manual_seed(2)
        g=ConditionalJointGates().train();g.begin_epoch(6,5)
        with torch.no_grad():g.router[-1].weight.zero_();g.router[-1].bias.fill_(-.3)
        gates,cost,_=g(torch.zeros(8192,40,8),torch.zeros(8192,8))
        sampled=float((gates.detach()*g.cost).sum(-1).mean()/60.)
        self.assertAlmostEqual(sampled,float(cost),delta=.005)

    def test_closed_three_groups_classification_gradient_and_missing_observations(self):
        torch.manual_seed(3)
        model=InputJointSelectionV5Fusion(7).train();model.gates.begin_epoch(6,5)
        x=self.input();x[0,:,:,:,1]=0.;x[1,:,:,4]=float('nan')
        frame=torch.ones(2,5,dtype=torch.bool);frame[0,-1]=False
        # Force all Bernoulli draws closed without removing p's surrogate.
        with patch('torch.rand_like',side_effect=lambda t:torch.full_like(t,.999)):
            logits,_,aux=model(x,frame,return_gate_values=True)
        aux['gate_logits'].retain_grad()
        self.assertEqual(float(aux['sampled_gate_values'].detach().sum()),0.)
        torch.nn.functional.cross_entropy(logits,torch.tensor([1,2])).backward()
        for lo,hi in ((0,14),(14,34),(34,40)):
            self.assertGreater(float(aux['gate_logits'].grad[:,lo:hi].abs().sum()),0.)
        model.eval()
        with torch.no_grad():self.assertTrue(torch.isfinite(model(torch.full_like(x,float('nan')),frame)).all())

    def test_conditioning_controls_and_checkpoint_roundtrip(self):
        torch.manual_seed(5)
        m=InputJointSelectionV5Fusion(7).eval();m.gates.begin_epoch(6,5)
        x=self.input();g,_,_=m.route(x,x[:,2:3]>0)
        self.assertGreater(float((g[0]-g[1]).abs().max()),1e-6)
        self.assertEqual(g.shape,(2,40))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'m.pt'
            torch.save(dict(architecture=m.ARCHITECTURE,model_config=dict(num_classes=7),model=m.state_dict()),p)
            r,_=load_checkpoint(p);r.eval()
            with torch.no_grad():torch.testing.assert_close(m(x),r(x))
        m.gates.forced_mode.fill_(1)
        self.assertTrue((m.route(x,x[:,2:3]>0)[0]==1).all())
        m.gates.forced_mode.fill_(2)
        self.assertTrue((m.route(x,x[:,2:3]>0)[0]==0).all())

    def test_repeated_sparsity_pressure_does_not_destroy_recovery(self):
        torch.manual_seed(6)
        g=ConditionalJointGates();g.begin_epoch(6,5)
        opt=torch.optim.Adam(g.parameters(),lr=.01)
        d,c=torch.randn(8,40,8),torch.randn(8,8)
        for _ in range(200):
            opt.zero_grad();_,cost,a=g(d,c)
            (cost+.01*a['logit_guard_loss']).backward();opt.step()
        opt.zero_grad();_,_,a=g(d,c)
        self.assertTrue(torch.isfinite(a['gate_logits']).all())
        self.assertGreaterEqual(float(a['gate_values'].min()),.05)
        (-a['gate_values'].mean()).backward()
        self.assertGreater(float(g.router[-1].bias.grad.abs()),.001)

    @unittest.skipUnless(hasattr(torch._dynamo.config,'fail_on_recompile_limit_hit'),'Server runtime required')
    def test_compile_capture_and_failure_propagation(self):
        graphs=[]
        def backend(graph,inputs):
            graphs.append(str(graph.graph).lower());return graph.forward
        torch._dynamo.reset()
        model=InputJointSelectionV5Fusion(7).train();model.gates.begin_epoch(6,5)
        try:
            with torch._dynamo.config.patch(suppress_errors=False,fail_on_recompile_limit_hit=True,allow_unspec_int_on_nn_module=True):
                compiled=torch.compile(model,backend=backend)
                y,p,a=compiled(self.input(),return_gate_values=True)
                (y.square().mean()+.1*p+.01*a['logit_guard_loss']).backward()
            captured=' '.join(graphs)
            for key in ('descriptor_norm','context_norm','router','node_embedding','body_blocks','hand_ctr_blocks','classifier'):
                self.assertIn(key,captured)
            self.assertNotIn('face_st_blocks',captured)
            with patch('torch.compile',side_effect=RuntimeError('compile failed')):
                with self.assertRaisesRegex(RuntimeError,'compile failed'):compile_model(model,'default')
            import torch._functorch.config as aot_config
            self.assertFalse(aot_config.donated_buffer)
        finally:torch._dynamo.reset()


if __name__=='__main__':unittest.main()
