"""Real CUDA F/B for v2 warmup, soft/hard validation and compact input."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaa.train import _prefer_bundled_cuda_libraries
import torch
from isaa.input_joint_training_v2 import compile_model, COMPILE_SCOPE, parse_args, make_optimizers
from isaa.models.input_joint_selection_v2 import InputJointSelectionV2Fusion
from isaa.models.input_joint_selection import OPTIONAL_BODY

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window-size',type=int,default=100)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Real CUDA required; no fallback')
    torch.set_num_threads(4)
    torch.manual_seed(1)
    torch.backends.cudnn.benchmark=True
    results=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                 compile=COMPILE_SCOPE,window_size=args.window_size,stages={})
    search=InputJointSelectionV2Fusion(60)
    search.gates.begin_epoch(10,5,5)
    with torch.no_grad():
        search.gates.mu.fill_(.2)
        search.gates.mu[::2]=.8
    subset=search.export_selection()
    for stage in ('select','compact'):
        model=(InputJointSelectionV2Fusion(60) if stage=='select' else
               InputJointSelectionV2Fusion(60,stage=stage,subset=subset)).cuda()
        compiled=compile_model(model,'default')
        backbone,gates=make_optimizers(model,parse_args([]))
        start=time.perf_counter()
        x=torch.randn(2,3,args.window_size,len(model.selection['input_indices']),2,device='cuda')*.05
        x[:,2]=1.; x[0,:,:,:,1]=0.
        mask=torch.ones(2,args.window_size,device='cuda',dtype=torch.bool); mask[0,-3:]=False
        grad_norms=[]
        for epoch in (1,6,10):
            active=model.gates.begin_epoch(epoch,5,5,gates) if stage=='select' else False
            compiled.train()
            backbone.zero_grad(set_to_none=True)
            if gates: gates.zero_grad(set_to_none=True)
            logits,penalty=compiled(x,mask,return_selection=True)
            loss=torch.nn.functional.cross_entropy(logits,torch.tensor([1,2],device='cuda'))+(.01*penalty if active else 0.*penalty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            if gates:
                norm=float(model.gates.mu.grad.norm())
                if epoch==1 and norm!=0.: raise AssertionError('Warmup gate gradient must be exactly zero')
                if active and norm==0.: raise AssertionError('No active compiled gate gradient')
                grad_norms.append(norm)
            backbone.step()
            if gates and active: gates.step(); model.gates.project_(gates)
            torch.cuda.synchronize()
            print(stage,'epoch',epoch,'compiled F/B loss',float(loss.detach()),flush=True)
        compiled.eval()
        for hard in (False,True) if gates else (False,):
            if gates: model.gates.hard_eval.fill_(hard)
            with torch.no_grad():
                predicted=compiled(x[:1],mask[:1])
                reference=model(x[:1],mask[:1])
            torch.testing.assert_close(predicted,reference,rtol=3e-3,atol=3e-4)
        if gates: model.gates.hard_eval.zero_()
        results['stages'][stage]=dict(compiled_forward_backward=True,steps=3,
            active_and_frozen_gate_gradients=grad_norms,eval_partial_batch_matches_eager=True,
            hard_eval_matches_eager=bool(gates),raw_input_count=len(model.selection['input_indices']),
            elapsed_seconds=time.perf_counter()-start)
        del compiled,model,backbone,gates,logits,penalty,loss,predicted,reference
        torch._dynamo.reset(); torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': main()
