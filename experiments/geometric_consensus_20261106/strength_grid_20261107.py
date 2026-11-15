import argparse,csv,hashlib,json,math,re,sys,time
from pathlib import Path
ROOT=Path('/home/xiong/Desktop/ISAA'); sys.path.insert(0,str(ROOT)); import isaa.train
import numpy as np, torch
from torch.nn import functional as F
from isaa.face_pair_training import _dataset,_loader,compile_model,compile_scope
from isaa.models.face_pair_fusion import FacePairFusion
from tools.check_face_pair_checkpoint import check_args,check_compile
EXPECTED='aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7'
checkpoint=Path('/home/xiong/Desktop/ISAA/outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490/best.pt'); archive=Path('/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip'); outdir=Path('/home/xiong/Desktop/ISAA/outputs/geometric_strength_grid_epoch83_20261107')
if outdir.exists(): raise FileExistsError(outdir)
if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=EXPECTED: raise ValueError('checkpoint')
ck=torch.load(checkpoint,map_location='cpu',weights_only=True); saved=ck['args']; check_args(saved,'checkpoint.args',expected_variant='best-face-pair'); check_compile(ck['compile'],saved,'checkpoint.compile')
model=FacePairFusion(saved['num_classes']); model.load_state_dict(ck['model'],strict=True); counts=model.routing_class_counts.float(); h=model.routing_hand_face_correct.float(); b=model.routing_body_face_correct.float(); w=torch.where(h+b>0,h/(h+b),torch.full_like(h,.5)).cuda()
args=argparse.Namespace(**saved); args.archive=str(archive); args.test_batch_size=64; args.num_workers=0; torch.set_num_threads(4); torch.backends.cudnn.benchmark=True; torch.backends.cudnn.allow_tf32=args.tf32; torch.backends.cuda.matmul.allow_tf32=args.tf32
data=_dataset(args,'val',False); manifest=json.loads((checkpoint.parent/'split_manifest.json').read_text()); members=list(data.members); assert members==manifest['validation']['members']; loader=_loader(args,data,torch.device('cuda'),False); compiled=compile_model(model.cuda().eval(),args.compile_mode)
strengths=[0.25,0.5,0.75,1.0]; correct={s:0 for s in strengths}; base=0; subjects=[]; net={s:[] for s in strengths}; started=time.perf_counter(); offset=0
with torch.inference_mode():
  for step,(features,labels,mask) in enumerate(loader,1):
    out=compiled(features.cuda(non_blocking=True),mask.cuda(non_blocking=True)); hf=F.log_softmax(out['hand_face_logits'].float(),-1); bf=F.log_softmax(out['body_face_logits'].float(),-1); ar=torch.logaddexp(w.log()+hf,torch.log1p(-w)+bf); ar=ar-torch.logsumexp(ar,-1,keepdim=True); geo=w*hf+(1-w)*bf; geo=geo-torch.logsumexp(geo,-1,keepdim=True); truth=labels.numpy(); basepred=ar.argmax(-1).cpu().numpy(); base += int((basepred==truth).sum()); subjects.extend([int(m.split("P",1)[1][:3]) for m in members[offset:offset+len(truth)]])
    for s in strengths:
      score=torch.logaddexp(torch.log1p(-torch.tensor(s,device='cuda'))+ar,torch.log(torch.tensor(s,device='cuda'))+geo); pred=score.argmax(-1).cpu().numpy(); correct[s]+=int((pred==truth).sum()); net[s].extend((pred==truth).astype(np.int8)-(basepred==truth).astype(np.int8))
    offset+=len(truth)
    if step==1 or step%50==0 or step==len(loader): print(f'progress={step}/{len(loader)} samples={offset}',flush=True)
unique,group=np.unique(subjects,return_inverse=True); rng=np.random.default_rng(20261107); draws=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=20000); result={'checkpoint_sha256':EXPECTED,'samples':offset,'baseline_correct':base,'baseline_accuracy_percent':100*base/offset,'strengths':{}}
for s in strengths:
 d=np.asarray(net[s]); gs=np.bincount(group,weights=d); sizes=np.bincount(group); boot=100*(draws@gs)/(draws@sizes); result['strengths'][str(s)]={'correct':correct[s],'accuracy_percent':100*correct[s]/offset,'difference_pp':100*d.mean(),'corrected':int((d==1).sum()),'regressed':int((d==-1).sum()),'subject_bootstrap_95_ci_pp':np.percentile(boot,[2.5,97.5]).tolist()}
outdir.mkdir(parents=True); (outdir/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))