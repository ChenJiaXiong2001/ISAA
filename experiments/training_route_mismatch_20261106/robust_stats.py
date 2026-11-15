import torch,numpy as np,csv
from pathlib import Path
root=Path('/home/xiong/Desktop/ISAA/experiments/training_route_mismatch_20261106'); z=np.load(root/'paired_validation_20261106/paired_branch_logits.npz'); rows=list(csv.DictReader(open(root/'paired_validation_20261106/paired_samples.csv'))); y=np.array([int(r['true_class_index']) for r in rows]); cond=np.array([r['condition'] for r in rows]);
out=Path('/home/xiong/Desktop/ISAA/outputs')
for model,sub in [('control','soft_route_matched_control_20ep4096_20261106/20261106-232005-85358121'),('candidate','soft_route_candidate_20ep4096_20261106/20261106-233445-24e4292d')]:
 ck=torch.load(out/sub/'last.pt',map_location='cpu',weights_only=False)['model'];h=ck['routing_hand_face_correct'].float();b=ck['routing_body_face_correct'].float(); w=torch.where(h+b>0,h/(h+b),torch.full_like(h,.5)).numpy();a=z[model];p=np.exp(a-a.max(2,keepdims=True));p/=p.sum(2,keepdims=True);pred=(p[:,0]*w[None,:]+p[:,1]*(1-w)[None,:]).argmax(1); print('\n',model)
 for c in np.unique(cond):
  q=cond==c; hf=a[q,0].argmax(1)==y[q]; bf=a[q,1].argmax(1)==y[q]; f=pred[q]==y[q]; print(c,q.sum(),'HF %.3f BF %.3f fusion %.3f either %.3f both %.3f'%(100*hf.mean(),100*bf.mean(),100*f.mean(),100*np.mean(hf|bf),100*np.mean(hf&bf)))
 # confusion transitions branch to fusion
 hf=a[:,0].argmax(1)==y;bf=a[:,1].argmax(1)==y;f=pred==y
 print('fusion gains over HF/BF',np.sum(f&~hf),np.sum(hf&~f),np.sum(f&~bf),np.sum(bf&~f))

