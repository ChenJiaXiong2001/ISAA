import torch,numpy as np,csv
from pathlib import Path
root=Path('/home/xiong/Desktop/ISAA/experiments/training_route_mismatch_20261106');out=Path('/home/xiong/Desktop/ISAA/outputs');
def load(split):
 z=np.load(root/split/'paired_branch_logits.npz');rows=list(csv.DictReader(open(root/split/'paired_samples.csv')));y=np.array([int(r['true_class_index']) for r in rows]);q=np.array([r['condition']=='clean' for r in rows]);a=z['candidate'][q];p=np.exp(a-a.max(2,keepdims=True));p/=p.sum(2,keepdims=True);return p,y[q]
def weight(sub):
 ck=torch.load(out/sub/'last.pt',map_location='cpu',weights_only=False)['model'];h=ck['routing_hand_face_correct'].float();b=ck['routing_body_face_correct'].float();return torch.where(h+b>0,h/(h+b),torch.full_like(h,.5)).numpy()
wc=weight('soft_route_matched_control_20ep4096_20261106/20261106-232005-85358121');wn=weight('soft_route_candidate_20ep4096_20261106/20261106-233445-24e4292d')
for split in ['paired_calibration_20261106','paired_validation_20261106']:
 p,y=load(split); print(split)
 for l in np.linspace(0,1,11):
  w=(1-l)*wc+l*wn; pred=(p[:,0]*w[None,:]+p[:,1]*(1-w)[None,:]).argmax(1);print(f'{l:.1f}:{np.mean(pred==y)*100:.3f}',end=' ')
 print()
