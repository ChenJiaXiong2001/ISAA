import torch,numpy as np,csv
from pathlib import Path
root=Path('/home/xiong/Desktop/ISAA/experiments/training_route_mismatch_20261106');out=Path('/home/xiong/Desktop/ISAA/outputs');z=np.load(root/'paired_validation_20261106/paired_branch_logits.npz');rows=list(csv.DictReader(open(root/'paired_validation_20261106/paired_samples.csv')));y=np.array([int(r['true_class_index']) for r in rows]);cond=np.array([r['condition'] for r in rows]);
def w(sub):
 ck=torch.load(out/sub/'last.pt',map_location='cpu',weights_only=False)['model'];h=ck['routing_hand_face_correct'].float();b=ck['routing_body_face_correct'].float();return torch.where(h+b>0,h/(h+b),torch.full_like(h,.5)).numpy()
wc=w('soft_route_matched_control_20ep4096_20261106/20261106-232005-85358121');wn=w('soft_route_candidate_20ep4096_20261106/20261106-233445-24e4292d');a=z['candidate'];p=np.exp(a-a.max(2,keepdims=True));p/=p.sum(2,keepdims=True);predc=(p[:,0]*wc[None,:]+p[:,1]*(1-wc)[None,:]).argmax(1);predn=(p[:,0]*wn[None,:]+p[:,1]*(1-wn)[None,:]).argmax(1)
for c in np.unique(cond):
 q=cond==c;cls=y[q]; diff=np.abs(wc[cls]-wn[cls]); bins=[diff<1e-6,(diff>1e-6)&(diff<.25),diff>=.25]
 print(c,[(int(b.sum()),100*np.mean(predc[q][b]==cls[b]),100*np.mean(predn[q][b]==cls[b]),100*np.mean(predc[q][b]!=predn[q][b])) for b in bins])
