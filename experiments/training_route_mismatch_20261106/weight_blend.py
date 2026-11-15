import torch,numpy as np,csv
from pathlib import Path
root=Path('/home/xiong/Desktop/ISAA/experiments/training_route_mismatch_20261106');out=Path('/home/xiong/Desktop/ISAA/outputs'); z=np.load(root/'paired_validation_20261106/paired_branch_logits.npz');rows=list(csv.DictReader(open(root/'paired_validation_20261106/paired_samples.csv')));y=np.array([int(r['true_class_index']) for r in rows]);cond=np.array([r['condition'] for r in rows]);
def weight(sub):
 ck=torch.load(out/sub/'last.pt',map_location='cpu',weights_only=False)['model'];h=ck['routing_hand_face_correct'].float();b=ck['routing_body_face_correct'].float();return torch.where(h+b>0,h/(h+b),torch.full_like(h,.5)).numpy()
wc=weight('soft_route_matched_control_20ep4096_20261106/20261106-232005-85358121'); wn=weight('soft_route_candidate_20ep4096_20261106/20261106-233445-24e4292d');a=z['candidate'];p=np.exp(a-a.max(2,keepdims=True));p/=p.sum(2,keepdims=True)
print('lambda candidate weight; overall and conditions')
for l in np.linspace(0,1,11):
 w=(1-l)*wc+l*wn; pred=(p[:,0]*w[None,:]+p[:,1]*(1-w)[None,:]).argmax(1); vals=[100*np.mean(pred==y)]+[100*np.mean(pred[cond==c]==y[cond==c]) for c in ['clean','point_0.05','point_0.1','group_0.05','group_0.1','anchor_pre_0.05','anchor_pre_0.1']];print(f'{l:.1f}',*[f'{v:.3f}' for v in vals])

