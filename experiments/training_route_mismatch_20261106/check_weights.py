import torch,numpy as np,csv
from pathlib import Path
root=Path('/home/xiong/Desktop/ISAA/experiments/training_route_mismatch_20261106'); out=Path('/home/xiong/Desktop/ISAA/outputs')
for model in ['control','candidate']:
 ck=torch.load((out/('soft_route_candidate_20ep4096_20261106/20261106-233445-24e4292d' if model=='candidate' else 'soft_route_matched_control_20ep4096_20261106/20261106-232005-85358121'))/'last.pt',map_location='cpu',weights_only=False)['model']; c=ck['routing_class_counts'].float();h=ck['routing_hand_face_correct'].float();b=ck['routing_body_face_correct'].float(); w=torch.where(h+b>0,h/(h+b),torch.full_like(c,.5)).numpy(); print(model,'weight',w[:10], 'sum',c.sum(), 'nonhalf',np.sum(w!=.5))
 z=np.load(root/'paired_validation_20261106'/'paired_branch_logits.npz'); rows=list(csv.DictReader(open(root/'paired_validation_20261106'/'paired_samples.csv',encoding='utf8'))); y=np.array([int(r['true_class_index']) for r in rows]);q=np.array([r['condition']=='clean' for r in rows]); a=z[model][q]; p=np.exp(a-a.max(2,keepdims=True));p/=p.sum(2,keepdims=True);pred=(p[:,0]*w[None,:]+p[:,1]*(1-w)[None,:]).argmax(1); csvp=np.array([int(rows[i]['candidate_prediction' if model=='candidate' else 'control_prediction']) for i in np.where(q)[0]]);print('calc',np.mean(pred==y[q])*100,'csv',np.mean(csvp==y[q])*100,'mismatch',np.sum(pred!=csvp), 'first',pred[:10],csvp[:10])

