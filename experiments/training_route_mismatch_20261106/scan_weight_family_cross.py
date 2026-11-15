import numpy as np,csv
from pathlib import Path
root=Path('experiments/training_route_mismatch_20261106')
def load(split,model):
 z=np.load(root/split/'paired_branch_logits.npz'); rows=list(csv.DictReader(open(root/split/'paired_samples.csv',encoding='utf8'))); q=np.array([r['condition']=='clean' for r in rows]); y=np.array([int(r['true_class_index']) for r in rows]);a=z[model];p=np.exp(a-a.max(2,keepdims=True));p/=p.sum(2,keepdims=True);return a[q],p[q],y[q]
for model in ['control','candidate']:
 ac,pc,yc=load('paired_calibration_20261106',model); av,pv,yv=load('paired_validation_20261106',model)
 hc=ac[:,0].argmax(1)==yc;bc=ac[:,1].argmax(1)==yc; count=np.bincount(yc,minlength=60); h=np.bincount(yc[hc],minlength=60); b=np.bincount(yc[bc],minlength=60)
 print(model,'cal',hc.mean()*100,bc.mean()*100)
 for alpha in [0,.5,1,2,5]:
  for gamma in [.5,1,2,3]:
   ha=(h+alpha)/(count+2*alpha);ba=(b+alpha)/(count+2*alpha);w=np.where(count>0,(ha**gamma)/(ha**gamma+ba**gamma+1e-12),.5)
   pred=(pv[:,0]*w[None,:]+pv[:,1]*(1-w)[None,:]).argmax(1)
   print(f'a{alpha:g} g{gamma:g} {np.mean(pred==yv)*100:.4f}',end=' | ')
  print()

