import numpy as np,csv
from pathlib import Path
p=Path('experiments/training_route_mismatch_20261106/paired_validation_20261106')
z=np.load(p/'paired_branch_logits.npz'); x=z['control']; y=z['candidate']
rows=list(csv.DictReader(open(p/'paired_samples.csv',encoding='utf8')))
labels=np.array([int(r['true_class_index']) for r in rows]); cond=np.array([r['condition'] for r in rows]);
for name,a in [('control',x),('candidate',y)]:
 h=a[:,0].argmax(1); b=a[:,1].argmax(1)
 print(name,'HF',np.mean(h==labels)*100,'BF',np.mean(b==labels)*100,'both',np.mean((h==labels)&(b==labels))*100,'either',np.mean((h==labels)|(b==labels))*100,'disagree',np.mean(h!=b)*100,flush=True)
 for c in np.unique(cond):
  q=cond==c; print(' ',c,q.sum(),np.mean(h[q]==labels[q])*100,np.mean(b[q]==labels[q])*100,np.mean((h[q]==labels[q])|(b[q]==labels[q]))*100,flush=True)
for name,a in [('control',x),('candidate',y)]:
 a=a-a.max(2,keepdims=True); prob=np.exp(a);prob/=prob.sum(2,keepdims=True)
 print(name,flush=True)
 for w in np.linspace(0,1,21):
  pred=(w*prob[:,0]+(1-w)*prob[:,1]).argmax(1)
  print(f'{w:.2f} {np.mean(pred==labels)*100:.4f}',end='; ',flush=True)
 print()
