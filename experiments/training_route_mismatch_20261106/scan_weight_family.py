import torch, numpy as np, csv
from pathlib import Path
root=Path('experiments/training_route_mismatch_20261106')
for split in ['paired_calibration_20261106','paired_validation_20261106']:
 z=np.load(root/split/'paired_branch_logits.npz'); rows=list(csv.DictReader(open(root/split/'paired_samples.csv',encoding='utf8'))); y=np.array([int(r['true_class_index']) for r in rows]);
 print('\n',split,len(y))
 for model in ['control','candidate']:
  a=z[model]; p=np.exp(a-a.max(2,keepdims=True));p/=p.sum(2,keepdims=True)
  # fit accuracies on calibration only and use each split's own for descriptive
  hc=(a[:,0].argmax(1)==y);bc=(a[:,1].argmax(1)==y)
  count=np.bincount(y,minlength=60); h=np.bincount(y[hc],minlength=60); b=np.bincount(y[bc],minlength=60)
  print(model,'branch',hc.mean()*100,bc.mean()*100)
  for alpha in [0,.5,1,2,5]:
   for gamma in [0.5,1,2]:
    ha=(h+alpha)/(count+2*alpha); ba=(b+alpha)/(count+2*alpha); w=(ha**gamma)/(ha**gamma+ba**gamma+1e-12); w=np.where(count>0,w,.5)
    pred=(p[:,0]*w[None,:]+p[:,1]*(1-w)[None,:]).argmax(1)
    print(f'a{alpha:g} g{gamma:g} {np.mean(pred==y)*100:.4f}',end=' | ')
   print()


