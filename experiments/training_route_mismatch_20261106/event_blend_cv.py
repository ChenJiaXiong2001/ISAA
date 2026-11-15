import numpy as np,csv,torch,re
from pathlib import Path
root=Path('/home/xiong/Desktop/ISAA/experiments/training_route_mismatch_20261106'); z=np.load(root/'paired_calibration_20261106/paired_branch_logits.npz'); rows=list(csv.DictReader(open(root/'paired_calibration_20261106/paired_samples.csv'))); q=np.array([r['condition']=='clean' for r in rows]);rows=[r for r,x in zip(rows,q) if x]; y=np.array([int(r['true_class_index']) for r in rows]); ev=np.array([re.search(r'(S\d{3}C\d{3}P\d{3}R\d{3}A\d{3})',r['sample']).group(1) for r in rows]);
def probs(a):
 p=np.exp(a-a.max(2,keepdims=True));return p/p.sum(2,keepdims=True)
pc=probs(z['control'][q]);pn=probs(z['candidate'][q]); lams=np.linspace(0,1,11);d={l:[] for l in lams};
for e in np.unique(ev):
 fit=ev!=e;held=~fit;weights=[]
 for p in [pc,pn]:
  hc=p[:,0].argmax(1)==y;bc=p[:,1].argmax(1)==y;cnt=np.bincount(y[fit],minlength=60);h=np.bincount(y[fit&hc],minlength=60);b=np.bincount(y[fit&bc],minlength=60);weights.append(np.where(h+b>0,h/(h+b),.5))
 for l in lams:
  w=(1-l)*weights[0]+l*weights[1];pred=(pn[held,0]*w[None,:]+pn[held,1]*(1-w)[None,:]).argmax(1);d[l].extend((pred==y[held]).tolist())
for l in lams: print(f'{l:.1f}',sum(d[l]),len(d[l]),100*np.mean(d[l]))
# per event means and paired differences versus l0
for l in lams:
 vals=[]
 for e in np.unique(ev):
  fit=ev!=e;held=~fit;weights=[]
  for p in [pc,pn]:
   hc=p[:,0].argmax(1)==y;bc=p[:,1].argmax(1)==y;cnt=np.bincount(y[fit],minlength=60);h=np.bincount(y[fit&hc],minlength=60);b=np.bincount(y[fit&bc],minlength=60);weights.append(np.where(h+b>0,h/(h+b),.5))
  w=(1-l)*weights[0]+l*weights[1];pred=(pn[held,0]*w[None,:]+pn[held,1]*(1-w)[None,:]).argmax(1);vals += [int(np.sum(pred==y[held]))]
 print('ev',f'{l:.1f}',sum(vals),len(vals),np.mean(vals))
