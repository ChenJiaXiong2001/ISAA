import csv,re,json,hashlib
from pathlib import Path
import numpy as np
p=Path('experiments/training_route_mismatch_20261106/paired_validation_20261106')
rows=list(csv.DictReader(open(p/'paired_samples.csv',encoding='utf8')))
rng=np.random.default_rng(20261109); out=[]
for cond in sorted(set(r['condition'] for r in rows)):
 rr=[r for r in rows if r['condition']==cond]; names=[]; vals=[]; ns=[]
 for r in rr:
  m=re.search(r'P(\d{3})R',r['sample']); names.append(m.group(1)); vals.append(int(int(r['shared_candidate_prediction'])==int(r['true_class_index'])) - int(int(r['control_prediction'])==int(r['true_class_index']))); 
 un=np.array(sorted(set(names))); d=np.array([sum(v for n,v in zip(names,vals) if n==u) for u in un],float); n=np.array([sum(1 for n0 in names if n0==u) for u in un],float)
 draws=rng.multinomial(len(un),np.full(len(un),1/len(un)),size=20000); boot=100*(draws@d)/(draws@n)
 out.append({'condition':cond,'samples':len(rr),'subjects':len(un),'weighted_difference_pp':100*d.sum()/n.sum(),'equal_subject_difference_pp':100*np.mean(d/n),'bootstrap_95_ci_pp':np.percentile(boot,[2.5,97.5]).tolist(),'corrected':int(sum(v==1 for v in vals)),'regressed':int(sum(v==-1 for v in vals)),'subject_differences_pp':{u:100*d[i]/n[i] for i,u in enumerate(un)}})
res={'method':'candidate logits fused with fixed control checkpoint gamma=1 calibration weights vs control own prediction','weight_provenance':'control matched hard-route checkpoint; weights read from its training calibration counters; no validation labels used to define weights','classification':'predeclared transfer-weight sensitivity in protocol, not independent efficacy evidence; candidate checkpoint and control share one short-run setup and validation partition','csv_sha256':hashlib.sha256((p/'paired_samples.csv').read_bytes()).hexdigest(),'results':out}
Path('experiments/training_route_mismatch_20261106/fixed_control_weight_bootstrap_20261107.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for x in out: print(x['condition'],x['weighted_difference_pp'],x['bootstrap_95_ci_pp'],x['equal_subject_difference_pp'],x['corrected'],x['regressed'])

