import csv, json, zipfile, io, re
from collections import defaultdict
root='/home/xiong/Desktop/ISAA/outputs'
canon=root+'/canonical_p10_pair_full_20261106/predictions.csv'
files={
'cw_clean':root+'/temporal_corruption_cw_singlepass_full_20261106/predictions_clean.csv',
'cw_p05':root+'/temporal_corruption_cw_singlepass_full_20261106/predictions_point_0.05.csv',
'cw_p10':root+'/temporal_corruption_cw_singlepass_full_20261106/predictions_point_0.1.csv',
'none_clean':root+'/temporal_corruption_none_center_full_20261106/predictions_clean.csv',
'none_p05':root+'/temporal_corruption_none_center_full_20261106/predictions_point_0.05.csv',
'none_p10':root+'/temporal_corruption_none_center_full_20261106/predictions_point_0.1.csv'}
rows=list(csv.DictReader(open(canon,newline='')))
pred={k:list(csv.DictReader(open(p,newline=''))) for k,p in files.items()}
pat=re.compile(r'P(\d{3}).*A(\d{3})')
lengths={}
with zipfile.ZipFile('/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip') as z:
 names=set(z.namelist())
 for r in rows:
  n=r['sample']; n2=n if n in names else [x for x in names if x.endswith(n)][0]
  with z.open(n2) as f: lengths[n]=int(__import__('numpy').load(io.BytesIO(f.read()),allow_pickle=False)['keypoints'].shape[0])
agg={}
for i,r in enumerate(rows):
 m=pat.search(r['sample']); act=m.group(2); L=lengths[r['sample']]
 for scope,ok in [('all',True),('long',L>100)]:
  if not ok: continue
  d=agg.setdefault((scope,act),{'n':0,'Lsum':0})
  d['n']+=1; d['Lsum']+=L
  for c in ('clean','p05','p10'):
   d.setdefault('cw_'+c,0); d.setdefault('none_'+c,0)
   d['cw_'+c]+=int(pred['cw_'+c][i]['prediction']==r['label']); d['none_'+c]+=int(pred['none_'+c][i]['prediction']==r['label'])
out={}
for (scope,act),d in agg.items():
 d['Lmean']=d.pop('Lsum')/d['n']
 for c in ('clean','p05','p10'):
  d[c+'_delta_pp']=100*(d['cw_'+c]-d['none_'+c])/d['n']
 out.setdefault(scope,{})[act]=d
json.dump(out,open(root+'/low_cost_temporal_action_summary_20261107.json','w'),indent=2)
for scope in ('all','long'):
 print('SCOPE',scope)
 for act,d in sorted(out[scope].items(),key=lambda kv:kv[1]['clean_delta_pp'],reverse=True):
  print(act,d['n'],round(d['Lmean'],1),*[round(d[c+'_delta_pp'],2) for c in ('clean','p05','p10')])
