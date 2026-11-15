import csv,json,zipfile,io,re
from collections import defaultdict
root='/home/xiong/Desktop/ISAA/outputs'; canon=root+'/canonical_p10_pair_full_20261106/predictions.csv'
files={f'{m}_{c}':root+f'/temporal_corruption_{"cw_singlepass" if m=="cw" else "none_center"}_full_20261106/predictions_{"clean" if c=="clean" else "point_0.05" if c=="p05" else "point_0.1"}.csv' for m in ('cw','none') for c in ('clean','p05','p10')}
rows=list(csv.DictReader(open(canon,newline=''))); pred={k:list(csv.DictReader(open(v,newline=''))) for k,v in files.items()}
bins=[(0,100),(101,110),(111,120),(121,130),(131,140),(141,150),(151,160),(161,170),(171,180),(181,190),(191,200),(201,220),(221,1000)]
def bname(l):
 for a,b in bins:
  if a<=l<=b:return f'{a}-{b}'
with zipfile.ZipFile('/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip') as z:
 names=set(z.namelist()); agg={}
 for i,r in enumerate(rows):
  n=r['sample']; n2=n if n in names else next(x for x in names if x.endswith(n))
  with z.open(n2) as f:l=int(__import__('numpy').load(io.BytesIO(f.read()),allow_pickle=False)['keypoints'].shape[0])
  d=agg.setdefault(bname(l),{'n':0,'min':999,'max':0});d['n']+=1;d['min']=min(d['min'],l);d['max']=max(d['max'],l)
  for c in ('clean','p05','p10'):
   d.setdefault(c,{'cw':0,'none':0,'diff':0})
   d[c]['cw']+=int(pred['cw_'+c][i]['prediction']==r['label']); d[c]['none']+=int(pred['none_'+c][i]['prediction']==r['label']);d[c]['diff']+=int(pred['cw_'+c][i]['prediction']!=pred['none_'+c][i]['prediction'])
for b,d in agg.items():
 print(b,d['n'],d['min'],d['max'],*[f"{c}:{100*(d[c]['cw']-d[c]['none'])/d['n']:.2f}/{d[c]['diff']}" for c in ('clean','p05','p10')])
json.dump(agg,open(root+'/low_cost_temporal_length_fine_20261107.json','w'),indent=2)
