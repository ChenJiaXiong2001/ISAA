import csv,json,zipfile,io
root='/home/xiong/Desktop/ISAA/outputs'; canon=root+'/canonical_p10_pair_full_20261106/predictions.csv'; rows=list(csv.DictReader(open(canon,newline='')))
files={f'{m}_{c}':root+f'/temporal_corruption_{"cw_singlepass" if m=="cw" else "none_center"}_full_20261106/predictions_{"clean" if c=="clean" else "point_0.05" if c=="p05" else "point_0.1"}.csv' for m in ('cw','none') for c in ('clean','p05','p10')}; pred={k:list(csv.DictReader(open(v,newline=''))) for k,v in files.items()}
with zipfile.ZipFile('/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip') as z:
 names=set(z.namelist()); lengths=[]
 for r in rows:
  n=r['sample']; n2=n if n in names else next(x for x in names if x.endswith(n))
  with z.open(n2) as f:lengths.append(int(__import__('numpy').load(io.BytesIO(f.read()),allow_pickle=False)['keypoints'].shape[0]))
for t in [100,110,120,130,140,150,160,180,200]:
 print('threshold',t,'use cw if L>',t)
 for c in ('clean','p05','p10'):
  cor=0; dif=0
  for i,r in enumerate(rows):
   m='cw' if lengths[i]>t else 'none'; cor+=int(pred[m+'_'+c][i]['prediction']==r['label']); dif+=int(pred['cw_'+c][i]['prediction']!=pred['none_'+c][i]['prediction'] and lengths[i]>t)
  print(c,cor,100*cor/len(rows), 'used_diff',dif)
