import csv, json, zipfile, re
from pathlib import Path
root=Path('/home/xiong/Desktop/ISAA/outputs')
canon=root/'canonical_p10_pair_full_20261106/predictions.csv'
# temporal outputs are index keyed, order fixed by same validation manifest.
files={
 'cw_clean':root/'temporal_corruption_cw_singlepass_full_20261106/predictions_clean.csv',
 'cw_point05':root/'temporal_corruption_cw_singlepass_full_20261106/predictions_point_0.05.csv',
 'cw_point10':root/'temporal_corruption_cw_singlepass_full_20261106/predictions_point_0.1.csv',
 'none_clean':root/'temporal_corruption_none_center_full_20261106/predictions_clean.csv',
 'none_point05':root/'temporal_corruption_none_center_full_20261106/predictions_point_0.05.csv',
 'none_point10':root/'temporal_corruption_none_center_full_20261106/predictions_point_0.1.csv',
}
rows=list(csv.DictReader(open(canon,newline='')))
assert len(rows)==16306
pred={k:list(csv.DictReader(open(p,newline=''))) for k,p in files.items()}
for k,v in pred.items(): assert len(v)==len(rows),(k,len(v))
# Use canonical rows to provide sample path/label; temporal order is audited fixed manifest.
# Decode subject/action from NTU sample names.
pat=re.compile(r'P(\d{3}).*A(\d{3})')
for i,r in enumerate(rows):
    m=pat.search(r['sample']); r['_subject']=int(m.group(1)); r['_action']=int(m.group(2))
    for k,v in pred.items(): r[k]=int(v[i]['prediction'])
# lengths from archive; npz files hold arrays with first axis frame count.
archive='/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip'
lengths={}
with zipfile.ZipFile(archive) as z:
    names=set(z.namelist())
    import io,numpy as np
    for r in rows:
        n=r['sample']
        if n not in names:
            # zip may have leading slash/alternate path
            matches=[x for x in names if x.endswith(n)]
            if len(matches)!=1: raise KeyError((n,len(matches)))
            n=matches[0]
        with z.open(n) as f: data=np.load(io.BytesIO(f.read()), allow_pickle=False)
        # NPZ stores metadata plus tensors; keypoints defines temporal length.
        arr=data['keypoints']
        lengths[r['sample']]=int(arr.shape[0])
# Aggregate accuracy by original raw length bins.
bins=[(0,100),(101,150),(151,200),(201,300),(301,10**9)]
def bname(n):
    for lo,hi in bins:
        if lo<=n<=hi:return f'{lo}-{hi}'
    return 'other'
agg={}
for r in rows:
    bn=bname(lengths[r['sample']]); d=agg.setdefault(bn,{'n':0,'length_min':999999,'length_max':0})
    d['n']+=1; d['length_min']=min(d['length_min'],lengths[r['sample']]); d['length_max']=max(d['length_max'],lengths[r['sample']])
    for c in ('clean','point05','point10'):
        for mode in ('cw','none'):
            d.setdefault(f'{mode}_{c}',0)
            if r[f'{mode}_{c}']==int(r['label']):d[f'{mode}_{c}']+=1
# Aggregate direct mode switches and whether compression is relevant.
summary={'n':len(rows),'bins':agg,'switches':{}}
for c in ('clean','point05','point10'):
    s={'cw_correct':0,'none_correct':0,'cw_only':0,'none_only':0,'both_wrong':0,'both_correct':0}
    for r in rows:
        cw=r[f'cw_{c}']==int(r['label']); no=r[f'none_{c}']==int(r['label'])
        s['cw_correct']+=cw;s['none_correct']+=no;s['cw_only']+=cw and not no;s['none_only']+=no and not cw;s['both_wrong']+=not cw and not no;s['both_correct']+=cw and no
    summary['switches'][c]=s
# Ensure methods differ only in long clips: compute diff by length <=100.
summary['short_check']={}
for c in ('clean','point05','point10'):
    sub=[r for r in rows if lengths[r['sample']]<=100]
    summary['short_check'][c]={'n':len(sub),'prediction_diffs':sum(r[f'cw_{c}']!=r[f'none_{c}'] for r in sub)}
print(json.dumps(summary,ensure_ascii=False,indent=2))
Path('/home/xiong/Desktop/ISAA/outputs/low_cost_temporal_length_summary_20261107.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
