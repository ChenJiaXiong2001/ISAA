"""Paired full validation screen for derived-geometry dropout candidate."""
from __future__ import annotations
import argparse,csv,hashlib,json,re,sys
from functools import partial
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); import isaa.train
from isaa.face_pair_training import compile_model
from isaa.models.face_pair_fusion import FacePairFusion
from isaa.train import collate_rtmw
from isaa.utils.seed import seed_everything
from tools.evaluate_face_pair_corruption import _soft_logits,_soft_weights
from experiments.derived_geometry_dropout_20261107.model import DerivedGeometryDropoutFacePairFusion
CONDS=(("clean",None,0.0,0),("point_0.05","joint_dropout_probability",.05,1),("point_0.1","joint_dropout_probability",.1,2),("group_0.05","grouped_joint_dropout_probability",.05,101),("group_0.1","grouped_joint_dropout_probability",.1,102))
def load(path,cls,device):
 ck=torch.load(path,map_location='cpu',weights_only=False); args=ck['args']
 expected_arch=cls.ARCHITECTURE
 if ck.get('architecture') != expected_arch:
  raise ValueError(f'Unexpected architecture for {path}: {ck.get("architecture")!r} != {expected_arch!r}')
 if not bool(args.get('compile')) or args.get('init_checkpoint') is not None or args.get('resume_checkpoint') is not None:
  raise ValueError(f'Checkpoint is not mandatory-compile scratch training: {path}')
 m=cls(args['num_classes']); m.load_state_dict(ck['model'],strict=True); weights=_soft_weights(m).to(device)
 return ck,args,compile_model(m.to(device).eval(),args['compile_mode']),weights

def _paired_subject_ci(old,new,labels,members,seed):
 subjects=[]
 for member in members:
  match=re.search(r'P(\d+)',member)
  if not match: raise ValueError(f'Validation member has no subject id: {member}')
  subjects.append(int(match.group(1)))
 subjects=np.asarray(subjects,dtype=np.int64)
 unique,group=np.unique(subjects,return_inverse=True)
 if len(unique)<2: raise ValueError('Need at least two validation subjects for bootstrap')
 delta=(new==labels).astype(np.int8)-(old==labels).astype(np.int8)
 sizes=np.bincount(group); net=np.bincount(group,weights=delta,minlength=len(unique))
 rng=np.random.default_rng(seed)
 draws=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=20000)
 ci=np.percentile(100*(draws@net)/(draws@sizes),[2.5,97.5]).tolist()
 return {'subjects':int(len(unique)),'bootstrap_replicates':20000,'bootstrap_seed':int(seed),'subject_cluster_bootstrap_95_ci_pp':ci}

def _args_signature(args):
 keys=('split','num_classes','window_size','max_persons','score_normalization','feature_mode','deterministic_temporal_crop','temporal_compression','max_samples','batch_size','test_batch_size','compile','compile_mode','tf32','drop_last','pair_calibration_fraction','pair_calibration_start_epoch','pair_head_loss_weight','pair_fusion_loss_weight')
 return {key:args.get(key) for key in keys}
def _member_sha(members):
 return hashlib.sha256('\n'.join(members).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--control',type=Path,required=True); p.add_argument('--candidate',type=Path,required=True); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--num-workers',type=int,default=8); p.add_argument('--bootstrap-seed',type=int,default=20261107); o=p.parse_args();
 if o.output_dir.exists(): raise FileExistsError(o.output_dir)
 if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for compiled paired evaluation')
 dev=torch.device('cuda'); c,ca,cm,cw=load(o.control,FacePairFusion,dev); n,na,nm,nw=load(o.candidate,DerivedGeometryDropoutFacePairFusion,dev)
 if _args_signature(ca)!=_args_signature(na): raise ValueError(f'Control/candidate args differ: {_args_signature(ca)} != {_args_signature(na)}')
 control_manifest=json.loads((o.control.parent/'split_manifest.json').read_text()); candidate_manifest=json.loads((o.candidate.parent/'split_manifest.json').read_text())
 if control_manifest['validation']['members']!=candidate_manifest['validation']['members']: raise ValueError('validation members differ')
 members=control_manifest['validation']['members']
 if ca['max_samples']!=0: raise ValueError(f'Full paired evaluation requires max_samples=0, got {ca["max_samples"]}')
 for manifest in (control_manifest,candidate_manifest):
  sections=[manifest[section]['members'] for section in ('fit','calibration','validation')]
  if any(len(values)!=len(set(values)) for values in sections): raise ValueError('Duplicate split member')
  if set(sections[0])&set(sections[1]) or set(sections[0])&set(sections[2]) or set(sections[1])&set(sections[2]): raise ValueError('Split sections overlap')
  for section in ('fit','calibration','validation'):
   item=manifest[section]
   if int(item.get('count',len(item['members'])))!=len(item['members']) or item.get('sha256')!=_member_sha(item['members']): raise ValueError(f'Invalid {section} manifest digest')
 if len(members)!=len(set(members)): raise ValueError('Duplicate validation member')
 members_sha=_member_sha(members)
 manifest_digests={'control':{s:control_manifest[s]['sha256'] for s in ('fit','calibration','validation')},'candidate':{s:candidate_manifest[s]['sha256'] for s in ('fit','calibration','validation')}}
 rows=[]; results=[]; reference_labels=None
 for name,key,prob,seed in CONDS:
  seed_everything(o.bootstrap_seed+seed)
  ds=__import__('isaa.data.rtmw_zip_dataset',fromlist=['RTMWZipDataset']).RTMWZipDataset(o.archive,split='val',split_protocol=ca['split'],window_size=ca['window_size'],num_joints=133,num_classes=ca['num_classes'],layout='rtmw_133',max_persons=ca['max_persons'],score_normalization=ca.get('score_normalization','auto'),feature_mode=ca['feature_mode'],deterministic_temporal_crop=ca.get('deterministic_temporal_crop','center'),max_samples=0,augment=key is not None,temporal_compression=ca['temporal_compression'],augmentation_config={'random_temporal_crop':False,**({key:prob} if key else {})})
  if list(ds.members)!=members: raise ValueError(f'Validation member ordering differs for {name}')
  loader_kwargs={'batch_size':32,'shuffle':False,'num_workers':o.num_workers,'collate_fn':partial(collate_rtmw,feature_mode='raw')}
  if o.num_workers:
   loader_kwargs.update(prefetch_factor=2,multiprocessing_context='spawn')
  loader=DataLoader(ds,**loader_kwargs)
  old=[]; new=[]; new_fixed=[]; labels=[]; offset=0; input_hash=hashlib.sha256(); mask_hash=hashlib.sha256()
  with torch.inference_mode():
   for feat,y,mask in loader:
    input_hash.update(feat.contiguous().numpy().tobytes()); mask_hash.update((feat[:,2]>0).contiguous().numpy().tobytes()); mask_hash.update(mask.contiguous().numpy().tobytes())
    feat,y,mask=feat.to(dev),y.to(dev),mask.to(dev); control_output=cm(feat,mask); candidate_output=nm(feat,mask); a=_soft_logits(control_output,cw).argmax(-1).cpu().numpy(); b=_soft_logits(candidate_output,nw).argmax(-1).cpu().numpy(); bf=_soft_logits(candidate_output,cw).argmax(-1).cpu().numpy(); yy=y.cpu().numpy(); old.extend(a);new.extend(b);new_fixed.extend(bf);labels.extend(yy)
    offset+=len(yy)
  if reference_labels is None: reference_labels=list(labels)
  elif list(labels)!=reference_labels: raise ValueError(f'Validation label ordering differs for {name}')
  old=np.array(old);new=np.array(new);new_fixed=np.array(new_fixed);yy=np.array(labels); d=(new==yy).astype(int)-(old==yy).astype(int); df=(new_fixed==yy).astype(int)-(old==yy).astype(int); result={'condition':name,'samples':len(yy),'control_top1_percent':float((old==yy).mean()*100),'candidate_own_weights_top1_percent':float((new==yy).mean()*100),'candidate_control_fixed_weights_top1_percent':float((new_fixed==yy).mean()*100),'difference_pp':float(d.mean()*100),'control_fixed_difference_pp':float(df.mean()*100),'corrected':int((d==1).sum()),'regressed':int((d==-1).sum()),'control_fixed_corrected':int((df==1).sum()),'control_fixed_regressed':int((df==-1).sum()),'input_sha256':input_hash.hexdigest(),'frame_mask_sha256':mask_hash.hexdigest(),'members_sha256':members_sha,'bootstrap_own_weights':_paired_subject_ci(old,new,yy,members,o.bootstrap_seed+seed),'bootstrap_control_fixed_weights':_paired_subject_ci(old,new_fixed,yy,members,o.bootstrap_seed+1000+seed)}; results.append(result)
  for i,(member,a,b,bf,l) in enumerate(zip(members,old,new,new_fixed,yy)): rows.append({'condition':name,'sample_index':i,'member':member,'label':int(l),'control_prediction':int(a),'candidate_prediction':int(b),'candidate_control_fixed_prediction':int(bf)})
 o.output_dir.mkdir(parents=True)
 with (o.output_dir/'paired_samples.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (o.output_dir/'summary.json').write_text(json.dumps({'control':str(o.control),'candidate':str(o.candidate),'control_args':_args_signature(ca),'candidate_args':_args_signature(na),'validation_members_sha256':members_sha,'manifest_digests':manifest_digests,'fit_calibration_members_may_differ_due_to_scratch_seed':True,'conditions':results,'compile':'Inductor mandatory','bootstrap_unit':'complete validation subject','bootstrap_replicates':20000,'evaluation_batch_size':32},indent=2)+'\n')
 print(json.dumps(results,indent=2))
if __name__=='__main__': main()
