"""Single-subject, repetition-held-out calibration screen; never reads validation."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(logits, w):
    z = logits.astype(np.float64)
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    out = w * p[:, 0] + (1-w) * p[:, 1]
    return out / out.sum(axis=-1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration-dir', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    o = p.parse_args()
    if o.output_dir.exists():
        raise FileExistsError(o.output_dir)
    paths = (o.control, o.candidate)
    expected = ('8905eeabf263de20078fe85e4ba27809ca8c1a287defc6e62832aa985bd35a43',
                'daac65f9076f033679f5ff94fd39bef76b14ff0d569ef83731e8bb524b9fb3b4')
    if tuple(map(digest, paths)) != expected:
        raise ValueError('unexpected checkpoint')
    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    summary = json.loads((o.calibration_dir / 'summary.json').read_text())
    if (summary['control_sha256'], summary['candidate_sha256']) != expected or summary['partition'] != 'calibration':
        raise ValueError('cache checkpoint mismatch')
    with (o.calibration_dir / 'paired_samples.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    arrays = np.load(o.calibration_dir / 'paired_branch_logits.npz')
    if arrays['control'].shape != (len(rows), 2, 60) or arrays['candidate'].shape != (len(rows), 2, 60):
        raise ValueError('cache shape mismatch')
    clean = np.array([r['condition'] == 'clean' for r in rows])
    rows = [r for r in rows if r['condition'] == 'clean']
    if len(rows) != 351:
        raise ValueError('unexpected clean calibration count')
    labels = np.array([int(r['true_class_index']) for r in rows])
    members = [r['sample'] for r in rows]
    if hashlib.sha256('\n'.join(members).encode()).hexdigest() != summary['members_sha256']:
        raise ValueError('cache manifest mismatch')
    repetitions = []
    for member, label in zip(members, labels):
        match = re.search(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})', member)
        if not match or match.group(1) != '008' or match.group(3) != '025' or int(match.group(5))-1 != label:
            raise ValueError('unexpected subject, setup, or label')
        repetitions.append(int(match.group(4)))
    repetitions = np.array(repetitions)
    if set(repetitions) != {1, 2}:
        raise ValueError('unexpected repetitions')
    results = {'calibration_logits_sha256': digest(o.calibration_dir/'paired_branch_logits.npz'),
        'calibration_rows_sha256': digest(o.calibration_dir/'paired_samples.csv'),
        'members_sha256': summary['members_sha256'], 'checkpoints': list(expected),
        'models': {}, 'predeclared_selection': 'candidate alpha 1 or 5 requires pooled >=3 net and both folds nonnegative; tie by NLL then lower alpha'}
    csvrows=[]
    for name, ckpt in zip(('control', 'candidate'), checkpoints):
        x = arrays[name][clean]
        pred = x.argmax(-1)
        saved = ckpt['model']
        savedh = saved['routing_hand_face_correct'].numpy()
        savedb = saved['routing_body_face_correct'].numpy()
        savedn = saved['routing_class_counts'].numpy()
        recalch = np.bincount(labels[pred[:, 0] == labels], minlength=60)
        recalcb = np.bincount(labels[pred[:, 1] == labels], minlength=60)
        recalcn = np.bincount(labels, minlength=60)
        result = {'saved_count_comparison': {'total_differences': int((savedn != recalcn).sum()),
            'hf_correct_differences': int((savedh != recalch).sum()), 'bf_correct_differences': int((savedb != recalcb).sum())},
            'full_class_counts': {'min': int(savedn.min()), 'median': float(np.median(savedn)), 'max': int(savedn.max()),
              'zero': int((savedn==0).sum()), 'at_most_six': int((savedn<=6).sum())},
            'folds': []}
        for holdout in (1,2):
            train, test = repetitions != holdout, repetitions == holdout
            h=np.bincount(labels[train][pred[train,0] == labels[train]], minlength=60)
            b=np.bincount(labels[train][pred[train,1] == labels[train]], minlength=60)
            n=np.bincount(labels[train], minlength=60)
            fold={'heldout_repetition': holdout, 'fit_samples': int(train.sum()), 'test_samples': int(test.sum()),
                  'fit_missing_classes': int((n == 0).sum()), 'both_heads_zero_classes': int((h+b == 0).sum()), 'methods': {}}
            for alpha in (0,1,5,'equal'):
                w = np.full(60,.5) if alpha == 'equal' else np.divide(h+alpha,h+b+2*alpha,out=np.full(60,.5),where=(h+b+2*alpha)>0)
                probs = scores(x[test], w)
                guess = probs.argmax(-1)
                hits = guess == labels[test]
                fold['methods'][str(alpha)] = {'correct': int(hits.sum()), 'nll_sum': float(-np.log(probs[np.arange(len(guess)), labels[test]]).sum()),
                                                'zero_weights': int((w==0).sum()), 'one_weights': int((w==1).sum())}
                for member, y, v in zip(np.array(members)[test], labels[test], guess):
                    csvrows.append(dict(model=name, heldout_repetition=holdout, method=alpha, sample=member, label=int(y), prediction=int(v)))
            result['folds'].append(fold)
        for alpha in ('1','5','equal'):
            gains=[f['methods'][alpha]['correct']-f['methods']['0']['correct'] for f in result['folds']]
            result.setdefault('comparisons', {})[alpha]={'fold_net':gains, 'net':sum(gains),
                'pooled_nll':sum(f['methods'][alpha]['nll_sum'] for f in result['folds'])/len(labels)}
        results['models'][name]=result
    comparisons=results['models']['candidate']['comparisons']
    eligible=[alpha for alpha in ('1','5') if comparisons[alpha]['net']>=3 and min(comparisons[alpha]['fold_net'])>=0]
    selected=sorted(eligible,key=lambda a:(-comparisons[a]['net'],comparisons[a]['pooled_nll'],int(a)))
    results['selected_alpha']=int(selected[0]) if selected else None
    o.output_dir.mkdir(parents=True, exist_ok=False)
    (o.output_dir/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    with (o.output_dir/'predictions.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0])); w.writeheader(); w.writerows(csvrows)
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
