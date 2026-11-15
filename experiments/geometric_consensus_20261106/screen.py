"""Screen fixed geometric-consensus fusion using only calibration subjects."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from torch.nn import functional as F


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    report = json.loads(args.report.read_text(encoding='utf-8'))
    members = report['manifest']['calibration']['members']
    digest = hashlib.sha256('\n'.join(members).encode()).hexdigest()
    expected = 'aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7'
    if cache['partition'] != 'calibration' or cache['epoch'] != 83 or cache['checkpoint_sha256'] != expected or cache['member_sha256'] != digest:
        raise ValueError('Unexpected checkpoint or calibration members')
    hf, bf, labels = cache['hf'].float(), cache['bf'].float(), cache['labels'].long()
    if hf.shape != bf.shape or hf.ndim != 2 or len(labels) != len(members) or len(set(members)) != len(members):
        raise ValueError('Invalid input shapes or members')
    if not torch.isfinite(hf).all() or not torch.isfinite(bf).all():
        raise ValueError('Nonfinite logits')
    subjects = []
    for member, label in zip(members, labels.tolist()):
        subject = re.search(r'P(\d{3})', member)
        action = re.search(r'A(\d{3})', member)
        if not subject or not action or int(action.group(1)) != label + 1:
            raise ValueError('Invalid member labels')
        subjects.append(int(subject.group(1)))
    if sorted(set(subjects)) != [9, 18]:
        raise ValueError('Unexpected calibration subjects')
    subjects = torch.tensor(subjects)
    results = []
    for heldout in (9, 18):
        fit = subjects != heldout
        test = ~fit
        fit_labels, test_labels = labels[fit], labels[test]
        c = hf.shape[1]
        hc = torch.bincount(fit_labels[hf[fit].argmax(-1) == fit_labels], minlength=c).float()
        bc = torch.bincount(fit_labels[bf[fit].argmax(-1) == fit_labels], minlength=c).float()
        w = torch.where(hc + bc > 0, hc / (hc + bc).clamp_min(1), 0.5)
        hlog, blog = hf[test].log_softmax(-1), bf[test].log_softmax(-1)
        arithmetic = w * hlog.exp() + (1 - w) * blog.exp()
        arithmetic /= arithmetic.sum(-1, keepdim=True)
        geometric = (w * hlog + (1 - w) * blog).softmax(-1)
        baseline = arithmetic.argmax(-1) == test_labels
        for strength in (0.25, 0.5, 1.0):
            scores = (1 - strength) * arithmetic + strength * geometric
            if not torch.isfinite(scores).all() or not torch.allclose(scores.sum(-1), torch.ones(len(scores)), atol=1e-6):
                raise ValueError('Invalid candidate scores')
            candidate = scores.argmax(-1) == test_labels
            gain = int((candidate & ~baseline).sum())
            loss = int((baseline & ~candidate).sum())
            n = len(test_labels)
            results.append({'heldout_subject': heldout, 'fit_samples': int(fit.sum()), 'samples': n,
                'strength': strength, 'baseline_correct': int(baseline.sum()),
                'candidate_correct': int(candidate.sum()), 'gain': gain, 'loss': loss,
                'difference_pp': 100 * (gain - loss) / n,
                'baseline_nll': float(F.nll_loss(arithmetic.clamp_min(1e-30).log(), test_labels)),
                'candidate_nll': float(F.nll_loss(scores.clamp_min(1e-30).log(), test_labels))})
    summary = {'checkpoint_sha256': expected, 'cache_sha256': sha(args.cache),
        'calibration_members_sha256': digest, 'only_calibration_data': True,
        'method': 'LOSO; class accuracy ratio from fit subject; each head log-softmax; '
                  'normalized weighted arithmetic and geometric pools mixed using fixed strength',
        'strengths': [0.25, 0.5, 1.0], 'folds': results,
        'limitations': 'Exploratory comparison; only two calibration subjects, also used in original training routing; no official validation read.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
