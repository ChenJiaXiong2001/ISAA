"""Cheap post-hoc HF/BF disagreement and top-k reranking screen.

Runs only on cached paired branch logits; it does not touch training or checkpoints.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).with_name("paired_validation_20261106")
z = np.load(ROOT / "paired_branch_logits.npz")
rows = list(csv.DictReader((ROOT / "paired_samples.csv").open(encoding="utf-8")))
y = np.array([int(r["true_class_index"]) for r in rows])
cond = np.array([r["condition"] for r in rows])

def prob(a, temperature=1.0):
    a = a / temperature
    a = a - a.max(2, keepdims=True)
    p = np.exp(a)
    return p / p.sum(2, keepdims=True)

def topk(a, k):
    ix = np.argpartition(-a, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(a, ix, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(ix, order, axis=1), np.take_along_axis(vals, order, axis=1)

def metrics(pred, mask=None):
    if mask is None: mask = np.ones(len(y), dtype=bool)
    return float(np.mean(pred[mask] == y[mask]) * 100)

def run(name, a):
    p = prob(a)
    # Existing arithmetic probability pool, equal branch weights.
    base = (p[:, 0] + p[:, 1]).argmax(1)
    # Confidence selector: if heads disagree, choose the branch with larger
    # top-1 probability; tau allows a conservative margin before switching.
    h = p[:, 0].argmax(1); b = p[:, 1].argmax(1)
    ht = p[:, 0].max(1); bt = p[:, 1].max(1)
    out = {"name": name, "overall": {"base_equal_pool": metrics(base)}}
    for tau in np.linspace(0, .30, 16):
        choose_h = (ht - bt) >= tau
        pred = np.where(choose_h, h, b)
        # At ties and agreement, either branch is identical.
        out["overall"][f"selector_tau_{tau:.3f}"] = metrics(pred)
    # Classwise top-1 confidence calibrated by branch correctness on all rows:
    # a cheap oracle-like rule uses the margin between top1 and top2 as signal.
    _, hv = topk(p[:, 0], 2); _, bv = topk(p[:, 1], 2)
    hm = hv[:, 0] - hv[:, 1]; bm = bv[:, 0] - bv[:, 1]
    for tau in np.linspace(0, .30, 16):
        pred = np.where((hm - bm) >= tau, h, b)
        out["overall"][f"margin_selector_tau_{tau:.3f}"] = metrics(pred)
    out["conditions"] = {}
    for c in np.unique(cond):
        q = cond == c
        out["conditions"][c] = {"n": int(q.sum()), "base": metrics(base, q)}
        for tau in (0.0, .02, .05, .10, .15, .20):
            choose_h = (ht - bt) >= tau
            pred = np.where(choose_h, h, b)
            out["conditions"][c][f"selector_{tau:.2f}"] = metrics(pred, q)
            pred = np.where((hm - bm) >= tau, h, b)
            out["conditions"][c][f"margin_{tau:.2f}"] = metrics(pred, q)
    # Oracle upper bounds and disagreement diagnostics.
    out["diagnostics"] = {
        "hf": metrics(h), "bf": metrics(b),
        "both_correct": int(np.sum((h == y) & (b == y))),
        "either_correct": int(np.sum((h == y) | (b == y))),
        "disagree": int(np.sum(h != b)),
        "oracle_gain_pp": metrics(np.where(h == y, h, b)) - metrics(base),
    }
    return out

def weighted_eval(a, weights, name):
    """Evaluate fixed gamma=1 class weights and low-cost top-k bonuses."""
    p = prob(a)
    w = np.asarray(weights, dtype=np.float64)[None, :]
    fused = w * p[:, 0] + (1.0 - w) * p[:, 1]
    base = fused.argmax(1)
    h, _ = topk(p[:, 0], 5); b, _ = topk(p[:, 1], 5)
    rank_h = np.full((len(y), 60), 99, dtype=np.int16); rank_b = rank_h.copy()
    rank_h[np.arange(len(y))[:, None], h] = np.arange(5)[None, :]
    rank_b[np.arange(len(y))[:, None], b] = np.arange(5)[None, :]
    # Bonuses are added in log-probability space. Evaluate on all conditions,
    # then expose clean-tuned and leave-one-condition summaries separately.
    out = {"name": name, "base": metrics(base), "conditions": {}}
    for k in (1, 2, 3, 5):
        both = (rank_h < k) & (rank_b < k)
        either = (rank_h < k) | (rank_b < k)
        for alpha in (0.005, .01, .02, .05, .10, .20):
            score = np.log(fused + 1e-30) + alpha * both + (alpha / 4.0) * either
            pred = score.argmax(1)
            out[f"both_k{k}_a{alpha:g}"] = metrics(pred)
            out[f"either_k{k}_a{alpha:g}"] = metrics(pred)
    # Correctly compute the same candidates by condition for a few settings.
    for c in np.unique(cond):
        q = cond == c; out["conditions"][c] = {"n": int(q.sum()), "base": metrics(base, q)}
        for k, alpha in ((1,.02),(2,.02),(3,.02),(5,.02),(2,.05),(3,.05),(5,.05)):
            both = (rank_h < k) & (rank_b < k); either = (rank_h < k) | (rank_b < k)
            for mode, bonus in (("both", both), ("either", either)):
                score = np.log(fused + 1e-30) + alpha * bonus
                out["conditions"][c][f"{mode}_k{k}_a{alpha:g}"] = metrics(score.argmax(1), q)
    return out

result = {"control": run("control", z["control"]), "candidate": run("candidate", z["candidate"]),
          "n": len(y), "conditions": sorted(set(cond.tolist()))}
try:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.evaluate_face_pair_corruption import _soft_weights
    ckpt = Path('/home/xiong/Desktop/ISAA/outputs/soft_route_candidate_20ep4096_20261106/20261106-233445-24e4292d/last.pt')
    state = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
    counts, hf, bf = (state[k].float() for k in ('routing_class_counts','routing_hand_face_correct','routing_body_face_correct'))
    ha = torch.where(counts > 0, hf / counts.clamp_min(1), torch.zeros_like(counts))
    ba = torch.where(counts > 0, bf / counts.clamp_min(1), torch.zeros_like(counts))
    weights = torch.where(ha + ba > 0, ha / (ha + ba), torch.full_like(ha, .5)).numpy()
    result['candidate_weighted'] = weighted_eval(z['candidate'], weights, 'candidate_weighted')
except Exception as e:
    result['weighted_error'] = repr(e)
print(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
