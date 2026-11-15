"""Analyze archived hard-route changes and frozen-logit objective gradients."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fused(h, b, w):
    hl, bl = F.log_softmax(h, -1), F.log_softmax(b, -1)
    value = torch.logaddexp(w.log() + hl, torch.log1p(-w) + bl)
    return value - torch.logsumexp(value, -1, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    for key in ("run-dir", "cache", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    o = parser.parse_args()
    if o.output.exists():
        raise FileExistsError(o.output)
    torch.set_num_threads(1)
    cp = o.run_dir / "best.pt"
    if sha(cp) != "aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7":
        raise ValueError("Wrong checkpoint")
    ck = torch.load(cp, map_location="cpu", weights_only=False)
    data = torch.load(o.cache, map_location="cpu", weights_only=True)
    section = json.loads((o.run_dir / "split_manifest.json").read_text(encoding="utf-8"))["calibration"]
    members = section["members"]
    md = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if data["checkpoint_sha256"] != sha(cp) or data["member_sha256"] != md or section["sha256"] != md:
        raise ValueError("Cache, checkpoint and manifest disagree")
    routing = []
    for epoch in range(1, 84):
        info = json.loads((o.run_dir / f"routing_epoch_{epoch:03d}.json").read_text(encoding="utf-8"))
        if info["epoch"] != epoch or len(info["classes"]) != 60:
            raise ValueError("Invalid routing history")
        routing.append(info)
    state = ck["model"]
    n, hc, bc = (state[k].float() for k in ("routing_class_counts", "routing_hand_face_correct", "routing_body_face_correct"))
    hard = state["hand_face_weight"].float()
    soft = torch.where(hc + bc > 0, hc / (hc + bc), 0.5)
    final = routing[-1]["classes"]
    if not torch.equal(hard, torch.tensor([x["hand_face_weight"] for x in final])) or not torch.equal(hc, torch.tensor([x["hand_face_correct"] for x in final])) or not torch.equal(bc, torch.tensor([x["body_face_correct"] for x in final])):
        raise ValueError("Checkpoint and epoch-83 routing history differ")
    active = [x for x in routing if x["active"]]
    changes, direct = [], []
    for earlier, later in zip(active, active[1:]):
        old = [x["hand_face_weight"] for x in earlier["classes"]]
        new = [x["hand_face_weight"] for x in later["classes"]]
        changes.append(sum(a != b for a, b in zip(old, new)))
        direct.append(sum({a, b} == {0.0, 1.0} for a, b in zip(old, new)))
    h, b, y = data["hf"].float(), data["bf"].float(), data["labels"].long()
    a, c = fused(h, b, hard), fused(h, b, soft)
    correct_a, correct_c = a.argmax(-1) == y, c.argmax(-1) == y
    weights = ck["args"]
    gradients = {}
    for full in (False, True):
        pair = []
        for route in (hard, soft):
            hx, bx = h.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
            loss = F.nll_loss(fused(hx, bx, route), y) * (weights["pair_fusion_loss_weight"] if full else 1.)
            if full:
                loss += weights["pair_head_loss_weight"] * (F.cross_entropy(hx, y) + F.cross_entropy(bx, y))
            hg, bg = torch.autograd.grad(loss, (hx, bx))
            pair.append(torch.cat((hg.flatten(), bg.flatten())))
        ga, gb = pair
        gradients["full_objective" if full else "fusion_only"] = {
            "cosine": float(F.cosine_similarity(ga, gb, dim=0)),
            "relative_l2_difference": float((ga - gb).norm() / ga.norm()),
            "hard_l2": float(ga.norm()), "soft_l2": float(gb.norm())}
    branch_h, branch_b = h.argmax(-1) == y, b.argmax(-1) == y
    true_hard = hard[y]
    result = {"checkpoint_sha256": sha(cp), "cache_sha256": sha(o.cache), "members_sha256": md,
              "source_zip_sha256": sha(o.run_dir / "source.zip"), "samples": len(y),
              "active_route_epochs": [x["epoch"] for x in active], "route_transitions": len(changes),
              "route_class_changes": {"total": sum(changes), "mean": sum(changes) / len(changes), "max": max(changes), "last_10": changes[-10:]},
              "direct_hf_bf_switches": {"total": sum(direct), "mean": sum(direct) / len(direct), "last_10": direct[-10:]},
              "epoch83_route_hf_bf_tied": [int((hard == 1).sum()), int((hard == 0).sum()), int((hard == .5).sum())],
              "epoch83_weight_mean_absolute_difference": float((hard - soft).abs().mean()),
              "loss_weights": {"fusion": weights["pair_fusion_loss_weight"], "each_head": weights["pair_head_loss_weight"]},
              "hard_correct": int(correct_a.sum()), "soft_correct": int(correct_c.sum()),
              "soft_corrected": int((~correct_a & correct_c).sum()), "soft_regressed": int((correct_a & ~correct_c).sum()),
              "hard_nll": float(F.nll_loss(a, y)), "soft_nll": float(F.nll_loss(c, y)),
              "true_class_hard_selected_wrong_other_correct": int((((true_hard == 1) & ~branch_h & branch_b) | ((true_hard == 0) & ~branch_b & branch_h)).sum()),
              "logit_gradients": gradients,
              "limitations": "Frozen calibration logits, routing and output gradients; two training calibration subjects, not training parameter gradients or an independent test."}
    o.output.parent.mkdir(parents=True, exist_ok=True)
    with o.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
