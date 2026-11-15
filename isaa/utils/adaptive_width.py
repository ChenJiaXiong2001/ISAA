"""Train-only partitioning and bounded, measured search for GCN widths."""
from __future__ import annotations

import math
import random
import re
import torch
from torch import nn
from torch.nn import functional as F

from .hand_requirements import split_routing_calibration
from isaa.models.class_adaptive_gcn import relative_cost


def split_width_subjects(members, fraction=0.2, seed=1):
    fit, held_out, subjects = split_routing_calibration(members, fraction, seed)
    if len(subjects) < 2:
        raise ValueError("Width search needs two held-out subjects; increase --calibration-fraction")
    shuffled = list(subjects)
    random.Random(seed + 1).shuffle(shuffled)
    search_subjects = set(shuffled[:max(1, len(shuffled) // 2)])
    search, selection = [], []
    for i in held_out:
        subject = int(re.search(r"P(\d{3})", members[i]).group(1))
        (search if subject in search_subjects else selection).append(i)
    return fit, search, selection, {"search_subjects": sorted(search_subjects),
                                   "selection_subjects": sorted(set(subjects) - search_subjects)}


def distillation_loss(logits, labels, teacher=None, weight=0.5, temperature=2.0):
    loss = F.cross_entropy(logits, labels)
    if teacher is not None and weight:
        loss = loss + weight * temperature ** 2 * F.kl_div(
            (logits / temperature).log_softmax(-1),
            (teacher.detach() / temperature).softmax(-1), reduction="batchmean")
    return loss


def prediction_metrics(logits, labels, target=None):
    logits = logits.detach().cpu().double()
    labels = labels.detach().cpu().long()
    if len(labels) == 0 or not torch.isfinite(logits).all():
        raise ValueError("Search predictions must be nonempty and finite")
    prediction = logits.argmax(1)
    losses = F.cross_entropy(logits, labels, reduction="none")
    result = {"loss": float(losses.mean()), "top1": float((prediction == labels).double().mean())}
    if target is not None:
        positive, negative = labels == target, labels != target
        result["recall"] = float((prediction[positive] == target).double().mean()) if positive.any() else None
        result["false_positive_rate"] = float((prediction[negative] == target).double().mean()) if negative.any() else 0.0
        # Both positive and negative examples have full num_classes logits.
        positive_loss = losses[positive].mean() if positive.any() else losses.mean()
        negative_loss = losses[negative].mean() if negative.any() else positive_loss
        result["loss"] = float((positive_loss + negative_loss) / 2)
    return result


def greedy_width_search(evaluate, maximum, *, proposed=None, target=None, group_size=8,
                        max_steps=12, cost_weight=0.05, accuracy_tolerance=0.02,
                        loss_tolerance=0.03, maxima=(64, 128, 256), progress=None):
    """Validate learned proposals and refine with coarse/fine reductions.

    This is a bounded local search, not a global-optimality guarantee. Every
    acceptance is compared to the original maximum, preventing cumulative loss.
    The evaluator must use fitting-only BN statistics and search-only examples.
    """
    maximum = tuple(maximum)
    cache = {}
    def assess(profile):
        profile = tuple(profile)
        if profile not in cache:
            values = evaluate(profile)
            metrics = prediction_metrics(values[0], values[1], target)
            cost = relative_cost(profile, maxima)
            cache[profile] = {**metrics, "cost_proxy": cost, "objective": metrics["loss"] + cost_weight * cost}
        return cache[profile]
    reference = assess(maximum)
    def acceptable(metrics):
        if metrics["top1"] < reference["top1"] - accuracy_tolerance or metrics["loss"] > reference["loss"] + loss_tolerance:
            return False
        if target is not None and (metrics["recall"] < reference["recall"] - accuracy_tolerance
                                   or metrics["false_positive_rate"] > reference["false_positive_rate"] + accuracy_tolerance):
            return False
        return True
    current = maximum
    if proposed is not None:
        metrics = assess(proposed)
        if acceptable(metrics) and metrics["objective"] < reference["objective"]:
            current = tuple(proposed)
    for step in range(max_steps):
        candidates = set()
        for coordinate, width in enumerate(current):
            for smaller in (width - group_size, max(group_size, (width // 2 // group_size) * group_size)):
                if group_size <= smaller < width:
                    candidate = list(current)
                    candidate[coordinate] = smaller
                    candidates.add(tuple(candidate))
        viable = [candidate for candidate in sorted(candidates) if acceptable(assess(candidate))]
        if not viable:
            break
        winner = min(viable, key=lambda p: (assess(p)["objective"], assess(p)["cost_proxy"], p))
        if assess(winner)["objective"] >= assess(current)["objective"] - 1e-9:
            break
        current = winner
        if progress:
            progress(step + 1, current, assess(current), len(cache))
    return {"profile": list(current), "reference": reference, "selected": assess(current),
            "candidates_evaluated": len(cache), "search": "bounded_coarse_and_group_step_greedy"}


@torch.no_grad()
def calibrate_profile_bn(model, batches, profile, device):
    """Recompute statistics using fitting examples only; leave live BN intact."""
    if not batches:
        raise ValueError("BN calibration requires fitting batches")
    snapshot = model.bn_snapshot()
    modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    settings = [(m.training, m.momentum) for m in modules]
    try:
        for module in modules:
            module.reset_running_stats()
            module.training, module.momentum = True, None
        for x, _, mask in batches:
            model.execute(x.to(device), mask.to(device), tuple(profile), compiled=False, update_bn=True)
        return model.bn_snapshot()
    finally:
        model.load_bn(snapshot)
        for module, (training, momentum) in zip(modules, settings):
            module.training, module.momentum = training, momentum


def select_search_examples(labels, maximum_logits, target, *, positives=64, negatives=64, seed=1):
    labels, logits = torch.as_tensor(labels), maximum_logits.detach().cpu()
    positive = (labels == target).nonzero().flatten().tolist()
    random.Random(seed + target).shuffle(positive)
    negative = (labels != target).nonzero().flatten()
    # Include likely false positives, not only unrelated easy negatives.
    scores = logits.softmax(1)[negative, target]
    ordered = negative[scores.argsort(descending=True)].tolist()
    hard_count = min(negatives // 2, len(ordered))
    remainder = ordered[hard_count:]
    random.Random(seed + target + 1000).shuffle(remainder)
    return sorted(positive[:positives] + ordered[:hard_count] + remainder[:negatives - hard_count])
