"""Estimate class-level hand benefit without reading the official validation split."""
from __future__ import annotations

import math
import random
import re

import torch
from torch.nn import functional as F


def split_routing_calibration(members, fraction=0.2, seed=1):
    """Hold out entire subjects inside the training protocol, not random frames."""
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must be between zero and one")
    groups = {}
    for index, member in enumerate(members):
        match = re.search(r"P(\d{3})", member)
        if match is None:
            raise ValueError(f"Cannot recover subject ID from {member!r}")
        groups.setdefault(int(match.group(1)), []).append(index)
    subjects = sorted(groups)
    if len(subjects) < 2:
        raise ValueError("Class-hand training needs at least two training subjects")
    random.Random(seed).shuffle(subjects)
    count = min(len(subjects) - 1, max(1, round(len(subjects) * fraction)))
    calibration_subjects = set(subjects[:count])
    fit = sorted(i for subject, indices in groups.items()
                 if subject not in calibration_subjects for i in indices)
    calibration = sorted(i for subject, indices in groups.items()
                         if subject in calibration_subjects for i in indices)
    return fit, calibration, sorted(calibration_subjects)


def build_hand_requirements(full_logits, no_hand_logits, labels, *,
                            margin=0.0, temperature=0.1, min_samples=5,
                            metadata=None):
    if not math.isfinite(temperature) or temperature <= 0 or min_samples < 1:
        raise ValueError("temperature and min_samples must be positive")
    if not math.isfinite(margin):
        raise ValueError("margin must be finite")
    full, no_hand = full_logits.detach().cpu().double(), no_hand_logits.detach().cpu().double()
    labels = labels.detach().cpu().long()
    if (full.ndim != 2 or no_hand.shape != full.shape or labels.shape != (full.shape[0],)
            or full.shape[0] == 0 or full.shape[1] < 1):
        raise ValueError("Expected nonempty NxC logits and N labels")
    if not torch.isfinite(full).all() or not torch.isfinite(no_hand).all():
        raise ValueError("Requirement estimation received nonfinite logits")
    if labels.min() < 0 or labels.max() >= full.shape[1]:
        raise ValueError("Labels are outside the class range")
    full_loss = F.cross_entropy(full, labels, reduction="none")
    no_hand_loss = F.cross_entropy(no_hand, labels, reduction="none")
    full_right, no_hand_right = full.argmax(1) == labels, no_hand.argmax(1) == labels
    rows = []
    for k in range(full.shape[1]):
        selected = labels == k
        count = int(selected.sum())
        gain = float((no_hand_loss[selected] - full_loss[selected]).mean()) if count else None
        reliable = count >= min_samples
        score = float(torch.sigmoid(torch.tensor((gain - margin) / temperature))) if reliable else 1.0
        full_correct, no_hand_correct = int(full_right[selected].sum()), int(no_hand_right[selected].sum())
        rows.append({"class_index": k, "action_id": f"A{k + 1:03}", "samples": count,
                     "full_correct": full_correct, "no_hand_correct": no_hand_correct,
                     "accuracy_gain_percentage_points": 100 * (full_correct - no_hand_correct) / count if count else None,
                     "mean_loss_reduction": gain, "need_score": score,
                     "recommend_hand": score >= 0.5,
                     "reliable_estimate": reliable,
                     "helped_samples": int((full_right & ~no_hand_right & selected).sum()),
                     "harmed_samples": int((~full_right & no_hand_right & selected).sum())})
    return {"schema_version": 1, "num_classes": full.shape[1],
            "score_definition": "sigmoid((mean_CE_without_hand_minus_full - margin)/temperature)",
            "margin": margin, "temperature": temperature, "min_samples": min_samples,
            "metadata": metadata or {}, "classes": rows}
