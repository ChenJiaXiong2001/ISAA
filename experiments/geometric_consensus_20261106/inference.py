"""Isolated post-forward fusion helper for the formal best-face-pair model.

The production model and training entry points are unchanged. This helper is
intended for an inference wrapper that has already run the compiled network.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def arithmetic_geometric_fusion(
    hand_face_logits: torch.Tensor,
    body_face_logits: torch.Tensor,
    hand_face_weights: torch.Tensor,
    strength: float = 0.5,
) -> torch.Tensor:
    """Return normalized log probabilities for the fixed geometric mixture."""
    if hand_face_logits.shape != body_face_logits.shape:
        raise ValueError("HF and BF logits must have the same shape")
    if hand_face_logits.ndim < 2 or hand_face_weights.shape != (hand_face_logits.shape[-1],):
        raise ValueError("Weights must have one value per class")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    weights = hand_face_weights.to(device=hand_face_logits.device, dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or bool((weights > 1).any()):
        raise ValueError("hand-face weights must be finite and in [0, 1]")
    hf = F.log_softmax(hand_face_logits.float(), dim=-1)
    bf = F.log_softmax(body_face_logits.float(), dim=-1)
    arithmetic = torch.logaddexp(weights.log() + hf, torch.log1p(-weights) + bf)
    arithmetic = arithmetic - torch.logsumexp(arithmetic, dim=-1, keepdim=True)
    geometric = weights * hf + (1.0 - weights) * bf
    geometric = geometric - torch.logsumexp(geometric, dim=-1, keepdim=True)
    strength_tensor = torch.as_tensor(float(strength), device=hf.device, dtype=hf.dtype)
    mixed = torch.logaddexp(
        torch.log1p(-strength_tensor) + arithmetic,
        torch.log(strength_tensor) + geometric,
    )
    return mixed - torch.logsumexp(mixed, dim=-1, keepdim=True)


def fuse_model_output(output: dict[str, torch.Tensor], hand_face_weights: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Fuse the two logits in a standard FacePairFusion output dictionary."""
    try:
        hand_face = output["hand_face_logits"]
        body_face = output["body_face_logits"]
    except KeyError as exc:
        raise ValueError("output must contain hand_face_logits and body_face_logits") from exc
    return arithmetic_geometric_fusion(hand_face, body_face, hand_face_weights, strength)
