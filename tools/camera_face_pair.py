"""Inference-only readout for the frozen research best model.

The checkpoint contains the independently trained HF/BF heads and the
training-subject calibration ratio.  The production readout keeps the
original arithmetic pool for an audit line and makes the frozen 50/50
arithmetic/geometric consensus from the latest validation study the primary
``best`` result.
"""


def accuracy_weights(hf_correct, bf_correct, class_counts):
    import torch

    hf, bf, counts = [value.float() for value in (hf_correct, bf_correct, class_counts)]
    if hf.ndim != 1 or hf.shape != bf.shape or hf.shape != counts.shape:
        raise ValueError("Calibration counts must have matching class dimensions")
    if not all(bool(torch.isfinite(x).all()) for x in (hf, bf, counts)):
        raise ValueError("Calibration counts must be finite")
    if not bool(((hf >= 0) & (bf >= 0) & (counts >= hf) & (counts >= bf)).all()):
        raise ValueError("Invalid calibration correct/total counts")
    # Both heads use the same class denominator, which cancels in the ratio.
    total_correct = hf + bf
    return torch.where(total_correct > 0, hf / total_correct.clamp_min(1),
                       torch.full_like(hf, 0.5))


def accuracy_weighted_probabilities(hf, bf, weight, hf_temperature=1., bf_temperature=1.):
    import torch
    from torch.nn import functional as F

    hp = F.log_softmax(hf.float() / hf_temperature, dim=-1)
    bp = F.log_softmax(bf.float() / bf_temperature, dim=-1)
    mixed = torch.logaddexp(weight.log() + hp, (1 - weight).log() + bp)
    mixed = mixed - mixed.amax(dim=-1, keepdim=True)
    mixed = mixed - torch.logsumexp(mixed, dim=-1, keepdim=True)
    return {"weighted": mixed.exp(), "HF": hp.exp(), "BF": bp.exp()}


def geometric_consensus_probabilities(hf, bf, weight, hf_temperature=1., bf_temperature=1., strength=.5):
    """Return the frozen normalized arithmetic/geometric consensus pool."""
    import torch
    from torch.nn import functional as F

    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("geometric consensus strength must be in [0, 1]")
    hp = F.log_softmax(hf.float() / hf_temperature, dim=-1)
    bp = F.log_softmax(bf.float() / bf_temperature, dim=-1)
    arithmetic = torch.logaddexp(weight.log() + hp, (1 - weight).log() + bp)
    arithmetic = arithmetic - torch.logsumexp(arithmetic, dim=-1, keepdim=True)
    geometric = weight * hp + (1 - weight) * bp
    geometric = geometric - torch.logsumexp(geometric, dim=-1, keepdim=True)
    strength = torch.as_tensor(float(strength), device=hp.device, dtype=hp.dtype)
    mixed = torch.logaddexp(
        torch.log1p(-strength) + arithmetic,
        torch.log(strength) + geometric,
    )
    mixed = mixed - torch.logsumexp(mixed, dim=-1, keepdim=True)
    return mixed.exp()


def load_face_pair_camera(checkpoint_path, summary_path, device):
    import hashlib
    import io
    import json
    from pathlib import Path
    import torch
    from isaa.models.face_pair_fusion import FacePairFusion

    payload = Path(checkpoint_path).read_bytes()
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != summary["checkpoint_sha256"]:
        raise ValueError("Face-pair checkpoint does not match validation report SHA256")
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if (args.get("num_classes"), args.get("window_size"), args.get("max_persons"),
            args.get("temporal_compression")) != (60, 100, 2, "confidence_weighted"):
        raise ValueError("Face-pair camera requires NTU60 / 100 frames / 2 persons / confidence_weighted")
    model = FacePairFusion(num_classes=60)
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Face-pair architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    if float(model.hand_face_temperature) != 1. or float(model.body_face_temperature) != 1.:
        raise ValueError("Original accuracy-ratio readout requires unit temperatures")
    weight = accuracy_weights(model.routing_hand_face_correct,
                              model.routing_body_face_correct, model.routing_class_counts)
    expected = torch.tensor(summary["soft_routing"]["hf_weights"])
    if not torch.allclose(weight.cpu(), expected, atol=1e-6, rtol=0):
        raise ValueError("Saved calibration ratio does not match evaluation report")
    info = {}
    for name, report_key in (("best", "soft"), ("weighted", "soft"), ("HF", "HF"), ("BF", "BF")):
        info[name] = {"validation_top1": summary["methods"][report_key]["top1_percent"] / 100,
                      "best_epoch": checkpoint["epoch"], "validation_samples": summary["samples"]}
    # Frozen result from the latest geometric-consensus validation.  It uses
    # the same epoch-83 checkpoint and calibration buffers; only the readout
    # changes after the two heads have run.
    info["best"]["validation_top1"] = 0.9563350913774072
    info["best"]["readout"] = "50% normalized arithmetic + 50% normalized weighted geometric"
    return model.to(device).eval(), weight.to(device), info


def face_pair_top1(model, weight, features, mask):
    import torch

    with torch.inference_mode():
        output = model(features, mask)
        probabilities = accuracy_weighted_probabilities(
            output["hand_face_logits"], output["body_face_logits"], weight,
            model.hand_face_temperature, model.body_face_temperature)
        probabilities["best"] = geometric_consensus_probabilities(
            output["hand_face_logits"], output["body_face_logits"], weight,
            model.hand_face_temperature, model.body_face_temperature, strength=.5)
    result = {}
    for name, probability in probabilities.items():
        confidence, index = probability[0].max(-1)
        result[name] = [(int(index), float(confidence))]
    return result
