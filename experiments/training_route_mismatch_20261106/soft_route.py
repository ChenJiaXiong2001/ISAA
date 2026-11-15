"""Isolated scratch-training candidate using lagged gamma=1 count weights."""
import torch
from torch.nn import functional as F

from isaa.models.face_pair_fusion import FacePairFusion


def gamma1_log_fusion(hf_logits, bf_logits, hf_correct, bf_correct, hf_temperature, bf_temperature):
    """Normalize class-weighted head probabilities without probability clipping."""
    h = hf_correct.to(dtype=torch.float32)
    b = bf_correct.to(dtype=torch.float32)
    total = h + b
    w = torch.where(total > 0, h / total.clamp_min(1), torch.full_like(total, 0.5))
    hf = F.log_softmax(hf_logits.float() / hf_temperature, dim=-1)
    bf = F.log_softmax(bf_logits.float() / bf_temperature, dim=-1)
    scores = torch.logaddexp(w.log() + hf, torch.log1p(-w) + bf)
    scores = scores - scores.amax(dim=-1, keepdim=True)
    return scores - torch.logsumexp(scores, dim=-1, keepdim=True)


class SoftRouteFacePair(FacePairFusion):
    """Keep baseline encoders and counts; change only the fusion readout."""

    ARCHITECTURE = "research_face_pair_lagged_gamma1_v1"

    def __init__(self, num_classes=60):
        super().__init__(num_classes=num_classes)
        self.model_config = {**self.model_config,
                             "classwise_selection": "previous_calibration_accuracy_ratio_gamma1",
                             "experimental_training_fusion": "previous_calibration_counts_gamma1_arithmetic",
                             "unseen_or_zero_count_weight": 0.5,
                             "hard_routing_buffer": "diagnostic_only_not_used_by_forward",
                             "checkpoint_architecture": self.ARCHITECTURE}

    def fuse_logits(self, hand_face_logits, body_face_logits):
        return gamma1_log_fusion(hand_face_logits, body_face_logits,
                                 self.routing_hand_face_correct, self.routing_body_face_correct,
                                 self.hand_face_temperature, self.body_face_temperature)
