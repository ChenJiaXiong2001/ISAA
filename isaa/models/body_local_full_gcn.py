"""Full GCN upgrade of the best wide-relative body/hand/face fusion model."""
from __future__ import annotations

import torch
from torch import nn

from isaa.graph.adjacency import build_joint_spatial_partitions
from isaa.graph.regions import build_region_partition
from isaa.layouts.rtmw_133 import register as register_rtmw_layout
from isaa.models.rtmw_local_ctr import FaceTokenCompression
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.backbones import CTRGCNBackbone, STGCNBackbone


FULL_GCN_VARIANT = "body-local-hand-ctr-wide-relative-full"


class BodyLocalFullGCNFusion(BodyLocalHandCTRWideRelativeFusion):
    """Preserve the best model's features/fusion, replace all three backbones."""

    ARCHITECTURE = "rtmw_full_ctr22_shared_handctr21_facest6_crosshanddistdir_v1"
    CHANNELS = CTRGCNBackbone.CHANNELS
    HAND_CHANNELS = CTRGCNBackbone.CHANNELS
    LOCAL_CHANNELS = STGCNBackbone.CHANNELS
    STRIDES = CTRGCNBackbone.STRIDES

    def __init__(self, num_classes=120, *, backbone_config=None, hand_input_channels=6):
        # Construct only the encoders used in forward; no inherited unused stack.
        nn.Module.__init__(self)
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if hand_input_channels not in (3, 6):
            raise ValueError("hand_input_channels must be 3 or 6")
        self.HAND_INPUT_CHANNELS = hand_input_channels
        config = dict(backbone_config or {})
        if set(config) - {"body", "hand", "face"}:
            raise ValueError("backbone_config only accepts body, hand and face")
        register_rtmw_layout()
        self.register_buffer("body_indices", torch.tensor(self.BODY_INDICES, dtype=torch.long))
        partition = build_region_partition("rtmw_133", 133)
        full_graph = build_joint_spatial_partitions(133, partition, "rtmw_133", scope="full")
        compressor = FaceTokenCompression(full_graph, torch.tensor(partition.joint_to_region))
        for name in ("face_indices", "face_members"):
            self.register_buffer(name, getattr(compressor, name).clone())
        self.register_buffer("face_original_to_token", compressor.original_to_token.clone())

        def induced(ids):
            ids = torch.tensor(ids, dtype=torch.long)
            graph = full_graph.index_select(1, ids).index_select(2, ids)
            return graph / graph.sum(-1, keepdim=True).clamp_min(1)

        self.body_encoder = CTRGCNBackbone(induced(self.BODY_INDICES), in_channels=3,
                                           **config.get("body", {}))
        self.hand_encoder = CTRGCNBackbone(induced(self.HAND_INDICES[:21]), in_channels=hand_input_channels,
                                           **config.get("hand", {}))
        # Contract all three face partitions. ST-GCN uses source,target order.
        face_graph = compressor.adjacency[:, 65:, 65:]
        self.face_encoder = STGCNBackbone(3, face_graph.transpose(-1, -2).contiguous(),
                                          **config.get("face", {}))
        self.CHANNELS = self.body_encoder.channels
        self.HAND_CHANNELS = self.hand_encoder.channels
        self.LOCAL_CHANNELS = self.face_encoder.channels
        self.hand_to_local = (nn.Identity() if self.HAND_CHANNELS[-1] == self.LOCAL_CHANNELS[-1]
                              else nn.Conv2d(self.HAND_CHANNELS[-1], self.LOCAL_CHANNELS[-1], 1, bias=False))
        self.body_projection = nn.Sequential(nn.Linear(self.CHANNELS[-1], 192), nn.ReLU())
        self.local_projection = nn.Sequential(nn.Linear(self.LOCAL_CHANNELS[-1], 96), nn.ReLU())
        self.classifier = nn.Linear(288, num_classes)
        self.backbone_config = {
            "body": {"channels": list(self.CHANNELS), "strides": list(self.body_encoder.strides),
                     "adaptive": self.body_encoder.adaptive},
            "hand": {"channels": list(self.HAND_CHANNELS), "strides": list(self.hand_encoder.strides),
                     "adaptive": self.hand_encoder.adaptive},
            "face": {"channels": list(self.LOCAL_CHANNELS), "strides": list(self.face_encoder.strides),
                     "dropout": config.get("face", {}).get("dropout", 0.0)},
        }

    def _run_body(self, body, mask):
        return self.body_encoder.forward_masked(body, mask)

    def _run_hand(self, hand, mask):
        return self.hand_encoder.forward_masked(hand, mask)

    def _run_face(self, face, mask):
        return self.face_encoder.forward_masked(face, mask)
