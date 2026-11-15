"""Classifier and audited presets around the official SkeleT backbone."""
import torch
from torch import nn

from .backbone import SkeleT_GCN

REFERENCE_COMMIT = "fb2b2b23cd9665057c11ff87b3b0289a9821e964"
EXPRESSIVE_INDICES = tuple(range(23)) + tuple(range(91, 133))
VARIANTS = ("dgstgcn", "stgcn++", "ctrgcn", "ctrgcn-official-config")


def backbone_config(variant, num_person=2, instance_pooling=False):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown SkeleT backbone: {variant}")
    config = dict(
        init_graph_cfg=dict(layout="expressive-keypoints", mode="skelet_spatial"),
        in_channels=3, num_person=num_person,
        pooling_stages=[5, 8], pooling_joints=[27, 11],
        num_groups=[1, 1, 1, 1, 2, 2, 2, 4, 4, 4],
        data_bn_type="MVC" if variant.startswith("ctrgcn") else "VC",
        same_transformation=True, same_transformation_importance=True,
        M_adaptive="init", num_subsets=8 if variant == "dgstgcn" else 3,
        split_way="split", concat_way="concat+conv",
        gcn_type="dggcn" if variant == "dgstgcn" else
                 "unit_ctrgcn" if variant == "ctrgcn" else "unit_gcn",
        tcn_type="dgmstcn" if variant == "dgstgcn" else "mstcn",
        with_instance_pooling=instance_pooling,
    )
    if config["gcn_type"] == "unit_gcn":
        config["gcn_adaptive"] = "init"
    if instance_pooling:
        config.update(ip_pe="VC", ip_drop_path=0.1)
    return config


class SkeleTGCN(nn.Module):
    """B,C,T,V,M -> class logits, with V=133 or preselected expressive V=65.

    Coordinates must already use official PreNormalize2D and GenSkeFeat.
    The average readout deliberately includes padded persons, as in GCNHead.
    """
    def __init__(self, num_classes=60, variant="dgstgcn", num_person=2,
                 instance_pooling=False):
        super().__init__()
        if num_classes < 1 or num_person < 1:
            raise ValueError("num_classes and num_person must be positive")
        self.num_classes, self.num_person = num_classes, num_person
        self.variant, self.instance_pooling = variant, instance_pooling
        self.config = backbone_config(variant, num_person, instance_pooling)
        self.register_buffer("expressive_indices", torch.tensor(EXPRESSIVE_INDICES))
        self.backbone = SkeleT_GCN(**self.config)
        self.fc = nn.Linear(256, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[4] != self.num_person:
            raise ValueError(f"Expected B,3,T,V,{self.num_person}, got {tuple(x.shape)}")
        if x.shape[3] == 133:
            x = x.index_select(3, self.expressive_indices)
        elif x.shape[3] != 65:
            raise ValueError("SkeleT requires 133 WholeBody or 65 Expressive Keypoints")
        x = self.backbone(x.permute(0, 4, 2, 3, 1).contiguous())
        if x.ndim == 5:
            x = x.mean(dim=(-1, -2)).mean(dim=1)
        return self.fc(x)
