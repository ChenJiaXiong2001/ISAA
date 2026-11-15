"""Load one exported full GCN independently for reuse or fine-tuning."""
from __future__ import annotations

import torch
from torch import nn
from .backbones import CTRGCNBackbone, STGCNBackbone
from .backbones.slimmable import forward_slim, STAGE_OF_LAYER


class AdaptiveGCNBranch(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.kind = checkpoint["kind"]
        self.stage_maxima = tuple(checkpoint["stage_maxima"])
        config = dict(channels=[self.stage_maxima[s] for s in STAGE_OF_LAYER],
                      strides=checkpoint["strides"])
        adjacency = checkpoint["weights"]["A"]
        self.encoder = (CTRGCNBackbone(adjacency, checkpoint["in_channels"], **config)
                        if self.kind == "ctr" else
                        STGCNBackbone(checkpoint["in_channels"], adjacency, **config))
        self.encoder.load_state_dict(checkpoint["weights"], strict=True)
        self.register_buffer("class_stage_widths", checkpoint["class_stage_widths"].clone())
        self.register_buffer("shared_stage_widths", checkpoint["shared_stage_widths"].clone())
        self.normalization_stats = checkpoint["normalization_stats"]

    def forward(self, x, mask=None, *, widths=None, class_index=None, update_bn=False):
        """Return padded max-channel features and mask; x is B x C x T x V.

        A single class or explicit profile applies to this call. For fine-tuning,
        use train() and update_bn=True, then recalibrate before evaluation.
        """
        if widths is not None and class_index is not None:
            raise ValueError("Specify widths or class_index")
        widths = tuple(self.shared_stage_widths.tolist() if widths is None and class_index is None
                       else self.class_stage_widths[class_index].tolist() if widths is None else widths)
        if len(widths) != 3 or any(type(w) is not int or w < 8 or w % 8 or w > cap
                                   for w, cap in zip(widths, self.stage_maxima)):
            raise ValueError("Expected three valid stage widths")
        if not update_bn:
            bank = self.normalization_stats.get(",".join(map(str, widths)))
            if bank is None:
                raise ValueError("This width profile has no calibrated BN; calibrate before inference")
            modules = dict(self.encoder.named_modules())
            with torch.no_grad():
                for name, values in bank.items():
                    for key, value in values.items():
                        getattr(modules[name], key).copy_(value)
        return forward_slim(self.encoder, x, mask, widths, kind=self.kind, update_bn=update_bn)

    @torch.no_grad()
    def calibrate_bn(self, batches, *, widths=None, class_index=None):
        """Fit this width's statistics from (features, node_mask) fitting batches."""
        if widths is not None and class_index is not None:
            raise ValueError("Specify widths or class_index")
        widths = tuple(self.shared_stage_widths.tolist() if widths is None and class_index is None
                       else self.class_stage_widths[class_index].tolist() if widths is None else widths)
        modules = {name: m for name, m in self.encoder.named_modules()
                   if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))}
        keys = ("running_mean", "running_var", "num_batches_tracked")
        original = {name: {k: getattr(m, k).clone() for k in keys} for name, m in modules.items()}
        settings = [(m, m.training) for m in self.modules()]
        momenta = {name: m.momentum for name, m in modules.items()}
        device = next(self.parameters()).device
        count = 0
        try:
            self.eval()
            for m in modules.values():
                m.reset_running_stats()
                m.train()
                m.momentum = None
            for x, mask in batches:
                self(x.to(device), None if mask is None else mask.to(device), widths=widths, update_bn=True)
                count += 1
            if not count:
                raise ValueError("Calibration needs fitting batches")
            self.normalization_stats[",".join(map(str, widths))] = {
                name: {k: getattr(m, k).detach().cpu().clone() for k in keys}
                for name, m in modules.items()}
        finally:
            for name, m in modules.items():
                for k in keys:
                    getattr(m, k).copy_(original[name][k])
                m.momentum = momenta[name]
            for m, state in settings:
                m.training = state

    def save(self, path):
        torch.save({"kind": self.kind, "weights": self.encoder.state_dict(),
                    "in_channels": self.encoder.in_channels, "strides": self.encoder.strides,
                    "stage_maxima": self.stage_maxima, "stage_layers": (4, 3, 3),
                    "class_stage_widths": self.class_stage_widths.cpu(),
                    "shared_stage_widths": self.shared_stage_widths.cpu(),
                    "normalization_stats": self.normalization_stats}, path)


def load_adaptive_branch(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return AdaptiveGCNBranch(checkpoint).to(device).eval()
