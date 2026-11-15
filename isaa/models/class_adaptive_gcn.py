"""Class-specific stage widths, shared full GCN weights, and label-free routing."""
from __future__ import annotations

import torch
import time
from torch import nn

from .body_local_full_gcn import BodyLocalFullGCNFusion
from .backbones.slimmable import forward_slim, ctr_block, st_block, STAGE_OF_LAYER

ADAPTIVE_GCN_VARIANT = "body-local-class-adaptive-full-gcn"


class ClassAdaptiveGCN(BodyLocalFullGCNFusion):
    ARCHITECTURE = "rtmw_full_gcn_class_stage_width_v1"

    def __init__(self, num_classes=60, *, stage_maxima=(64, 128, 256), group_size=8,
                 router_hidden=128, confidence_threshold=0.8, fallback_threshold=0.45,
                 topk=3):
        maxima = tuple(stage_maxima)
        if (len(maxima) != 3 or any(type(v) is not int or v < 8 or v % 8 for v in maxima)
                or not maxima[0] < maxima[1] < maxima[2]):
            raise ValueError("stage_maxima must be three increasing multiples of eight")
        if type(group_size) is not int or group_size < 8 or group_size % 8 or any(v % group_size for v in maxima):
            raise ValueError("group_size must divide all stage maxima and be a multiple of eight")
        if not 0 <= fallback_threshold <= confidence_threshold <= 1 or topk < 1 or router_hidden < 1:
            raise ValueError("Invalid routing thresholds, topk or router_hidden")
        channels = [maxima[stage] for stage in STAGE_OF_LAYER]
        super().__init__(num_classes, backbone_config={name: {"channels": channels}
                                                     for name in ("body", "hand", "face")})
        self.stage_maxima = maxima
        self.group_size = group_size
        self.num_classes = num_classes
        self.confidence_threshold = float(confidence_threshold)
        self.fallback_threshold = float(fallback_threshold)
        self.topk = min(int(topk), num_classes)
        self.router_hidden = router_hidden
        # Eight temporal/person statistics for each body, hand and face token.
        self.router = nn.Sequential(nn.Linear((22 + 42 + 6) * 8, router_hidden),
                                    nn.LayerNorm(router_hidden), nn.ReLU(),
                                    nn.Linear(router_hidden, num_classes))
        logits = torch.zeros(num_classes, 9, maxima[-1] // group_size)
        for coordinate in range(9):
            logits[:, coordinate, maxima[coordinate % 3] // group_size - 1] = 4.0
        self.width_logits = nn.Parameter(logits)
        self.register_buffer("class_widths", torch.tensor(maxima * 3).repeat(num_classes, 1))
        self.register_buffer("shared_widths", torch.tensor(maxima * 3))
        self.register_buffer("profiles_ready", torch.tensor(False))
        self.register_buffer("router_temperature", torch.tensor(1.0))
        self._active_profile = maxima * 3
        self._active_gates = None
        self._update_bn = False
        self.normalization_stats = {}
        self._compiled = {}
        self._compiled_blocks = {}
        self._compile_active = False
        self.compile_enabled = False
        self.compile_mode = "reduce-overhead"
        self.compile_backend = "inductor"
        self.compile_limit = 64

    @property
    def max_profile(self):
        return self.stage_maxima * 3

    @property
    def min_profile(self):
        return (self.group_size,) * 9

    def model_config(self):
        return {"variant": ADAPTIVE_GCN_VARIANT, "num_classes": self.num_classes,
                "stage_maxima": list(self.stage_maxima), "group_size": self.group_size,
                "router_hidden": self.router_hidden, "confidence_threshold": self.confidence_threshold,
                "fallback_threshold": self.fallback_threshold, "topk": self.topk}

    def validate_profile(self, profile):
        profile = tuple(profile)
        if len(profile) != 9 or any(type(w) is not int or w < self.group_size or w % self.group_size
                                    or w > self.stage_maxima[i % 3] for i, w in enumerate(profile)):
            raise ValueError("A profile needs nine valid body/hand/face stage widths")
        return profile

    def set_profiles(self, table, shared):
        values = torch.as_tensor(table)
        if values.shape != (self.num_classes, 9) or values.is_floating_point() or values.dtype == torch.bool:
            raise ValueError("Class width table must have shape num_classes x 9 and integer entries")
        for row in values.tolist():
            self.validate_profile(row)
        self.class_widths.copy_(values)
        self.shared_widths.copy_(torch.tensor(self.validate_profile(shared), device=self.shared_widths.device))
        self.profiles_ready.fill_(True)

    def width_probabilities(self, coordinate):
        count = self.stage_maxima[coordinate % 3] // self.group_size
        return self.width_logits[:, coordinate, :count].softmax(-1)

    def proposed_profiles(self):
        return torch.stack([(self.width_probabilities(i).argmax(1) + 1) * self.group_size
                            for i in range(9)], dim=1)

    def expected_width_fraction(self, labels):
        values = []
        for i in range(9):
            p = self.width_probabilities(i).index_select(0, labels)
            widths = torch.arange(1, p.shape[1] + 1, device=p.device, dtype=p.dtype) / p.shape[1]
            values.append((p * widths).sum(1))
        return torch.stack(values, 1).mean()

    def _gates(self, labels, persons):
        result = []
        for i in range(9):
            p = self.width_probabilities(i).index_select(0, labels)
            survival = p.flip(1).cumsum(1).flip(1)
            if i < 6:
                gate = survival.repeat_interleave(self.group_size // 4, 1).repeat(1, 4)
            else:
                gate = survival.repeat_interleave(self.group_size, 1)
            result.append(gate.repeat_interleave(persons, 0))
        return result

    def _encode(self, encoder, x, mask, branch, kind):
        gates = None if self._active_gates is None else self._active_gates[branch * 3:branch * 3 + 3]
        executor = None
        if self._compile_active:
            def executor(block, value, inputs, outputs, adjacency, layer):
                return self._execute_block(block, value, inputs, outputs, adjacency,
                                           branch, kind, layer, self._update_bn)
        return forward_slim(encoder, x, mask, self._active_profile[branch * 3:branch * 3 + 3],
                            kind=kind, gates=gates, update_bn=self._update_bn,
                            block_executor=executor)

    def _execute_block(self, block, x, inputs, outputs, adjacency, branch, kind, layer, update):
        key = (branch, layer, len(inputs), len(outputs), update)
        first = key not in self._compiled_blocks
        if first:
            if kind == "ctr":
                def path(value, cin, cout, a):
                    return ctr_block(block, value, cin, cout, update)
            else:
                def path(value, cin, cout, a):
                    return st_block(block, value, a, cin, cout, update)
            options = {"backend": self.compile_backend, "dynamic": None}
            if self.compile_backend == "inductor":
                # Keep Inductor optimization; CUDA Graph capture is incompatible
                # with sandwich backward passes and mixed-profile accumulation.
                settings = dict(torch._inductor.list_mode_options(self.compile_mode))
                settings["triton.cudagraphs"] = False
                options["options"] = settings
            self._compiled_blocks[key] = torch.compile(path, **options)
            print(f"compile block start: branch={('body', 'hand', 'face')[branch]} "
                  f"layer={layer + 1}/10 channels={len(inputs)}->{len(outputs)} "
                  f"update_bn={update} cudagraphs=False", flush=True)
        started = time.perf_counter()
        value = self._compiled_blocks[key](x, inputs, outputs, adjacency)
        if first:
            print(f"compile block forward ready: branch={('body', 'hand', 'face')[branch]} "
                  f"layer={layer + 1}/10 seconds={time.perf_counter() - started:.1f}; "
                  "backward kernels compile on first backward", flush=True)
        return value

    def _run_body(self, x, mask):
        return self._encode(self.body_encoder, x, mask, 0, "ctr")

    def _run_hand(self, x, mask):
        return self._encode(self.hand_encoder, x, mask, 1, "ctr")

    def _run_face(self, x, mask):
        return self._encode(self.face_encoder, x, mask, 2, "st")

    def forward_fixed(self, x, frame_mask=None, profile=None, *, update_bn=False, labels=None,
                      compile_blocks=False):
        self._active_profile = self.max_profile if profile is None else profile
        self._update_bn = update_bn
        self._active_gates = None if labels is None else self._gates(labels, x.shape[-1] if x.ndim == 5 else 1)
        previous = self._compile_active
        self._compile_active = compile_blocks
        try:
            return super().forward(x, frame_mask)
        finally:
            self._compile_active = previous

    def enable_compile(self, enabled=True, *, mode="reduce-overhead", backend="inductor", limit=64):
        self.compile_enabled = bool(enabled)
        self.compile_mode, self.compile_backend, self.compile_limit = mode, backend, limit
        self._compiled.clear()
        self._compiled_blocks.clear()
        if enabled:
            # Bounded profile configurations share individual GCN block paths.
            for name in ("recompile_limit", "cache_size_limit"):
                if hasattr(torch._dynamo.config, name):
                    setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), limit * 30))

    def execute(self, x, mask, profile, *, compiled=True, update_bn=False, labels=None):
        profile = self.validate_profile(profile)
        key = (profile, update_bn)
        if self.compile_enabled and compiled:
            if key not in self._compiled and len(self._compiled) < self.compile_limit:
                self._compiled[key] = True
            if key in self._compiled:
                return self.forward_fixed(x, mask, profile, update_bn=update_bn, labels=labels,
                                          compile_blocks=True)
        return self.forward_fixed(x, mask, profile, update_bn=update_bn, labels=labels)

    def bn_snapshot(self):
        return {name: {key: getattr(module, key).detach().cpu().clone()
                       for key in ("running_mean", "running_var", "num_batches_tracked")}
                for name, module in self.named_modules() if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))}

    def load_bn(self, snapshot):
        modules = dict(self.named_modules())
        with torch.no_grad():
            for name, values in snapshot.items():
                for key, value in values.items():
                    getattr(modules[name], key).copy_(value)

    @staticmethod
    def profile_key(profile):
        return ",".join(map(str, profile))

    def apply_profile_bn(self, profile):
        bank = self.normalization_stats.get(self.profile_key(profile))
        if bank is not None:
            self.load_bn(bank)

    def route_logits(self, x, frame_mask=None):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] not in (3, 8) or x.shape[3] != 133:
            raise ValueError("Expected B x 3/8 x T x 133 x M")
        raw = x[:, :3]
        mask = (raw[:, 2:3] > 0) & torch.isfinite(raw).all(1, keepdim=True)
        if frame_mask is not None:
            mask = mask & frame_mask[:, None, :, None, None].bool()
        raw = torch.where(mask, raw, torch.zeros_like(raw))
        torso = torch.tensor((5, 6, 11, 12), device=x.device)
        points, weights = raw[:, :2].index_select(3, torso), mask.index_select(3, torso).to(raw.dtype)
        center = (points * weights).sum(3, keepdim=True) / weights.sum(3, keepdim=True).clamp_min(1)
        relative = torch.cat((raw[:, :2] - center, raw[:, 2:3]), 1).masked_fill(~mask, 0)
        ids = torch.tensor(self.HAND_INDICES, device=x.device)
        body, hands = raw.index_select(3, self.body_indices), relative.index_select(3, ids)
        bm, hm = mask.index_select(3, self.body_indices), mask.index_select(3, ids)
        b, _, t, _, m = raw.shape
        faces = relative.index_select(3, self.face_indices.flatten()).reshape(b, 3, t, 6, 12, m)
        fm = mask.index_select(3, self.face_indices.flatten()).reshape(b, 1, t, 6, 12, m)
        fm = fm & self.face_members[None, None, None, :, :, None]
        counts = fm.sum(4)
        faces = (faces * fm.to(faces.dtype)).sum(4) / counts.clamp_min(1)
        statistics = []
        for features, valid in ((body, bm), (hands, hm), (faces, counts > 0)):
            w = valid.to(features.dtype)
            count = w.sum((2, 4)).clamp_min(1)
            mean = (features * w).sum((2, 4)) / count
            variance = ((features - mean[:, :, None, :, None]).square() * w).sum((2, 4)) / count
            pair = w[:, :, 1:] * w[:, :, :-1]
            velocity = ((features[:, :2, 1:] - features[:, :2, :-1]).abs() * pair).sum((2, 4)) / pair.sum((2, 4)).clamp_min(1)
            statistics.append(torch.cat((mean, (variance + 1e-8).sqrt(), velocity), 1).flatten(1))
        return self.router(torch.cat(statistics, 1))

    def resolve_profiles(self, router_logits):
        probabilities = (router_logits / self.router_temperature).softmax(-1)
        confidence, predicted = probabilities.max(1)
        widths = self.class_widths.index_select(0, predicted)
        candidates = probabilities.topk(self.topk, dim=1).indices
        union = self.class_widths[candidates].amax(1)
        widths = torch.where((confidence < self.confidence_threshold)[:, None], union, widths)
        fallback = confidence < self.fallback_threshold
        widths = torch.where(fallback[:, None], self.shared_widths[None], widths)
        if not bool(self.profiles_ready):
            widths = self.shared_widths[None].expand_as(widths)
            fallback = torch.ones_like(fallback)
        return widths, confidence, fallback

    def forward(self, x, valid_frame_mask=None, *, return_routing=False, mode="adaptive"):
        if mode not in {"adaptive", "shared", "max"}:
            raise ValueError("mode must be adaptive/shared/max")
        if mode != "adaptive":
            profile = self.max_profile if mode == "max" else tuple(self.shared_widths.tolist())
            self.apply_profile_bn(profile)
            logits = self.execute(x, valid_frame_mask, profile)
            return {"logits": logits} if return_routing else logits
        routing = self.route_logits(x, valid_frame_mask)
        widths, confidence, fallback = self.resolve_profiles(routing)
        # Resolve missing banks before forming buckets; every example executes once.
        if self.normalization_stats:
            for row in torch.unique(widths, dim=0).detach().cpu().tolist():
                if self.profile_key(row) not in self.normalization_stats:
                    selected = (widths == widths.new_tensor(row)).all(1)
                    widths = torch.where(selected[:, None], self.shared_widths[None], widths)
                    fallback = fallback | selected
        logits = x.new_zeros((len(x), self.num_classes))
        for profile_list in torch.unique(widths, dim=0).detach().cpu().tolist():
            profile = tuple(profile_list)
            selected = (widths == widths.new_tensor(profile)).all(1).nonzero().flatten()
            self.apply_profile_bn(profile)
            result = self.execute(x.index_select(0, selected), None if valid_frame_mask is None else valid_frame_mask.index_select(0, selected), profile)
            logits = logits.index_copy(0, selected, result)
        if return_routing:
            return {"logits": logits, "router_logits": routing, "widths": widths,
                    "router_confidence": confidence, "fallback": fallback,
                    "width_fraction": (widths / widths.new_tensor(self.max_profile)).float().mean(1)}
        return logits


def relative_cost(profile, maxima=(64, 128, 256)):
    """Width-squared proxy, NOT measured MACs or hardware latency."""
    total = maximum = 0.0
    for branch in range(3):
        previous = previous_max = 3 if branch != 1 else 6
        for layer, stage in enumerate(STAGE_OF_LAYER):
            width, cap = profile[branch * 3 + stage], maxima[stage]
            factor = (22, 42, 6)[branch] / (1 if layer < 4 else 2 if layer < 7 else 4)
            total += factor * (previous * width + width * width)
            maximum += factor * (previous_max * cap + cap * cap)
            previous, previous_max = width, cap
    return total / maximum
