"""Verify real RTMW CUDA Inductor training, both pair heads and class fusion."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Set the CUDA-wheel library path before importing torch.
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import FacePairFusion, FacePairRawXYFusion
from isaa.train import collate_rtmw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/ntu60_skeletons_rtmw.zip")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-variant", choices=("best-face-pair", "best-face-pair-raw-xy"),
                        default="best-face-pair")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not hasattr(torch, "compile"):
        raise RuntimeError("Real CUDA and torch.compile are required; no eager fallback")
    if args.batch_size < 1 or args.window_size < 1:
        raise ValueError("batch-size and window-size must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(1)
    torch.backends.cudnn.benchmark = True
    # Match the production trainer's TF32 policy. Deep convolutional networks
    # can differ by a few millilogits between Inductor and eager kernels; class
    # probabilities and actual decisions receive separate, tighter checks.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256
    if not hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        raise RuntimeError("PyTorch must support fail_on_recompile_limit_hit to reject eager fallback")
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    dataset = RTMWZipDataset(
        args.archive, split="train", split_protocol="xsub60", window_size=args.window_size,
        num_joints=133, num_classes=60, layout="rtmw_133", max_persons=2,
        feature_mode="raw", augment=False, temporal_compression="confidence_weighted",
        max_samples=args.batch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0,
                        collate_fn=partial(collate_rtmw, feature_mode="raw"))
    features, labels, frame_mask = next(iter(loader))
    x, labels, frame_mask = features.cuda(), labels.cuda(), frame_mask.cuda()
    model_class = FacePairRawXYFusion if args.model_variant == "best-face-pair-raw-xy" else FacePairFusion
    model = model_class(60).cuda().train()
    graphs = []
    # Delegate every captured graph to the genuine Inductor backend. Recording
    # Dynamo graphs makes it possible to prove the new heads/fusion are inside
    # compiled regions rather than counting eager forward calls as compilation.
    from torch._inductor.compile_fx import compile_fx

    def make_recording_inductor(captured_graphs):
        def recording_inductor(graph_module, example_inputs, **kwargs):
            captured_graphs.append(str(graph_module.graph))
            # Dynamo passes custom-backend options directly, whereas the genuine
            # Inductor compile_fx API expects them as config_patches.
            config_patches = kwargs.pop("options", None)
            if kwargs:
                raise RuntimeError(f"Unexpected Inductor backend options: {sorted(kwargs)}")
            return compile_fx(graph_module, example_inputs, config_patches=config_patches)
        return recording_inductor

    options = dict(torch._inductor.list_mode_options("default"))
    options["triton.cudagraphs"] = False
    compiled = torch.compile(model, backend=make_recording_inductor(graphs), options=options)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    started = time.perf_counter()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if name.startswith(("hand_face_head.", "body_face_head."))}
    output = compiled(x, frame_mask)
    loss = (F.cross_entropy(output["logits"], labels)
            + 0.5 * F.cross_entropy(output["hand_face_logits"], labels)
            + 0.5 * F.cross_entropy(output["body_face_logits"], labels))
    if not torch.isfinite(loss) or not all(torch.isfinite(value).all() for value in output.values()):
        raise RuntimeError("Nonfinite real-data compiled forward")
    loss.backward()
    groups = {"body": "body_blocks.", "hand": "hand_ctr_blocks.", "face": "face_st_blocks.",
              "hand_face_head": "hand_face_head.", "body_face_head": "body_face_head."}
    gradient_report = {}
    for group, prefix in groups.items():
        parameters = [(name, p) for name, p in model.named_parameters() if name.startswith(prefix) and p.requires_grad]
        if not parameters or any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in parameters):
            raise RuntimeError(f"Missing/nonfinite compiled gradient in {group}")
        norm = sum(float(p.grad.detach().float().square().sum()) for _, p in parameters) ** 0.5
        if norm <= 0:
            raise RuntimeError(f"Zero compiled gradient in {group}")
        gradient_report[group] = norm
    module_gradient_report = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        prefix = name.split(".", 1)[0]
        group = module_gradient_report.setdefault(prefix, {"parameters": 0, "gradients": 0, "squared_norm": 0.0})
        group["parameters"] += 1
        if parameter.grad is not None:
            group["gradients"] += 1
            group["squared_norm"] += float(parameter.grad.detach().double().square().sum())
    for group in module_gradient_report.values():
        group["norm"] = group.pop("squared_norm") ** 0.5
    max_grad_norm = 5.0
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm, error_if_nonfinite=True)
    postclip_gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float("inf"), error_if_nonfinite=True)
    if float(postclip_gradient_norm) > max_grad_norm + 1e-5:
        raise RuntimeError(f"Global gradient clipping failed: {float(postclip_gradient_norm)} > {max_grad_norm}")
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in before and torch.equal(parameter, before[name]):
            raise RuntimeError(f"Optimizer did not update {name}")
    torch.cuda.synchronize()
    print(f"CUDA_COMPILED_TRAIN_OK loss={float(loss.detach()):.6f} "
          f"preclip_gradients={gradient_report} postclip_norm={float(postclip_gradient_norm):.6f}", flush=True)
    model.eval()
    parity_report = {}

    def verify_parity(actual, expected, phase):
        phase_report = {}
        for key in expected:
            difference = (actual[key] - expected[key]).abs()
            actual_probability = actual[key].float().softmax(-1)
            expected_probability = expected[key].float().softmax(-1)
            probability_difference = (actual_probability - expected_probability).abs()
            same_top1 = bool((actual[key].argmax(-1) == expected[key].argmax(-1)).all())
            # A TF32 absolute logit budget avoids meaningless relative errors
            # around zero. The probability budget additionally constrains the
            # resulting classification distribution, including fused scores.
            torch.testing.assert_close(actual[key], expected[key], rtol=3e-3, atol=5e-3)
            torch.testing.assert_close(actual_probability, expected_probability, rtol=3e-3, atol=3e-4)
            if not same_top1:
                raise RuntimeError(f"Compiled/eager top1 differs for {phase}/{key}")
            phase_report[key] = {"max_abs_logit_delta": float(difference.max()),
                                 "mean_abs_logit_delta": float(difference.mean()),
                                 "max_abs_probability_delta": float(probability_difference.max()),
                                 "top1_identical": same_top1}
        parity_report[phase] = phase_report
        print(json.dumps({"parity_phase": phase, "outputs": phase_report}), flush=True)

    with torch.no_grad():
        predicted = compiled(x, frame_mask)
        reference = model(x, frame_mask)
        verify_parity(predicted, reference, "equal_weight_real_data")
        # Routing updates occur between epochs and must not bake the initial
        # equal weights into the graph. Exercise a mixed per-class selection.
        total = torch.full((60,), 10, dtype=torch.long, device="cuda")
        hf = torch.arange(60, device="cuda") % 3
        bf = (torch.arange(60, device="cuda") + 1) % 3
        model.set_routing_counts(hf, bf, total)
        routed = compiled(x[:1], frame_mask[:1])
        routed_reference = model(x[:1], frame_mask[:1])
        verify_parity(routed, routed_reference, "updated_routing_partial_real_data")
        missing = compiled(torch.full_like(x[:1], float("nan")), frame_mask[:1])
        if not all(torch.isfinite(value).all() for value in missing.values()):
            raise RuntimeError("All-missing compiled inference is nonfinite")
    torch.cuda.synchronize()
    text = "\n".join(graphs)
    evidence = {"hand_face_head": "hand_face_head" in text,
                "body_face_head": "body_face_head" in text,
                "class_weights": "hand_face_weight" in text,
                "probability_normalization": "softmax" in text and "log" in text,
                "log_softmax": "log_softmax" in text,
                "logaddexp": "logaddexp" in text,
                "logsumexp": "logsumexp" in text}
    if not all(evidence.values()):
        raise RuntimeError(f"New neural heads/fusion missing from Inductor graphs: {evidence}")

    class FusionOnly(torch.nn.Module):
        """Exercise the exact model fusion method without encoder computation."""
        def __init__(self, source):
            super().__init__()
            self.source = source

        def forward(self, hand_face_logits, body_face_logits):
            return self.source.fuse_logits(hand_face_logits, body_face_logits)

    fusion_graphs = []
    fusion_only = FusionOnly(model)
    compiled_fusion = torch.compile(
        fusion_only, backend=make_recording_inductor(fusion_graphs), options=options, fullgraph=True)
    with torch.no_grad():
        model.hand_face_weight.copy_(torch.tensor([0.0, 1.0, 0.5], device="cuda").repeat(20))
    class_ids = torch.arange(60, device="cuda")[None]
    sample_ids = torch.arange(6, device="cuda")[:, None]
    signs = torch.where((class_ids + sample_ids) % 2 == 0, 1.0, -1.0)
    # Opposing extreme predictions and all three routing choices stress the
    # same log-domain implementation used by the full compiled model.
    extreme_hf = (signs * 1e4).detach().requires_grad_(True)
    extreme_bf = (-signs * 1e4).detach().requires_grad_(True)
    extreme_labels = torch.arange(6, device="cuda")
    extreme_fused = compiled_fusion(extreme_hf, extreme_bf)
    extreme_losses = {"fused": F.cross_entropy(extreme_fused, extreme_labels),
                      "hand_face": F.cross_entropy(extreme_hf, extreme_labels),
                      "body_face": F.cross_entropy(extreme_bf, extreme_labels)}
    extreme_loss = extreme_losses["fused"] + 0.5 * (extreme_losses["hand_face"] + extreme_losses["body_face"])
    if not torch.isfinite(extreme_fused).all() or not torch.isfinite(extreme_loss):
        raise RuntimeError("Extreme compiled log-domain fusion or three-way loss is nonfinite")
    extreme_loss.backward()
    extreme_gradient_report = {}
    for head, tensor in (("hand_face", extreme_hf), ("body_face", extreme_bf)):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise RuntimeError(f"Extreme compiled fusion gradient is missing/nonfinite for {head}")
        norm = float(torch.linalg.vector_norm(tensor.grad.float()))
        if norm <= 0:
            raise RuntimeError(f"Extreme compiled fusion gradient is zero for {head}")
        extreme_gradient_report[head] = {"norm": norm, "max_abs": float(tensor.grad.abs().max()),
                                          "finite": True, "nonzero": True}
    with torch.no_grad():
        extreme_reference = fusion_only(extreme_hf, extreme_bf)
        torch.testing.assert_close(extreme_fused, extreme_reference, rtol=3e-3, atol=5e-3)
        torch.testing.assert_close(extreme_fused.softmax(-1), extreme_reference.softmax(-1),
                                   rtol=3e-3, atol=3e-4)
        probability_sums = extreme_fused.exp().sum(-1)
        torch.testing.assert_close(probability_sums, torch.ones_like(probability_sums), rtol=0, atol=1e-5)
    fusion_text = "\n".join(fusion_graphs)
    fusion_evidence = {operation: operation in fusion_text
                       for operation in ("log_softmax", "logaddexp", "logsumexp")}
    if not all(fusion_evidence.values()):
        raise RuntimeError(f"Extreme fusion log-domain operations missing from Inductor graph: {fusion_evidence}")
    torch.cuda.synchronize()
    extreme_report = {"backend": "inductor", "fullgraph": True,
        "input_shape": list(extreme_hf.shape), "input_min": -1e4, "input_max": 1e4,
        "routing_weights": [0.0, 1.0, 0.5], "classes_per_weight": 20,
        "losses": {key: float(value.detach()) for key, value in extreme_losses.items()},
        "loss_weights": {"fused": 1.0, "hand_face": 0.5, "body_face": 0.5},
        "total_loss": float(extreme_loss.detach()), "gradient_norms": extreme_gradient_report,
        "forward_backward_finite": True, "both_head_gradients_nonzero": True,
        "probabilities_normalized": True, "compiled_matches_reference": True,
        "max_abs_reference_delta": float((extreme_fused - extreme_reference).abs().max()),
        "graph_count": len(fusion_graphs), "compiled_graph_evidence": fusion_evidence}
    print(json.dumps({"EXTREME_COMPILED_FUSION_OK": extreme_report}), flush=True)
    report = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "archive": str(args.archive),
        "real_data_members": dataset.members, "input_shape": list(x.shape),
        "compile": {"enabled": True, "required": True, "backend": "inductor",
                    "mode": "default", "triton_cudagraphs": False,
                    "fail_on_recompile_limit_hit": True,
                    "suppress_errors": False,
                    "cache_size_limit": 64, "accumulated_cache_size_limit": 256,
                    "scope": "body/hand encoders, pair feature concatenation, both heads, probability fusion",
                    "inherited_eager_boundary": "face LocalSTBlock.forward"},
        "loss": float(loss.detach()), "gradient_norm": float(gradient_norm),
        "gradient_norms": gradient_report, "gradient_norms_timing": "before_global_clipping",
        "preclip_gradients_by_module_prefix": module_gradient_report,
        "gradient_clipping": {"max_norm": max_grad_norm, "preclip_norm": float(gradient_norm),
                              "postclip_norm": float(postclip_gradient_norm),
                              "postclip_limit_verified": True},
        "graph_count": len(graphs),
        "compiled_graph_evidence": evidence,
        "extreme_log_domain_fusion": extreme_report,
        "numerical_policy": {"cudnn_allow_tf32": True, "matmul_allow_tf32": True,
                             "logit_rtol": 3e-3, "logit_atol": 5e-3,
                             "probability_rtol": 3e-3, "probability_atol": 3e-4,
                             "top1_must_match": True},
        "compiled_reference_deltas": parity_report,
        "compiled_training_forward_backward": True, "both_heads_updated": True,
        "compiled_eval_matches_reference": True, "routing_update_matches_reference": True,
        "all_missing_finite": True, "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    args.output.with_suffix(".graphs.txt").write_text(text, encoding="utf-8")
    args.output.with_suffix(".fusion_graphs.txt").write_text(fusion_text, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
