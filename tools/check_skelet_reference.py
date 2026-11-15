"""Compare port outputs/gradients with a pinned official source checkout.

Optional verification dependency: einops (not required to train this port).
The reference is executed in isolated module namespaces without MMCV/PYSKL.
Only imports/registries and the unreachable expressive-layout branch are
adapted; graph, temporal and mapping calculations come from the checkout.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from isaa.models.skelet import SkeleTGCN
from isaa.models.skelet.model import VARIANTS
from isaa.data.skelet_dataset import preprocess_keypoints, uniform_indices


def module_from_source(name, source):
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def load_reference(root):
    package = types.ModuleType("skelet_reference")
    package.__path__ = []
    sys.modules[package.__name__] = package
    for filename in ("init_func", "init_mapping_matrices", "gcn", "tcn"):
        source = (root / f"pyskl/models/gcns/utils/{filename}.py").read_text(encoding="utf-8")
        source = source.replace("from mmcv.cnn import build_activation_layer, build_norm_layer",
                                "from isaa.models.skelet.layers import build_activation_layer, build_norm_layer")
        source = source.replace("from mmcv.cnn import build_norm_layer", "from isaa.models.skelet.layers import build_norm_layer")
        source = source.replace("from mmcv.cnn import build_activation_layer", "from isaa.models.skelet.layers import build_activation_layer")
        source = source.replace("from mmcv.utils import _BatchNorm", "from torch.nn.modules.batchnorm import _BatchNorm")
        module_from_source(f"skelet_reference.{filename}", source)
    source = (root / "pyskl/utils/graph.py").read_text(encoding="utf-8")
    source = source.replace("elif layout.startswith('coco-wholebody'):",
                            "elif layout in ('coco-wholebody', 'expressive-keypoints'):")
    module_from_source("skelet_reference.graph", source)
    source = (root / "pyskl/models/gcns/skelet_gcn.py").read_text(encoding="utf-8")
    for line in ("from mmcv.cnn import normal_init\n", "from mmcv.runner import load_checkpoint\n",
                 "from ..builder import BACKBONES\n", "@BACKBONES.register_module()\n"):
        source = source.replace(line, "")
    source = source.replace("from ...utils.graph import Graph", "from .graph import Graph")
    source = source.replace("from .utils import mstcn, residual_block, grouped_mapping_framework, MSTCN, dgmstcn",
                            "from .gcn import residual_block, grouped_mapping_framework\nfrom .tcn import mstcn, MSTCN, dgmstcn")
    source = source.replace("from .utils import mapping_matrices", "from .init_mapping_matrices import mapping_matrices")
    return module_from_source("skelet_reference.backbone", source)


def compare(root):
    ref = load_reference(root)
    results = []
    for variant in VARIANTS:
        for ip in (False, True):
            torch.manual_seed(1)
            port = SkeleTGCN(7, variant, instance_pooling=ip)
            config = dict(port.config)
            if ip:
                config["ip_drop_path"] = 0
                port.backbone.instance_pooling_block.drop_path = 0
            original = ref.SkeleT_GCN(**config)
            original.load_state_dict(port.backbone.state_dict(), strict=True)
            for training in (False, True):
                port.train(training)
                original.train(training)
                a = torch.randn(2, 2, 16, 65, 3, requires_grad=True)
                b = a.detach().clone().requires_grad_(True)
                # Compare backbone tensors before classifier/readout.
                ya, yb = port.backbone(a), original(b)
                torch.testing.assert_close(ya, yb, rtol=1e-5, atol=1e-6)
                port.zero_grad(set_to_none=True)
                original.zero_grad(set_to_none=True)
                ya.square().mean().backward()
                yb.square().mean().backward()
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
                left = dict(port.backbone.named_parameters())
                for name, param in original.named_parameters():
                    if param.grad is not None:
                        torch.testing.assert_close(left[name].grad, param.grad, rtol=1e-4, atol=1e-6)
                results.append(dict(backbone=variant, instance_pooling=ip, training=training,
                                    max_output_difference=(ya-yb).abs().max().item(),
                                    max_input_gradient_difference=(a.grad-b.grad).abs().max().item()))
    return results


def compare_pipeline(root):
    source = (root / "pyskl/datasets/pipelines/pose_related.py").read_text(encoding="utf-8")
    source = source.replace("elif self.dataset.startswith('coco-wholebody'):",
                            "elif self.dataset in ('coco-wholebody', 'expressive-keypoints'):")
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
             and n.name in ("PreNormalize2D", "JointToBone", "ToMotion")]
    sampling = ast.parse((root / "pyskl/datasets/pipelines/sampling.py").read_text(encoding="utf-8"))
    nodes += [n for n in sampling.body if isinstance(n, ast.ClassDef) and n.name == "UniformSampleFrames"]
    for node in nodes:
        node.decorator_list = []
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "reference_pipeline", "exec"), namespace)
    rng = np.random.RandomState(12)
    raw = rng.randn(3, 27, 65, 2).astype(np.float32)
    raw[:2] = raw[:2]*100 + 600
    raw[2] = rng.rand(27, 65, 2)
    for stream in ("j", "b", "jm", "bm"):
        results = dict(keypoint=raw.transpose(3, 1, 2, 0).copy(), img_shape=(1080, 1920))
        results = namespace["PreNormalize2D"]()(results)
        if stream in ("b", "bm"):
            results = namespace["JointToBone"](dataset="expressive-keypoints")(results)
        if stream in ("jm", "bm"):
            results = namespace["ToMotion"](dataset="expressive-keypoints", target="keypoint")(results)
        np.testing.assert_allclose(preprocess_keypoints(raw, stream), results["keypoint"], rtol=0, atol=0)
    for training in (True, False):
        for frames in (3, 16, 25, 1000):
            np.random.seed(100)
            port = uniform_indices(frames, 16, training=training, num_clips=10)
            np.random.seed(100)
            original = namespace["UniformSampleFrames"](clip_len=16, num_clips=10)(
                dict(total_frames=frames, start_index=0, test_mode=not training))["frame_inds"]
            np.testing.assert_array_equal(port.reshape(-1), original)
    return dict(streams=4, sampling_cases=8, exact_match=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = compare(args.reference_root)
    report = dict(reference_root=str(args.reference_root), checks=results, passed=len(results),
                  pipeline=compare_pipeline(args.reference_root))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
