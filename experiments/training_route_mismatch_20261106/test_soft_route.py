import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import torch
from torch.nn import functional as F

from experiments.training_route_mismatch_20261106.soft_route import SoftRouteFacePair, gamma1_log_fusion
from experiments.training_route_mismatch_20261106.train_soft_route import record_experiment_sources, save_effective_routing, training_arguments
from experiments.training_route_mismatch_20261106.check_soft_run import check_experiment_metadata
from isaa.models.face_pair_fusion import FacePairFusion
from isaa.train import parse_args


def fuse(h, b, hc, bc):
    return gamma1_log_fusion(h, b, hc, bc, torch.tensor(1.0), torch.tensor(1.0))


def test_matches_direct_normalized_probability_and_gradients():
    h = torch.tensor([[2., -1., .5, 3.], [0.2, 1.1, -2., .3]], requires_grad=True)
    b = torch.tensor([[1., .7, -2., .2], [-.6, .3, 2., -.3]], requires_grad=True)
    hc, bc = torch.tensor([5, 0, 3, 0]), torch.tensor([0, 8, 1, 0])
    result = fuse(h, b, hc, bc)
    w = torch.tensor([1., 0., .75, .5])
    p = w * h.softmax(-1) + (1 - w) * b.softmax(-1)
    p = p / p.sum(-1, keepdim=True)
    assert torch.allclose(result.exp(), p, atol=1e-7)
    grad_a = torch.autograd.grad(F.nll_loss(result, torch.tensor([0, 2])), (h, b), retain_graph=True)
    grad_b = torch.autograd.grad(F.nll_loss(p.log(), torch.tensor([0, 2])), (h, b))
    assert all(torch.allclose(x, y, atol=1e-7) for x, y in zip(grad_a, grad_b))


def test_extreme_logits_zero_and_unit_weights_finite_forward_backward():
    h = torch.tensor([[10000., -10000., 0., -10000.]], requires_grad=True)
    b = torch.tensor([[-10000., 10000., 0., -10000.]], requires_grad=True)
    result = fuse(h, b, torch.tensor([0, 1, 1, 0]), torch.tensor([1, 0, 1, 0]))
    assert torch.isfinite(result).all()
    assert torch.allclose(result.exp().sum(-1), torch.ones(1), atol=1e-6)
    F.nll_loss(result, torch.tensor([2])).backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(b.grad).all()


def test_full_objective_at_endpoint_weights():
    h = torch.tensor([[1.4, -.3, .9, -1.1], [-.7, 1.6, -.5, .8]], requires_grad=True)
    b = torch.tensor([[-.6, 1.2, .3, .2], [1.1, -.2, 1.4, -.8]], requires_grad=True)
    labels = torch.tensor([0, 1])
    hc, bc = torch.tensor([0, 3, 1, 3]), torch.tensor([3, 0, 1, 1])
    fused = fuse(h, b, hc, bc)
    loss = F.cross_entropy(fused, labels) + .5 * (F.cross_entropy(h, labels) + F.cross_entropy(b, labels))
    gradients = torch.autograd.grad(loss, (h, b), retain_graph=True)
    weights = torch.tensor([0., 1., .5, .75])
    probabilities = weights * h.softmax(-1) + (1 - weights) * b.softmax(-1)
    normalized = probabilities / probabilities.sum(-1, keepdim=True)
    reference = F.nll_loss(normalized.log(), labels) + .5 * (F.cross_entropy(h, labels) + F.cross_entropy(b, labels))
    reference_gradients = torch.autograd.grad(reference, (h, b))
    assert torch.isfinite(loss)
    assert torch.allclose(loss, reference, atol=1e-7)
    for actual, expected in zip(gradients, reference_gradients):
        assert torch.isfinite(actual).all() and actual.norm() > 0
        assert torch.allclose(actual, expected, atol=1e-7)


def test_counts_update_and_model_identity():
    model = SoftRouteFacePair(4)
    assert model.ARCHITECTURE != FacePairFusion.ARCHITECTURE
    assert set(model.state_dict()) == set(FacePairFusion(4).state_dict())
    h, b = torch.tensor([[1., 2., 3., 4.]]), torch.tensor([[4., 3., 2., 1.]])
    zero_counts = model.fuse_logits(h, b)
    model.set_routing_counts([3, 0, 1, 0], [1, 2, 1, 0], [4, 4, 4, 4])
    updated = model.fuse_logits(h, b)
    assert not torch.equal(zero_counts, updated)
    assert torch.allclose(updated, fuse(h, b, torch.tensor([3, 0, 1, 0]), torch.tensor([1, 2, 1, 0])))
    assert model.model_config["experimental_training_fusion"] == "previous_calibration_counts_gamma1_arithmetic"


def test_entry_guards():
    invalid = [[], ["--save-dir", "test", "--no-compile"],
        ["--save-dir", "test", "--init-checkpoint", "bad.pt"],
        ["--save-dir", "test", "--resume-checkpoint", "bad.pt"],
        ["--save-dir", "test", "--model-variant", "best-face-pair"]]
    for overrides in invalid:
        with unittest.TestCase().assertRaises(ValueError):
            training_arguments(overrides)


def test_entry_defaults_compile():
    args = training_arguments(["--save-dir", "test"])
    assert "--compile" in args and "--no-compile" not in args
    assert args[args.index("--model-variant") + 1] == "best-face-pair"
    with patch("sys.argv", ["train_soft_route.py", *args]):
        parsed = parse_args()
    assert parsed.compile and parsed.model_variant == "best-face-pair"
    assert not parsed.init_checkpoint and not parsed.resume_checkpoint


def test_source_archive_and_config():
    class Records:
        def update_config(self, **values):
            self.config = values
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        with ZipFile(directory / "source.zip", "w"):
            pass
        records = Records()
        record_experiment_sources(directory, records)
        import json
        records.config["architecture"] = SoftRouteFacePair.ARCHITECTURE
        records.config["args"] = {"epochs": 20, "batch_size": 32, "max_samples": 4096,
            "warmup_epochs": 1, "seed": 17, "lr": 0.025, "pair_calibration_start_epoch": 1,
            "compile": True, "compile_mode": "default"}
        for epoch in range(1, 21):
            report = {"epoch": epoch, "active": True, "classes": [
                {"class_index": 0, "action_id": "A001", "samples": 4,
                 "hand_face_correct": 3, "body_face_correct": 1}]}
            path = directory / f"routing_epoch_{epoch:03}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            save_effective_routing(path, report)
        (directory / "config.json").write_text(json.dumps(records.config), encoding="utf-8")
        check_experiment_metadata(directory)
        with ZipFile(directory / "source.zip") as archive:
            names = archive.namelist()
        assert len(names) == 3
        assert all(n in records.config["experimental_training_fusion"]["source_sha256"] for n in names)
        with unittest.TestCase().assertRaises(RuntimeError):
            record_experiment_sources(directory, records)
        records.config["experimental_training_fusion"]["source_sha256"][names[0]] = "0" * 64
        (directory / "config.json").write_text(json.dumps(records.config), encoding="utf-8")
        with unittest.TestCase().assertRaises(ValueError):
            check_experiment_metadata(directory)


def test_effective_routing_log():
    import json
    with TemporaryDirectory() as temp:
        path = Path(temp) / "routing_epoch_004.json"
        report = {"epoch": 4, "active": True, "classes": [
            {"class_index": 0, "action_id": "A001", "samples": 4, "hand_face_correct": 3, "body_face_correct": 1},
            {"class_index": 1, "action_id": "A002", "samples": 4, "hand_face_correct": 0, "body_face_correct": 0}]}
        save_effective_routing(path, report)
        result = json.loads((Path(temp) / "effective_routing_epoch_004.json").read_text(encoding="utf-8"))
        assert [r["hand_face_weight"] for r in result["classes"]] == [0.75, 0.5]
        assert result["applies_to"] == "current_epoch_validation_and_next_epoch_training"


class SoftRouteTests(unittest.TestCase):
    test_math = staticmethod(test_matches_direct_normalized_probability_and_gradients)
    test_extremes = staticmethod(test_extreme_logits_zero_and_unit_weights_finite_forward_backward)
    test_full_endpoint_objective = staticmethod(test_full_objective_at_endpoint_weights)
    test_counts = staticmethod(test_counts_update_and_model_identity)
    test_guards = staticmethod(test_entry_guards)
    test_compile = staticmethod(test_entry_defaults_compile)
    test_sources = staticmethod(test_source_archive_and_config)
    test_routing = staticmethod(test_effective_routing_log)


if __name__ == "__main__":
    unittest.main()
