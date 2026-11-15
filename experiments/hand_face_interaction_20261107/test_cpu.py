import importlib.util
import pathlib
import sys
import unittest

import torch

from experiments.hand_face_interaction_20261107.model import HandFaceTemporalInteractionFusion


class InteractionModelTest(unittest.TestCase):
    def test_zero_initialized_residual_preserves_base_logits(self):
        model = HandFaceTemporalInteractionFusion(60)
        self.assertEqual(float(model.interaction_readout.weight.abs().sum()), 0.0)
        hand = torch.randn(2, 4, 48)
        face = torch.randn(2, 4, 48)
        valid = torch.ones(2, 4, dtype=torch.bool)
        self.assertTrue(torch.equal(model.interaction_logits(hand, face, valid, valid, 1, 2),
                                    torch.zeros(1, 60)))

    def test_masked_interaction_is_finite_and_has_gradients(self):
        model = HandFaceTemporalInteractionFusion(60)
        with torch.no_grad():
            model.interaction_readout.weight.normal_(0, 0.01)
        hand = torch.randn(2, 4, 48, requires_grad=True)
        face = torch.randn(2, 4, 48, requires_grad=True)
        hand_valid = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        face_valid = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
        residual = model.interaction_logits(hand, face, hand_valid, face_valid, 1, 2)
        self.assertTrue(torch.isfinite(residual).all())
        residual.sum().backward()
        self.assertGreater(float(model.interaction_hand.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.interaction_face.weight.grad.abs().sum()), 0.0)

    def test_training_entry_rejects_no_compile_and_checkpoints(self):
        path = pathlib.Path(__file__).with_name("train.py")
        spec = importlib.util.spec_from_file_location("hf_interaction_train", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.training_arguments(["--no-compile"])
        with self.assertRaises(ValueError):
            module.training_arguments(["--init-checkpoint", "old.pt"])
        args = module.training_arguments(["--dry-run", "--device", "cpu", "--num-workers", "0"])
        self.assertIn("--model-variant", args)
        self.assertEqual(args[args.index("--model-variant") + 1], "best-face-pair")


if __name__ == "__main__":
    unittest.main()

