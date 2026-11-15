"""CPU parity, mask isolation and trainability for the standalone variant."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from isaa.models.face_pair_fusion import FacePairFusion
from model import BodyFaceInteractionFusion
import train as experiment_train


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_model_parity(self):
        torch.manual_seed(4)
        baseline = FacePairFusion(6).eval()
        candidate = BodyFaceInteractionFusion(6).eval()
        missing, unexpected = candidate.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(set(missing), {"interaction_body.weight", "interaction_face.weight", "interaction_readout.weight"})
        self.assertEqual(unexpected, [])
        x = torch.rand(2, 3, 8, 133, 2)
        x[:, 2] = 1
        x[0, 2, 3:6, 23:80, 1] = 0
        x[1] = 0
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[1, 5:] = False
        with torch.no_grad():
            expected, actual = baseline(x, mask), candidate(x, mask)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_entry_variant_and_scratch_guards(self):
        parsed = []
        original_main = experiment_train.main_training
        original_pair = experiment_train.pair_training
        with patch.object(sys, "argv", ["train.py"]), \
             patch.object(original_main, "FACE_PAIR_VARIANT"), \
             patch.object(original_main, "FacePairFusion"), \
             patch.object(original_pair, "FACE_PAIR_VARIANT"), \
             patch.object(original_pair, "FacePairFusion"), \
             patch.object(original_pair, "COMPILE_SCOPE", original_pair.COMPILE_SCOPE), \
             patch.object(original_main, "main", lambda: parsed.append(original_main.parse_args())):
            experiment_train.main()
            args = parsed[0]
            self.assertEqual(args.model_variant, experiment_train.VARIANT)
            self.assertTrue(args.compile)
            self.assertIsNone(args.init_checkpoint)
            self.assertIsNone(args.resume_checkpoint)
            self.assertIn("masked temporal body-face", original_pair.compile_scope(args)["scope"])
            self.assertEqual(BodyFaceInteractionFusion().model_config["variant"], args.model_variant)
        for flag in ("--no-compile", "--init-checkpoint", "--resume-checkpoint"):
            with patch.object(sys, "argv", ["train.py", flag]), self.assertRaises(ValueError):
                experiment_train.main()

    def test_invalid_interactions_and_training_gradients(self):
        torch.manual_seed(6)
        model = BodyFaceInteractionFusion(5)
        body = torch.randn(2, 4, model.interaction_body.in_features)
        face = torch.randn(2, 4, model.interaction_face.in_features)
        valid = torch.tensor([[True, False, True, False], [False, True, False, False]])
        face_valid = torch.tensor([[True, True, False, False], [False, True, False, False]])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        prediction = model.interaction_logits(body, face, valid, face_valid, 2, 1)
        self.assertEqual(torch.count_nonzero(prediction).item(), 0)
        prediction.square().sum().backward()
        self.assertEqual(torch.count_nonzero(model.interaction_readout.weight.grad).item(), 0)
        optimizer.zero_grad()
        model.interaction_logits(body, face, valid, face_valid, 2, 1).sum().backward()
        self.assertGreater(model.interaction_readout.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        model.interaction_logits(body, face, valid, face_valid, 2, 1).sum().backward()
        self.assertGreater(model.interaction_body.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.interaction_face.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            before = model.interaction_logits(body, face, valid, face_valid, 2, 1)
            body2, face2 = body.clone(), face.clone()
            body2[~(valid & face_valid)] = 10000
            face2[~(valid & face_valid)] = -10000
            after = model.interaction_logits(body2, face2, valid, face_valid, 2, 1)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            empty = model.interaction_logits(body, face, torch.zeros_like(valid), face_valid, 2, 1)
            self.assertEqual(torch.count_nonzero(empty).item(), 0)


if __name__ == "__main__":
    unittest.main()
