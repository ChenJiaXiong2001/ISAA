import unittest

import torch

from experiments.derived_geometry_dropout_20261107.model import DerivedGeometryDropoutFacePairFusion


class DerivedGeometryDropoutTest(unittest.TestCase):
    def test_finite_train_and_eval(self):
        torch.manual_seed(4)
        model = DerivedGeometryDropoutFacePairFusion(60)
        x = torch.rand(2, 3, 8, 133, 1)
        x[:, 2] = 1
        mask = torch.ones(2, 8, dtype=torch.bool)
        model.train()
        train_out = model(x, mask)
        model.eval()
        eval_out = model(x, mask)
        self.assertEqual(tuple(train_out["logits"].shape), (2, 60))
        self.assertTrue(torch.isfinite(train_out["logits"]).all())
        self.assertTrue(torch.isfinite(eval_out["logits"]).all())


if __name__ == "__main__":
    unittest.main()
