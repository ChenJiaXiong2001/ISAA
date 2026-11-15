"""CPU scalar-reference tests for the fixed three-frame smoother."""
import importlib.util
import sys
from pathlib import Path
import unittest

import torch

path = Path(__file__).with_name('screen.py')
spec = importlib.util.spec_from_file_location('screen', path)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def reference(x, frames):
    out = x.clone()
    for t in range(1, x.shape[1] - 1):
        for joint in range(x.shape[2]):
            for person in range(x.shape[3]):
                if not all(bool(frames[k]) and float(x[2, k, joint, person]) > 0 for k in (t-1, t, t+1)):
                    continue
                w = [float(x[2, t-1, joint, person]), 2*float(x[2, t, joint, person]), float(x[2, t+1, joint, person])]
                for axis in (0, 1):
                    out[axis, t, joint, person] = sum(w[i] * float(x[axis, t-1+i, joint, person]) for i in range(3)) / sum(w)
    return out


class TestSmooth(unittest.TestCase):
    def test_scalar_reference_mask_edges_and_score(self):
        torch.manual_seed(123)
        x = torch.rand(3, 7, 8, 2)
        x[2, 3, 1, 0] = 0
        frames = torch.tensor([True, True, True, True, False, True, True])
        before = x.clone()
        out = screen.smooth(x, frames)
        torch.testing.assert_close(out, reference(x, frames), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(out[2], x[2]))
        self.assertTrue(torch.equal(out[:, 0], x[:, 0]))
        self.assertTrue(torch.equal(out[:, -1], x[:, -1]))
        self.assertTrue(torch.equal(x, before))


if __name__ == '__main__':
    unittest.main()
