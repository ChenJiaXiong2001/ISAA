"""CPU tests for the fixed temporal input screen protocol."""
import hashlib
import unittest

import torch

from tools.evaluate_face_pair_temporal_input_screen import (
    SCREEN_MODES,
    _digest_update,
    _file_sha256,
    _member_digest,
    mode_specs,
)


class TemporalInputScreenTests(unittest.TestCase):
    def test_modes_are_fixed_and_include_all_none_crops(self):
        self.assertEqual(mode_specs(), SCREEN_MODES)
        self.assertEqual(
            [(name, compression, crop) for name, compression, crop in mode_specs()],
            [("confidence_weighted", "confidence_weighted", "center"),
             ("none_start", "none", "start"),
             ("none_center", "none", "center"),
             ("none_end", "none", "end")],
        )

    def test_input_digest_depends_on_exact_tensor_and_member_order(self):
        members = ["a", "b"]
        features = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        labels = torch.tensor([1, 2])
        frames = torch.tensor([[True, True], [True, False]])
        first = hashlib.sha256()
        second = hashlib.sha256()
        _digest_update(first, members, features, labels, frames)
        _digest_update(second, list(reversed(members)), features.flip(0), labels.flip(0), frames.flip(0))
        self.assertNotEqual(first.hexdigest(), second.hexdigest())
        self.assertEqual(_member_digest(members), hashlib.sha256(b"a\nb").hexdigest())

    def test_file_digest_uses_binary_contents(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"abc" * 100)
            self.assertEqual(_file_sha256(path), hashlib.sha256(b"abc" * 100).hexdigest())


if __name__ == "__main__":
    unittest.main()
