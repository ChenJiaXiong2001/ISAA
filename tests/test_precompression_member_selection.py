"""Validate manifest subset selection for precompression experiments."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_face_pair_precompression_repair import select_members


class MemberSelectionTests(unittest.TestCase):
    def test_default_and_prefix_preserve_manifest_order(self):
        self.assertEqual(select_members(["a", "b", "c"], None, 0)[0], ["a", "b", "c"])
        self.assertEqual(select_members(["a", "b", "c"], None, 2)[0], ["a", "b"])

    def test_subset_preserves_file_order_and_records_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "members.csv"
            payload = b"sample,frames\nc,120\na,101\n"
            path.write_bytes(payload)
            members, source = select_members(["a", "b", "c"], path, 0)
            self.assertEqual(members, ["c", "a"])
            self.assertEqual(source["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(source["order"], "members file row order")

    def test_reject_invalid_files_and_ambiguous_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "members.csv"
            for payload in (b"sample\na\na\n", b"sample\na\noutside\n", b"other\na\n", b"sample\n", b"sample,sample\na,a\n"):
                path.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    select_members(["a", "b"], path, 0)
            path.write_bytes(b"sample\na\n")
            with self.assertRaises(ValueError):
                select_members(["a", "b"], path, 1)
            with self.assertRaises(ValueError):
                select_members(["a", "a"], path, 0)


if __name__ == "__main__":
    unittest.main()
