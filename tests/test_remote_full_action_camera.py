import unittest
from argparse import Namespace

import numpy as np

from tools.remote_full_action_camera import REMOTE_PROGRAM, RemoteActionClient, _format_action_lines


class RemoteFullActionCameraTests(unittest.TestCase):
    def test_formatter_shows_only_top1_and_keeps_validation_accuracy(self):
        action = {
            "best": [(0, 0.74), (1, 0.16), (2, 0.10)],
            "best_info": {"validation_top1": 0.9563350913774072},
            "weighted": [(0, 0.72), (1, 0.18), (2, 0.10)],
            "weighted_info": {"validation_top1": 0.952778},
            "HF": [(1, 0.61), (0, 0.23)],
            "HF_info": {"validation_top1": 0.884889},
            "BF": [(2, 0.58), (0, 0.27)],
            "BF_info": {"validation_top1": 0.927757},
            "ctrgcn25": [(3, 0.49), (0, 0.31)],
            "ctrgcn25_info": {"validation_top1": None},
        }

        lines, report = _format_action_lines(action)

        self.assertEqual(len(lines), 5)
        self.assertTrue(all("Top1 " in line for line in lines))
        self.assertTrue(all("Top2 " not in line for line in lines))
        self.assertIn("验证准确率 95.63%", lines[0])
        self.assertIn("验证准确率 88.49%", lines[3])
        self.assertIn("验证准确率 92.78%", lines[4])
        self.assertNotIn("验证准确率", lines[1])
        self.assertEqual(report, "; ".join(lines))

    def test_remote_program_loads_face_pair_and_ctr_gcn_only(self):
        compile(REMOTE_PROGRAM, "remote_full_action_camera.py", "exec")
        self.assertIn("--face-checkpoint", REMOTE_PROGRAM)
        self.assertIn("--face-summary", REMOTE_PROGRAM)
        self.assertIn("--ctrgcn-checkpoint", REMOTE_PROGRAM)
        self.assertNotIn("--pool-checkpoint", REMOTE_PROGRAM)
        self.assertLess(REMOTE_PROGRAM.index("pair_model, pair_weight, pair_info"),
                        REMOTE_PROGRAM.index("from tools.realtime_pool_camera"))

    def test_client_queues_microbatch_and_counts_overflow(self):
        args = Namespace(batch_size=4, max_pending_frames=4, batch_wait_ms=1,
                         jpeg_quality=70)
        client = RemoteActionClient(args)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.assertTrue(client.submit(frame, 0))
        self.assertTrue(client.submit(frame, 1))
        self.assertTrue(client.submit(frame, 2))
        self.assertTrue(client.submit(frame, 3))
        self.assertFalse(client.submit(frame, 4))
        stats = client.stats()
        self.assertEqual((stats['submitted'], stats['queued'], stats['dropped']), (4, 4, 1))
        self.assertEqual([item[1] for item in client.pending], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
