"""Small checks for the strict, mask-free CTR-GCN baseline."""

import unittest

import torch

from isaa.models.original_ctrgcn import (
    OriginalCTRGCN,
    build_official_rtmw_adjacency,
    build_rtmw_adjacency_for_nodes,
)
from isaa.layouts.rtmw_133 import RTMW_25_NODE_INDICES, RTMW_32_NODE_INDICES


class OriginalCTRGCNTests(unittest.TestCase):
    def test_official_rtmw_graph_shape_and_column_normalization(self):
        adjacency = build_official_rtmw_adjacency()
        self.assertEqual(tuple(adjacency.shape), (3, 133, 133))
        torch.testing.assert_close(adjacency[0], torch.eye(133))
        for partition in adjacency[1:]:
            column_sums = partition.sum(dim=0)
            nonempty = column_sums > 0
            torch.testing.assert_close(
                column_sums[nonempty], torch.ones_like(column_sums[nonempty])
            )

    def test_forward_accepts_bctvm_and_uses_official_pooling_shape(self):
        graph = torch.eye(5).repeat(3, 1, 1)
        model = OriginalCTRGCN(
            num_classes=3,
            num_point=5,
            num_person=2,
            graph=graph,
            channels=(8,) * 10,
        ).eval()
        x = torch.randn(2, 3, 4, 5, 2)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_progressive_rtmw_induced_graphs_and_duplicate_25_mapping(self):
        for count, expected in ((25, RTMW_25_NODE_INDICES), (32, RTMW_32_NODE_INDICES), (133, tuple(range(133)))):
            adjacency, indices = build_rtmw_adjacency_for_nodes(count)
            self.assertEqual(indices, expected)
            self.assertEqual(tuple(adjacency.shape), (3, count, count))
            self.assertEqual(len(indices), count)
            if count == 25:
                self.assertGreaterEqual(len(set(indices)), 19)
                self.assertNotEqual(len(set(indices)), count)
            for partition in adjacency[1:]:
                column_sums = partition.sum(dim=0)
                nonempty = column_sums > 0
                torch.testing.assert_close(
                    column_sums[nonempty], torch.ones_like(column_sums[nonempty])
                )


if __name__ == "__main__":
    unittest.main()
