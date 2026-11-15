# SkeleT-GCN source attribution

Derived from https://github.com/YijieYang23/SkeleT-GCN at
`fb2b2b23cd9665057c11ff87b3b0289a9821e964`, retrieved 2026-10-12.
Original authors: Yijie Yang, Jinlu Zhang, Jiaxu Zhang, Zhigang Tu.
Upstream includes PYSKL/OpenMMLab code. The original Apache-2.0 license is
retained in LICENSE. This directory is a modified distribution.

Files adapted from upstream:
- `gcn.py`, `tcn.py`, `init_func.py`, `init_mapping_matrices.py`: corresponding
  `pyskl/models/gcns/utils/` files; replace MMCV factories with native PyTorch.
- `graph.py`: `pyskl/utils/graph.py`; fix the unreachable expressive-keypoints
  branch (the original tests only `layout.startswith('coco-wholebody')`).
- `backbone.py`: `pyskl/models/gcns/skelet_gcn.py`; remove registry/MMCV and
  pretrained loading, avoid redundant tensor copy, move Instance Pooling.
- `instance_pooling.py`: same upstream module, replace einops with explicit
  tensor reshapes, express stochastic residual dropout as an on-device tensor
  operation to support full-graph torch.compile. No inference equation change.

`model.py` and `layers.py` are new adapters. Upstream uses `pinv(M) @ A @ M`
for adjacency initialization, rather than the paper's `M.T @ A @ M`; that
upstream choice and its rounded 0.1667 mapping entries are retained.
The public CTRGCN config selects `unit_gcn`, not `unit_ctrgcn`; both are
available under explicitly distinct names, without silently renaming them.
