# Original pooled model with two-hand temporal relation fusion

Entry: `train_best_hand_relation_100_cosine.py`. The experiment keeps the original
22-node body CTR-GCN, shared 21-node-per-hand CTR-GCN, six face tokens compressed
from the original 68 face points, six-channel hand distance/direction features,
all original node/person/time means, and the 192+96 classifier interface.
The new relation residual is inserted before the original local projection.
It is a separate variant and starts from random initialization; the canonical
94.6400% baseline checkpoint is not loaded (its training-log value was 94.6461%).

For each of 21 corresponding left/right joints, the relation branch combines
the encoded feature mean, absolute difference and product with torso-normalized
distance, direction, consecutive-frame distance change and pair confidence.
The pooled body feature queries masked joint-pair attention. Parallel depthwise
temporal convolutions (kernels 3 and 5, the latter dilated by 2) produce a
48-channel relation feature. A body/hand/quality gate and a bounded residual
scale, initialized to 0.1, add it to the original `(hand + face) / 2` feature.
No valid hand pair means an exactly zero relation residual; there are no
cross-person pairings or action-label routing decisions.

The default protocol is NTU60 XSub, confidence-weighted full-sequence compression
to 100 frames, batch 32, SGD/Nesterov with momentum 0.9 and weight decay 0.0004,
seed 1, and 100 epochs. Each training batch updates a five-epoch linear warmup
from 0.001 to 0.1 followed by cosine decay to 0.0001. The dedicated and generic
entry points require `torch.compile` and reject `--no-compile`; compilation
failure is fatal. The actual compiled neural scope includes the body/hand
encoders, original pooling, pair attention, temporal relation fusion, gate and
classifier. The inherited face `LocalSTBlock` eager boundary remains.

```bash
cd /home/xiong/Desktop/ISAA
~/venvs/isaa-cu126/bin/python train_best_hand_relation_100_cosine.py --device cuda
```

The run is saved under `outputs/best_hand_relation_100_cosine_ntu60_xsub/`.
After training, evaluate its best checkpoint twice with `tools/evaluate_best.py`,
once normally and once with `--interaction-mode none`. The second result is the
original path *within the jointly trained model*, not the independently trained
94.6461% baseline. Compare both with the independent baseline on the same
validation samples and inspect per-class errors, especially A010/A034.
