# Training route and objective mismatch audit

The epoch-83 checkpoint uses per-class hard routing in its actual training
fusion loss, but the target gamma=1 evaluation uses normalized classwise
accuracy-ratio probability fusion. The archived `source.zip` shows each epoch
trains first, then updates the routing buffers from training-subject holdout
counts. Consequently epoch 83 used epoch-82 routing, not epoch-83 routing.
Both heads still receive individual cross-entropy losses with weight 0.5, so
the hard route does not eliminate all learning in the unselected branch.

In the 78 adjacent active route updates from epochs 5 through 83, on average
22.08 of 60 class choices changed per update; 15.32 were direct HF-to-BF or
BF-to-HF flips. These changes reflect both model evolution and noisy finite
calibration counts; the audit does not isolate those causes. At epoch 83,
17 classes selected HF, 35 selected BF and 8 tied. The mean absolute distance
between the hard selection and gamma=1 continuous weights was 0.4088.

The frozen checkpoint's saved epoch-83 routing was applied to its 3,191
calibration logits to compare objectives. This is an output-layer probe, not
the exact route seen during epoch-83 training. Hard vs gamma=1 arithmetic
correct counts were 2990 and 3014 (46 corrections, 22 new errors); NLL was
0.24521 and 0.24671, respectively. In 79 samples the hard-selected true-class
head was wrong while the other head was right. Both training head-loss terms
remain active regardless. For the complete objective, output-logit gradients
had cosine 0.8761 and relative L2 difference 0.4834; this is not a model
parameter gradient and does not predict the outcome of training. Calibration
subjects already contributed the checkpoint's route and are not independent.

The isolated `SoftRouteFacePair` preserves the original encoders, heads,
parameter/buffer shapes and auxiliary losses, but uses the previous
calibration counts to compute gamma=1 weights inside the forward pass. Both
zero counts imply 0.5. The readout calculates
`s[c] = w[c]*softmax(HF)[c] + (1-w[c])*softmax(BF)[c]`, then normalizes
`s` across all classes. Log-softmax, logaddexp and shifted logsumexp implement
this expression without clamping probabilities. It is distinct from simply
raising the fusion-loss coefficient or imposing KL agreement, both of which
were previously screened. Full derivation, fixed screen and limits are in
`protocol.md`.

The independent entry `train_soft_route.py` preserves the old training entry
and model class, blocks checkpoint initialization and disabling compile via
the inherited guards, and writes the experimental source files and hashes to
`source.zip`/config. The inherited `routing_epoch_*.json` and hard selection
buffer remain diagnostic only; `effective_routing_epoch_*.json` records the
actual gamma=1 weights used for that epoch's validation and the following
epoch's training. Seven CPU tests passed with Python 3.10: direct probability
and gradient equivalence, logits +/-10000 with zero/unit weights and finite
backward, lagged-count buffer updates, architecture/state shape, entry/parser
guards, and source archive/config integrity. The real-data A6000 CUDA Inductor
smoke test completed training forward/backward and an optimizer step, with
nonzero finite gradients in the body, hand, face and both classifier heads.
Updating counts changed the compiled output and matched the uncompiled
reference. Both heads and the count-weighted log-domain fusion appeared in
captured Inductor graphs; the original face LocalSTBlock.forward eager boundary
remains. Details are in `cuda_check_20261106.json`. This was not a training
experiment and the candidate has no measured training or validation accuracy.
A dedicated checker validates its distinct
architecture, source archive, fixed training budget and all 20 effective
routing records. A dedicated paired evaluator is being prepared before any
comparison.
