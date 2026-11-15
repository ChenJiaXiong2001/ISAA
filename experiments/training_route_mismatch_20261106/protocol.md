# Training-route mechanism audit

Read the original epoch-83 run's archived source, epoch routing reports,
checkpoint and calibration logits. No model training, validation labels or
new inference. Archived source determines whether routing is updated before
or after training and whether fusion uses hard selection. For consecutive
active epochs leading up to epoch 83, count class selection changes including
ties, direct HF-to-BF/BF-to-HF switches separately, and compare weights with
the smooth gamma=1 ratio from the same epoch's class counts.

At the frozen epoch-83 calibration logits, compare the existing hard-routing
fusion objective to a counterfactual gamma=1 soft arithmetic fusion objective,
holding both head logits and routing counts fixed. Report calibration cross
entropy, accuracy, paired prediction changes, and gradients with respect to
head logits both for fusion loss alone and for the complete existing objective
(fusion weight 1 and each auxiliary head loss weight 0.5). Read actual loss
weights from the checkpoint. These are output-layer probes, not parameter
gradients or evidence that retraining would improve. Epoch-83 saved routing is
used as the frozen probe; epoch 83 training itself used epoch-82 routing.

Candidate worth future screening: a new scratch-only entry that uses previous
epoch's gamma=1 counts for differentiable soft fusion during training, while
preserving both head losses, input features, old model and evaluation. Keep
mandatory Inductor and existing face eager boundary, reject eager fallback,
and verify compiled CUDA forward/backward/update. Compare fixed epoch-20 last
checkpoint with same-budget hard-route control and identical train/calibration
members, then same-input clean and corruption evaluations. This is a new
training intervention, distinct from changing fusion-loss magnitude or adding
HF/BF consistency. Do not assume it helps or claim a causal benefit from this
audit; the calibration data were used to set the route and consist of only two
subjects. A runtime/initialization and training variation caveat remains.

## Short-screen decision and mathematical checks

Use a matched original `best-face-pair` baseline (not the hand-motion model)
and this candidate, same 4096 training-prefix samples, 20 epochs, batch 32,
warmup one epoch, seed 17, learning rate 0.025, and training calibration
start epoch one. Both run from scratch; the only intended model difference is
the fusion used for the loss and readout. Fix `last.pt` at epoch 20 before
examining results. If an existing baseline is used, first verify its complete
configuration, source, member manifest, training architecture and checkpoint
digests; otherwise train a matched baseline. No early checkpoint selection.

Compare the training-calibration and same-input complete validation clean
accuracy plus synthetic original-frame 5%/10% anchor and body-point 5%/10%
and hand/face group 5%/10% conditions. Use each checkpoint's gamma=1 counts,
and repeat fixed-common-weight sensitivity from the same cached head logits.
Report corrected/new errors, per-subject results, sample-weighted 20-subject
bootstrap intervals and equal-subject averages, and check input/mask hashes.
Require a predeclared clean point estimate at least +0.25pp versus baseline,
plus no individual corruption point estimate below -0.25pp before considering
further research; even passing is not confirmation of generalization, and CI
must be reported separately. Predeclare before the CUDA run, keep all failures.

For class c with head probabilities pH[c], pB[c], counts h[c], b[c], let
`w[c] = h[c]/(h[c]+b[c])` when the sum is positive and 0.5 otherwise.
`p[c] = (w[c]*pH[c]+(1-w[c])*pB[c]) / sum_j(...)`. The sum over classes
is necessary because class-dependent weights do not preserve normalization.
The candidate computes its logarithm with log-softmax/logaddexp/logsumexp;
hard endpoint weights are represented by log(0) = -infinity in an additive
term, not probability clamps, while at least one head remains finite per
class. Unit tests check forward and backward against the direct probability
formula at moderate logits and finite outputs/gradients for +/-10000 logits
with 0, 1 and 0.5 weights. An explicit CUDA compiled forward/backward and
optimizer update must precede training; verify compiled outputs change after
the same model's calibration buffers change, and reject any eager fallback.

Because the isolated architecture is deliberately distinct, old checkpoint
checkers and evaluation scripts must not treat it as the old architecture.
Prepare an explicit new checker and shared-input paired evaluator before any
GPU training, then verify `config.json`, `source.zip`, compile metadata,
saved counts, split manifests and predicted outputs. CUDA tests and training
have not yet been performed.
