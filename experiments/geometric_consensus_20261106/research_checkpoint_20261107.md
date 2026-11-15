# ISAA research checkpoint 2026-11-07

## Formal baseline

Formal epoch-83 best-face-pair checkpoint SHA256: `aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`. Validation manifest SHA256: `82fd60410fd04e381c65d3bb9f24f18f7993124ee2581902942fbfb4a2ecddc0` (16,306 samples). Clean accuracy is 95.28394456% under the accelerated CUDA evaluator.

The score-power compression screen gave 95.28394456% for p=0, 0.5 and 1, and 95.29007727% for p=1.5 and 2. The maximum change is one sample (+0.00613pp), so this direction is rejected.

Formal corruption baseline: point 5% 85.90089538%, point 10% 74.92334110%, group 5% 93.98994235%, group 10% 92.37703913%.

## Current leading candidate: frozen geometric fusion

The compiled network is unchanged. A weighted-geometric probability pool is mixed with the existing arithmetic pool after the forward pass. Strength 0.5 is currently preferred:

- clean +0.35570pp, subject-bootstrap 95% CI [+0.234,+0.530]
- same-formal corruption rerun: point 5% +0.675pp CI [+0.283,+1.136], point 10% +0.092pp CI crossing zero, group 5% +0.497pp CI [+0.275,+0.894], group 10% +0.442pp CI [+0.282,+0.716]
- a different mask seed gave point 5% +0.932pp, group 5/10% about +0.540pp
- raw-gap + max5 repair + geometric fusion gave clean +0.35570pp and gap2/5/10 +0.288/+0.270/+0.399pp in a second mask run
- batch-32 network latency increased by about 0.042ms; fusion adds about 0.11ms

Strength 0.25 is a viable lower-variance alternative but has smaller clean gain. Strengths 0.75 and 1.0 are rejected because point-10 corruption becomes negative in the strength corruption grid. Classwise strength selection was tested with two calibration-subject LOSO and produced held-out changes of 0.00pp and +0.04pp; it adds complexity without evidence of gain.

These are frozen inference and synthetic-corruption results on a repeatedly used validation set. They are not independent camera-domain confirmation.

## Rejected scratch candidate

An 8-epoch random-scratch hand-face temporal interaction residual passed real CUDA/Inductor compilation and training, but paired clean accuracy fell about 0.47pp while point/group corruption improved strongly. It is retained as robustness-only evidence and is not promoted to long training.

## Rejected mask-aware derived geometry variants

An isolated compiled CUDA screen added two scratch-only variants without touching the production model or entry points. `geometry_only` masked cross-hand derived distance/direction whenever torso anchors were incomplete; it was neutral to slightly negative (clean -0.006pp, point 5/10% -0.006pp, groups unchanged). `center_scale` replaced partial torso references with center zero and scale one; it reduced clean by -0.215pp and collapsed point-5/10 accuracy by about 76/69pp. Both variants are rejected for training. The isolated implementation, report, and CPU tests are in `experiments/derived_geometry_mask_20261107/`.

## Decision

Prioritize the frozen strength-0.5 geometric fusion as the practical candidate, optionally with max5 anchor repair for detector-like raw-anchor gaps. Do not spend a full training run on score-power compression, temporal interaction, or the mask-aware derived-geometry variants.

## Precompression composition check

Using the formal checkpoint, a 5% precompression anchor-dropout screen compared
the existing arithmetic pool with strength-0.5 geometric fusion before and
after max5 interpolation. Geometric fusion improved clean unmodified inputs by
+0.343pp (CI [+0.251,+0.534]) and corrupted unmodified inputs by +1.098pp (CI
[+0.804,+1.265]). After max5 repair, clean and repaired-corrupted inputs both
showed +0.435pp (CI [+0.330,+0.593]). This supports composing max5 repair for
detector-like missing anchors with geometric fusion for the final readout.

A 10% precompression rerun gave +0.448pp before repair (CI crossing zero) and +0.429pp after max5 repair (CI [+0.295,+0.631]). The repaired composition retained a positive geometric gain under both 5% and 10% synthetic precompression dropout.

## Integration audit

The existing isolated CUDA evaluator already verifies checkpoint and manifest hashes, compiled Inductor forward, normalized arithmetic/geometric log-probabilities, frozen model state, full validation traversal, and subject bootstrap. The synchronized benchmark measured only about +0.042ms network latency at batch 32 and about +0.11ms fusion-only work. No production entry point needs modification for this post-forward candidate.

A reusable isolated helper is now in `experiments/geometric_consensus_20261106/inference.py`. It accepts the compiled model's HF/BF logits and frozen class weights, returns normalized log probabilities, and keeps strength configurable while defaulting to 0.5. Four CPU numerical tests cover normalization and endpoint behavior; production model and entry points remain untouched.
