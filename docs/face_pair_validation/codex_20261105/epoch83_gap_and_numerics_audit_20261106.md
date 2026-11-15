# Epoch 83 gap repair and inference audit

The frozen baseline is the original gamma=1 HF/BF checkpoint, SHA256
`aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`.
Repair fills score-zero shoulder/hip anchor gaps of at most five compressed
frames, without accessing the synthetic mask or gap length. The full
validation set has 16,306 samples from 20 subjects. The official validation
set was previously used for model and repair exploration, so these are
exploratory paired results on synthetic corruptions, not independent results.

| Original-frame condition | max5 minus unmodified (pp) | 20-subject bootstrap 95% CI (pp) | Equal-subject difference (pp) | Corrected / new errors | Subjects improved / same / degraded |
| --- | ---: | --- | ---: | ---: | ---: |
| clean | -0.0061 | [-0.0119, 0] | -0.0009 | 0 / 1 | 0 / 19 / 1 |
| independent anchor loss 5% | +5.2189 | [+4.6952, +6.1806] | +5.6366 | 999 / 148 | 20 / 0 / 0 |
| gap 2 | +4.0108 | [+3.6776, +4.5095] | +4.0862 | 796 / 142 | 20 / 0 / 0 |
| gap 5 | +2.3918 | [+1.9890, +3.0661] | +2.7040 | 530 / 140 | 20 / 0 / 0 |
| gap 10 | +0.0307 | [-0.0306, +0.0763] | -0.0009 | 13 / 8 | 4 / 13 / 3 |

The independent-loss results are in
`/home/xiong/Desktop/ISAA/outputs/precompression_epoch83_max5_validation_full_p05_cuda_20261106/`.
The gap conditions, input hashes, predictions, per-subject bootstrap, and
per-class audit are in
`/home/xiong/Desktop/ISAA/outputs/epoch83_raw_anchor_gaps_max5_validation_20261106/`.
The fixed gap protocol is documented in
`epoch83_raw_anchor_gaps_protocol_20261106.md`. Class counts for gap 2 are
44 improved / 8 unchanged / 8 degraded; for gap 5, 39 / 12 / 9; for gap 10,
6 / 50 / 4. For gap 2, A010 has 2 corrections and 11 regressions (-3.297pp).
For gap 5, A037 has 2 corrections and 9 regressions (-2.545pp). These are
descriptive post-hoc subgroup results, not multiplicity-adjusted estimates.

## Clean prediction discrepancies

The repair evaluator's unmodified eager CUDA run scored 15,537/16,306. A
later compiled CUDA geometric-fusion evaluation of the same checkpoint
scored 15,536/16,306 with the original gamma=1 fusion. Their ordered members,
clean input tensors and frame masks have matching SHA256 digests. Four sample
predictions differ: eager correctly classifies two that the compiled run
misses, misses one that it gets right, and both miss one with different
labels. The previous compiled logits cache yields the exact same predictions
as the eager run on all 16,306 samples. Thus the net one-sample difference
cannot be attributed to eager versus compiled execution alone.

For the four disagreements, the top-two fused-probability margins from the
previous compiled cache are 0.00504, 0.00332, 0.00048 and 0.00164. The new
compiled prediction was the cached runner-up in each case. The historical
cache and later compiled run both record A6000, PyTorch 2.14.0+cu126,
CUDA 12.6, batch 32, TF32 enabled, Inductor default, and cuDNN benchmark
enabled. Their data-loading and warm-up histories differ; common recorded
settings do not determine the exact convolution algorithm.

The 4-batch same-process audit used the original 32-sample batch containing
each disagreement and saved HF/BF logits and fused probabilities for eager
and compiled inference. All five audit phases produced the same top-one
classes on those four samples, matching the eager run and older cache; the
four flips observed in the later compiled evaluation were not reproduced.
Across the 128 samples, toggling benchmark and inserting a calibration-batch
warm-up produced zero maximum absolute difference within each execution
mode. With benchmark disabled, compiled versus eager maximum absolute
differences were 0.04323 for HF logits, 0.00309 for BF logits, and 0.00534
for fused probabilities. The same-process cuDNN algorithm cache can persist
between phases, so this does not isolate warm-up order or establish the cause
of the earlier prediction differences.

The class and member audit is `class_and_clean_audit.json` in the gap output
directory; cached margins are in `cached_margins.json`. Full four-batch
results and settings are under
`/home/xiong/Desktop/ISAA/outputs/epoch83_numerics_4batch_20261106/`.
Their reproducible source tools are
`tools/audit_face_pair_gap_validation.py`,
`tools/audit_face_pair_cached_margins.py`, and
`tools/audit_face_pair_inference_numerics.py`.
