# Bounded anchor-gap interpolation candidates

This is a frozen-checkpoint stress test on 1,420 training-subject calibration samples. The same samples contributed the checkpoint's calibration weights, so the result is development evidence rather than an independent accuracy estimate. The mask probabilities were fixed before observing results and produce approximately 5% hidden valid anchor point-frames. The model and gamma=1 fusion weights are unchanged.

The three candidates use linear interpolation of anchor XY and score between valid endpoints. They require a future endpoint and do not extrapolate across invalid frame-mask regions:

| Candidate | Filled gap | Look-ahead requirement |
|---|---:|---:|
| Single-frame | exactly 1 missing frame | up to 1 frame |
| Max-2 | 1 or 2 missing frames | up to 2 frames |
| Max-5 | 1 through 5 missing frames | up to 5 frames |

The table below reports gamma=1 fusion (soft) Top-1. “Gain/loss” is paired sample-level change against the unmodified corrupted input; the bootstrap interval is a sample bootstrap on this calibration set.

| Condition | Unmodified | Single-frame | Max-2 | Max-5 | Single gain/loss | Max-2 gain/loss | Max-5 gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 94.366% | 94.366% | 94.366% | 94.366% | 0/0 | 0/0 | 0/0 |
| Independent 5% | 89.718% | 94.014% | 93.944% | 94.014% | +77/-16 | +78/-18 | +79/-18 |
| Contiguous gap 2 | 90.493% | 90.493% | 93.803% | 94.225% | 0/0 | +64/-17 | +66/-13 |
| Contiguous gap 5 | 91.901% | 91.901% | 91.901% | 93.944% | 0/0 | 0/0 | +46/-17 |
| Contiguous gap 10 | 92.887% | 92.887% | 92.887% | 92.887% | 0/0 | 0/0 | 0/0 |

For independent 5% missingness, single-frame repair improved 4.296 percentage points (paired bootstrap 95% interval [3.028, 5.634]); extending the bound did not improve the aggregate result. For a two-frame contiguous gap, Max-5 was best at 94.225% (+3.732 points versus unmodified; [2.535, 5.000]); Max-2 reached 93.803% (+3.310 points; [2.042, 4.577]). For a five-frame gap, Max-5 reached 93.944% (+2.042 points; [0.986, 3.169]). No candidate repairs a ten-frame gap.

The observed look-ahead is a protocol delay, not a measured GPU runtime. No synchronized timing run was started after the comparison because the requested short training experiments had priority. The actual inference-time overhead and wall-clock latency therefore remain unmeasured. The fixed protocol requires at most 1, 2, or 5 future compressed frames respectively; at 30 FPS these correspond to approximately 33, 67, or 167 ms of look-ahead, before any model compute. This conversion is a frame-rate example, not a measured project runtime.

The result files are `summary.json`, `predictions.csv`, and `paired.json`; the evaluator is `tools/evaluate_face_pair_anchor_gaps.py`, and the paired analysis is `tools/summarize_face_pair_anchor_gap_candidates.py`.
