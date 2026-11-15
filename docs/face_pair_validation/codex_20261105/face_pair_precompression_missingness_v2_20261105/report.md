# Pre-compression anchor missingness: input-only analysis

This analysis uses 256 fixed samples from the checkpoint's training-subject calibration set. It independently hides each valid anchor point with probability 5% before temporal compression and, using a different random draw, after compression. It is a synthetic input comparison, not a measurement of detector failures or model accuracy.

| Group | Samples | Valid output anchor point-frames | Before compression: affected output bins | Become zero | Remain positive with lower score | After compression: become zero |
|---|---:|---:|---:|---:|---:|---:|
| All | 256 | 85,022 | 4,389 | 4,133 | 256 | 4,252 |
| Over 100 original frames, compressed | 34 | 13,602 | 806 | 550 | 256 | 683 |
| At most 100 original frames | 222 | 71,420 | 3,583 | 3,583 | 0 | 3,569 |

All 256 nonzero affected outputs in the compressed group changed coordinates by more than 0.001 archive coordinate units, and 255 changed by more than 0.01. Affected bins are restricted to those containing deliberately hidden original points. The score comparison uses a 0.0001 threshold, and coordinate thresholds reduce sensitivity to prefix-sum numerical noise. This analysis does not claim these coordinate changes are large in a physical or perceptual sense.

In this sample, compression can convert a missing original frame into a lower-confidence, nonzero output, which zero-score interpolation cannot identify. Different random draws are used for the before/after experiments; their individual counts should not be treated as paired observations. `samples.csv` and `summary.json` contain the sample-level counts and exact protocol. Script: `tools/analyze_face_pair_precompression_missingness.py`.
