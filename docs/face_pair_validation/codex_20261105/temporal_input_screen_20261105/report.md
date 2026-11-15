# Frozen temporal input construction screen

This screen evaluates the seed2 `best-face-pair` checkpoint on the independent validation manifest (16,306 samples, with a fixed 4,096-member prefix reported separately). The model weights, gamma=1 calibration counters, member order, raw x/y/score representation, and frame-mask handling are fixed. No training, validation recalibration, augmentation, or synthetic corruption is used.

| Mode | Full soft Top-1 | Full HF | Full BF | Prefix 4,096 soft | Prefix HF | Prefix BF |
|---|---:|---:|---:|---:|---:|---:|
| `confidence_weighted` | 95.5109% | 88.5196% | 92.9290% | 94.4824% | 87.8418% | 92.4805% |
| `none_start` | 94.0942% | 86.6675% | 91.0585% | 93.3594% | 86.5723% | 90.8936% |
| `none_center` | 94.3150% | 87.2072% | 91.2057% | 93.4082% | 86.9629% | 91.1377% |
| `none_end` | 93.9715% | 87.2010% | 90.4698% | 93.5303% | 87.3291% | 90.6006% |

Relative to the existing `confidence_weighted` input, full soft accuracy falls by 1.4167 percentage points for `none_start`, 1.1959 points for `none_center`, and 1.5393 points for `none_end`. The same ordering appears on the fixed 4,096 prefix, where the drops are 1.1230, 1.0742, and 0.9521 points respectively. The center crop is the best of the no-compression alternatives, but it remains below confidence-weighted compression on the full validation set and on the prefix.

The exact input digests are in `summary.json`. Full digests are:

| Mode | Input SHA-256 |
|---|---|
| `confidence_weighted` | `0270bad405c732de49238ac6032eed7802cadf3c0a7657f521fe9d03cb961ca7` |
| `none_start` | `5276b826bb63435665c13b69ce83e8bed53bee6abd6d1edff6591918e077fecf` |
| `none_center` | `6d321335d35fd4ef02410a248ee529f86a0062f05b3477616a296f7c1a714852` |
| `none_end` | `d8a0a7860854b951628f6473ad9c40940fe76273247b2fd3317b29fb3871d359` |

The result supports retaining the current confidence-weighted compression as the input-construction candidate. The no-compression variants are useful negative controls: deterministic crop placement alone does not recover the checkpoint's accuracy. This does not isolate whether the gain comes from confidence weighting, complete-sequence coverage, or both; that would require an additional frozen input construction with a different compression rule.

The remote command used one sequential CUDA run with `batch-size 32`, `num-workers 8`, and no benchmark flag. Outputs are `summary.json` and `predictions.csv` in this directory. The protocol description is [temporal_input_screen_20261105.md](../temporal_input_screen_20261105.md); evaluator: `tools/evaluate_face_pair_temporal_input_screen.py`.
