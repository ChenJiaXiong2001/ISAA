# Temporal input construction screen

This frozen-checkpoint screen tests whether the confidence-weighted temporal compression used by `best-face-pair` is an input bottleneck. It uses the seed2 checkpoint and its saved gamma=1 calibration weights on the first 4096 members of the fixed validation manifest. No labels are used to choose a mode, no model parameters are changed, and no synthetic corruption is applied.

| Input mode | HF | BF | gamma=1 soft |
|---|---:|---:|---:|
| confidence-weighted compression | 87.8418% | 92.4805% | **94.4824%** |
| no compression, start crop | 86.5723% | 90.8936% | 93.3594% |
| no compression, center crop | 86.9629% | 91.1377% | 93.4082% |
| no compression, end crop | 87.3291% | 90.6006% | 93.5303% |

The best no-compression mode is 0.9521 percentage points below the default on this fixed prefix; all three are lower. The result supports retaining confidence-weighted compression for the current model. It does not test a retrained model under a different compression rule, so it does not rule out a future input-specific retraining experiment.

Command and machine-readable outputs are recorded in `summary.json` and `predictions.csv`. The evaluator is `tools/evaluate_face_pair_temporal_input_screen.py`; the screen used CUDA inference on the A6000 and refused a CPU fallback.
