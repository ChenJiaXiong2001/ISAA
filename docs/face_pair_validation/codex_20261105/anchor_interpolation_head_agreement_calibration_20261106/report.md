# Anchor repair head-agreement gate calibration screen

This is a read-only analysis of frozen paired predictions. The rule is fixed before any validation evaluation.

Source: `C:\Users\15637\AppData\Local\Temp\face_pair_anchor_gap_calibration_20261105_predictions.csv`; SHA256 `a40e0c2ec9e9b2b37a0acdb5487043fd5f75936d55cc54db9f9343310929e67d`; samples: 1420.
Checkpoint SHA256: `25037c807a45e8ec647f89b58557f63c9a68c89ce28a62949bc5b99ce6e47432`.
Calibration members SHA256: `9b56a5b88334edfb127434815ba4bf09a2fa9249efc27be4423c65bd8783aba7`; mask seed: `1729`.
Fusion: gamma=1 HF/BF original-accuracy-proportional soft fusion.

Gate: use the repaired gamma=1 soft prediction only when repaired HF and BF argmax classes agree; otherwise use the unmodified gamma=1 soft prediction.

| condition | method | unmodified | always repaired | always delta (pp) | always corrected/regressed | gated | gated delta (pp) | gated corrected/regressed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | max1 | 94.366% | 94.366% | +0.000 | 0/0 | 94.366% | +0.000 | 0/0 |
| clean | max2 | 94.366% | 94.366% | +0.000 | 0/0 | 94.366% | +0.000 | 0/0 |
| clean | max5 | 94.366% | 94.366% | +0.000 | 0/0 | 94.366% | +0.000 | 0/0 |
| independent05 | max1 | 89.718% | 94.014% | +4.296 | 77/16 | 92.606% | +2.887 | 41/0 |
| independent05 | max2 | 89.718% | 93.944% | +4.225 | 78/18 | 92.887% | +3.169 | 45/0 |
| independent05 | max5 | 89.718% | 94.014% | +4.296 | 79/18 | 92.887% | +3.169 | 45/0 |
| gap2 | max1 | 90.493% | 90.493% | +0.000 | 0/0 | 90.493% | +0.000 | 0/0 |
| gap2 | max2 | 90.493% | 93.803% | +3.310 | 64/17 | 92.817% | +2.324 | 34/1 |
| gap2 | max5 | 90.493% | 94.225% | +3.732 | 66/13 | 93.028% | +2.535 | 36/0 |
| gap5 | max1 | 91.901% | 91.901% | +0.000 | 0/0 | 91.901% | +0.000 | 0/0 |
| gap5 | max2 | 91.901% | 91.901% | +0.000 | 0/0 | 91.901% | +0.000 | 0/0 |
| gap5 | max5 | 91.901% | 93.944% | +2.042 | 46/17 | 93.028% | +1.127 | 16/0 |
| gap10 | max1 | 92.887% | 92.887% | +0.000 | 0/0 | 92.887% | +0.000 | 0/0 |
| gap10 | max2 | 92.887% | 92.887% | +0.000 | 0/0 | 92.887% | +0.000 | 0/0 |
| gap10 | max5 | 92.887% | 92.887% | +0.000 | 0/0 | 92.887% | +0.000 | 0/0 |

## Decision

The max5 head-agreement gate is the candidate to register for a later locked validation traversal. It trades recovery for zero observed calibration regressions in the independent-5%, gap-2, and gap-5 conditions, while leaving clean and gap-10 unchanged on this calibration set.

This screen does not establish deployment benefit. It uses two training-subject calibration members, and the checkpoint's calibration counts also define the gamma=1 fusion weights. Validation labels were not used to select the rule. A later validation run must use the exact rule and report clean accuracy, each corruption condition, subject-level results, and added two-forward-pass latency.
