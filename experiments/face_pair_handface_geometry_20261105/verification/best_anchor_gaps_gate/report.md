# Anchor repair head-agreement gate calibration screen

This is a read-only analysis of frozen paired predictions. The rule is fixed before any validation evaluation.

Source: `/home/xiong/Desktop/ISAA/experiments/face_pair_handface_geometry_20261105/verification/best_anchor_gaps/predictions.csv`; SHA256 `4892db4240784d26b24e22bfa342e7a57dcf9eb6d36fad7597e34ffb40c8f901`; samples: 16306.
Checkpoint SHA256: `03ac2f6a56483378201b85733f5e9bfb03f1832d60a82c002273ebcc694577b9`.
Calibration members SHA256: `9b56a5b88334edfb127434815ba4bf09a2fa9249efc27be4423c65bd8783aba7`; mask seed: `1729`.
Fusion: gamma=1 HF/BF original-accuracy-proportional soft fusion.

Gate: use the repaired gamma=1 soft prediction only when repaired HF and BF argmax classes agree; otherwise use the unmodified gamma=1 soft prediction.

| condition | method | unmodified | always repaired | always delta (pp) | always corrected/regressed | gated | gated delta (pp) | gated corrected/regressed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | max1 | 95.137% | 95.137% | +0.000 | 0/0 | 95.137% | +0.000 | 0/0 |
| clean | max2 | 95.137% | 95.137% | +0.000 | 0/0 | 95.137% | +0.000 | 0/0 |
| clean | max5 | 95.137% | 95.143% | +0.006 | 1/0 | 95.137% | +0.000 | 0/0 |
| independent05 | max1 | 90.905% | 94.750% | +3.845 | 767/140 | 93.064% | +2.159 | 365/13 |
| independent05 | max2 | 90.905% | 95.088% | +4.183 | 820/138 | 93.481% | +2.576 | 433/13 |
| independent05 | max5 | 90.905% | 95.118% | +4.213 | 823/136 | 93.493% | +2.588 | 434/12 |
| gap2 | max1 | 91.715% | 91.721% | +0.006 | 1/0 | 91.715% | +0.000 | 0/0 |
| gap2 | max2 | 91.715% | 94.861% | +3.146 | 666/153 | 93.530% | +1.815 | 313/17 |
| gap2 | max5 | 91.715% | 95.057% | +3.342 | 690/145 | 93.739% | +2.024 | 347/17 |
| gap5 | max1 | 93.346% | 93.346% | +0.000 | 0/0 | 93.346% | +0.000 | 0/0 |
| gap5 | max2 | 93.346% | 93.340% | -0.006 | 0/1 | 93.346% | +0.000 | 0/0 |
| gap5 | max5 | 93.346% | 95.082% | +1.736 | 416/133 | 94.327% | +0.981 | 178/18 |
| gap10 | max1 | 94.346% | 94.352% | +0.006 | 1/0 | 94.346% | +0.000 | 0/0 |
| gap10 | max2 | 94.346% | 94.352% | +0.006 | 1/0 | 94.346% | +0.000 | 0/0 |
| gap10 | max5 | 94.346% | 94.352% | +0.006 | 1/0 | 94.346% | +0.000 | 0/0 |

## Decision

The max5 head-agreement gate is the candidate to register for a later locked validation traversal. It trades recovery for zero observed calibration regressions in the independent-5%, gap-2, and gap-5 conditions, while leaving clean and gap-10 unchanged on this calibration set.

This screen does not establish deployment benefit. It uses two training-subject calibration members, and the checkpoint's calibration counts also define the gamma=1 fusion weights. Validation labels were not used to select the rule. A later validation run must use the exact rule and report clean accuracy, each corruption condition, subject-level results, and added two-forward-pass latency.
