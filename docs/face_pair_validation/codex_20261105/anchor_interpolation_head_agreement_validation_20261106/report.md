# Frozen validation: anchor repair head-agreement gate

This is the pre-registered validation of the rule fixed from the two-subject training calibration screen. The model, checkpoint, gamma=1 fusion weights, mask seed, and sample order were frozen before this traversal. Validation labels are used here only to measure the result, not to choose a rule.

Checkpoint SHA256: `03ac2f6a56483378201b85733f5e9bfb03f1832d60a82c002273ebcc694577b9` (epoch 86 best, 16,306 validation samples). Validation members SHA256: `82fd60410fd04e381c65d3bb9f24f18f7993124ee2581902942fbfb4a2ecddc0`. Mask seed: `1729`. Fusion: gamma=1 HF/BF original-accuracy-proportional soft fusion.

Fixed rule: after temporal compression, interpolate the four shoulder/hip anchors across gaps of at most five valid frames; use the repaired gamma=1 soft prediction only when repaired HF and BF argmax classes agree; otherwise fall back to the unmodified gamma=1 soft prediction. No endpoint extrapolation or invalid-frame crossing is allowed.

| condition | unmodified | always max-5 repair | gated max-5 | always corrected/regressed | gated corrected/regressed |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 95.1368% | 95.1429% (+0.0061pp) | 95.1368% (+0.0000pp) | 1 / 0 | 0 / 0 |
| independent 5% | 90.9052% | 95.1184% (+4.2132pp) | 93.4932% (+2.5880pp) | 823 / 136 | 434 / 12 |
| contiguous gap 2 | 91.7147% | 95.0570% (+3.3423pp) | 93.7391% (+2.0245pp) | 690 / 145 | 347 / 17 |
| contiguous gap 5 | 93.3460% | 95.0816% (+1.7356pp) | 94.3272% (+0.9812pp) | 416 / 133 | 178 / 18 |
| contiguous gap 10 | 94.3456% | 94.3518% (+0.0061pp) | 94.3456% (+0.0000pp) | 1 / 0 | 0 / 0 |

The gate reduces the always-repair recovery but sharply limits regressions. On this fixed validation traversal, it keeps clean accuracy unchanged, recovers 434 samples with 12 regressions at independent 5% corruption, 347/17 at gap 2, and 178/18 at gap 5. The rule does not repair a ten-frame gap. These are fixed-sample results for the specified synthetic corruption and do not establish real-world occlusion performance.

The raw evaluator output is `best_anchor_gaps/summary.json` and `best_anchor_gaps/predictions.csv` under the server verification directory. The paired gate aggregation is in `screen.json`; its source prediction SHA256 is `4892db4240784d26b24e22bfa342e7a57dcf9eb6d36fad7597e34ffb40c8f901`. The full evaluator took about 1,662 seconds on the A6000. Inference latency for running both original and repaired inputs was not benchmarked, so deployment cost remains open.
