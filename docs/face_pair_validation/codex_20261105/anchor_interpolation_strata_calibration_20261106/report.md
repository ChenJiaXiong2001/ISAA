# Anchor interpolation stratification

This is a read-only analysis of the frozen training-subject calibration predictions. It does not train, tune, or rerun model inference.

Source: `C:\Users\15637\PycharmProjects\ISAA\docs\face_pair_validation\codex_20261105\anchor_interpolation\calibration\predictions.csv`; SHA256 `66fdc107a24c3f6d7c0ffa1aa38bbf45e51daaec08aabb8fec9ac52386ab3432`; samples: 1420.

## Overall

| condition | before | repaired | delta (pp) | corrected | regressed | drop reduction (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 94.3662% | 94.3662% | +0.0000 | 0 | 0 | n/a |
| point05 | 85.7042% | 92.7465% | +7.0423 | 117 | 17 | +7.0423 |
| point10 | 73.9437% | 89.1549% | +15.2113 | 230 | 14 | +15.2113 |

## Classes with largest point-10 improvement

| class | n | delta (pp) | drop before (pp) | drop after (pp) | recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| A028 | 24 | +62.500 | 66.667 | 4.167 | 93.8% |
| A054 | 24 | +58.333 | 66.667 | 8.333 | 87.5% |
| A004 | 24 | +54.167 | 58.333 | 4.167 | 92.9% |
| A044 | 24 | +45.833 | 70.833 | 25.000 | 64.7% |
| A032 | 24 | +45.833 | 45.833 | 0.000 | 100.0% |
| A025 | 24 | +41.667 | 62.500 | 20.833 | 66.7% |
| A012 | 24 | +41.667 | 50.000 | 8.333 | 83.3% |
| A001 | 24 | +41.667 | 58.333 | 16.667 | 71.4% |
| A039 | 24 | +37.500 | 45.833 | 8.333 | 81.8% |
| A037 | 24 | +33.333 | 33.333 | 0.000 | 100.0% |
| A031 | 24 | +33.333 | 45.833 | 12.500 | 72.7% |
| A041 | 23 | +30.435 | 47.826 | 17.391 | 63.6% |
| A038 | 24 | +29.167 | 33.333 | 4.167 | 87.5% |
| A036 | 24 | +29.167 | 70.833 | 41.667 | 41.2% |
| A050 | 24 | +25.000 | 29.167 | 4.167 | 85.7% |

Across 60 classes, point-5 improvement was positive for 29, zero for 26, and negative for 5; point-10 improvement was positive for 34, zero for 24, and negative for 2.
The class-level Pearson correlation between pre-repair point-drop loss and interpolation gain was r=0.922 at 5% and r=0.939 at 10%.

At point-10, the largest wrong-pair reductions (Atrue -> Apred, before -> after) were A054->A053 12->1, A004->A014 10->0, A005->A053 11->3, A028->A003 9->1, A001->A049 8->1.

## Subject range

| subject | n | point-10 delta (pp) | corrected | regressed |
| --- | ---: | ---: | ---: | ---: |
| P013 | 1063 | +16.369 | 185 | 11 |
| P014 | 357 | +11.765 | 45 | 3 |

## Interpretation

The rule's value is concentrated in classes and subjects whose point-drop accuracy loss is large; this supports an input repair/temporal continuity mechanism rather than a clean-accuracy change.
Both calibration subjects improved at point-10, with a subject-level range of +11.765 to +16.369 percentage points.
Classes with small or negative deltas should not be used to claim universal benefit. The class table is descriptive because the class labels and corruption masks were fixed before this audit.

Full machine-readable class and subject tables are in `stratification.json`.
