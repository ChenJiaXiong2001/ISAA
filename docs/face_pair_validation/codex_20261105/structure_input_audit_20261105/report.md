# Anchor interpolation stratification

This is a read-only analysis of the frozen validation predictions. It does not train, tune, or rerun model inference.

Source: `C:\Users\15637\PycharmProjects\ISAA\docs\face_pair_validation\codex_20261105\anchor_interpolation\validation\predictions.csv`; SHA256 `7890934ec79f792e94b4492af1763589ecc82f8f79bbcc82e99971663404e90b`; samples: 16306.

## Overall

| condition | before | repaired | delta (pp) | corrected | regressed | drop reduction (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 95.5109% | 95.5047% | -0.0061 | 1 | 2 | n/a |
| point05 | 88.3724% | 94.1801% | +5.8077 | 1067 | 120 | +5.8077 |
| point10 | 80.0196% | 91.2915% | +11.2719 | 1973 | 135 | +11.2719 |

## Classes with largest point-10 improvement

| class | n | delta (pp) | drop before (pp) | drop after (pp) | recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| A012 | 274 | +40.511 | 46.350 | 5.839 | 87.4% |
| A038 | 272 | +38.971 | 42.647 | 3.676 | 91.4% |
| A044 | 276 | +36.957 | 51.812 | 14.855 | 71.3% |
| A033 | 273 | +36.630 | 44.322 | 7.692 | 82.6% |
| A032 | 275 | +32.727 | 37.455 | 4.727 | 87.4% |
| A025 | 273 | +30.403 | 45.788 | 15.385 | 66.4% |
| A004 | 275 | +30.182 | 35.636 | 5.455 | 84.7% |
| A039 | 276 | +29.348 | 31.884 | 2.536 | 92.0% |
| A054 | 276 | +28.261 | 43.841 | 15.580 | 64.5% |
| A011 | 276 | +25.362 | 20.290 | -5.072 | 125.0% |
| A031 | 276 | +23.913 | 30.072 | 6.159 | 79.5% |
| A028 | 276 | +23.551 | 28.986 | 5.435 | 81.2% |
| A019 | 273 | +22.344 | 26.374 | 4.029 | 84.7% |
| A041 | 273 | +19.048 | 36.630 | 17.582 | 52.0% |
| A036 | 272 | +16.912 | 59.926 | 43.015 | 28.2% |

Across 60 classes, point-5 improvement was positive for 46, zero for 7, and negative for 7; point-10 improvement was positive for 53, zero for 3, and negative for 4.
The class-level Pearson correlation between pre-repair point-drop loss and interpolation gain was r=0.978 at 5% and r=0.910 at 10%.

At point-10, the largest wrong-pair reductions (Atrue -> Apred, before -> after) were A012->A030 115->27, A054->A053 98->34, A004->A014 64->1, A039->A010 55->6, A033->A046 48->2.

## Subject range

| subject | n | point-10 delta (pp) | corrected | regressed |
| --- | ---: | ---: | ---: | ---: |
| P023 | 356 | +17.135 | 63 | 2 |
| P011 | 356 | +16.292 | 59 | 1 |
| P024 | 354 | +15.819 | 59 | 3 |
| P032 | 356 | +14.607 | 57 | 5 |
| P010 | 711 | +13.361 | 101 | 6 |
| P021 | 1059 | +13.126 | 151 | 12 |
| P036 | 356 | +12.640 | 52 | 7 |
| P020 | 689 | +12.046 | 97 | 14 |
| P007 | 5688 | +11.252 | 671 | 31 |
| P029 | 359 | +11.142 | 42 | 2 |
| P022 | 341 | +10.850 | 43 | 6 |
| P026 | 360 | +10.833 | 40 | 1 |
| P037 | 1404 | +10.043 | 147 | 6 |
| P003 | 1422 | +9.916 | 152 | 11 |
| P006 | 359 | +9.749 | 39 | 4 |
| P012 | 354 | +9.605 | 39 | 5 |
| P030 | 359 | +9.192 | 36 | 3 |
| P033 | 357 | +8.683 | 34 | 3 |
| P039 | 713 | +7.714 | 63 | 8 |
| P040 | 353 | +6.516 | 28 | 5 |

## Interpretation

The rule's value is concentrated in classes and subjects whose point-drop accuracy loss is large; this supports an input repair/temporal continuity mechanism rather than a clean-accuracy change.
All 20 of 20 validation subjects improved at point-10, with a subject-level range of +6.516 to +17.135 percentage points.
Classes with small or negative deltas should not be used to claim universal benefit. The class table is descriptive because the class labels and corruption masks were fixed before this audit.

Full machine-readable class and subject tables are in `stratification.json`.
