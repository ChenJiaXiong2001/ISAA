# Frozen Model: Single-Frame Anchor Interpolation

## Rule and Protocol

The frozen `best-face-pair` epoch-100 checkpoint (`SHA256 25037c807a45e8ec647f89b58557f63c9a68c89ce28a62949bc5b99ce6e47432`) and its saved training-subject calibration counts determine the unchanged gamma=1 HF/BF fusion weights. No weights are trained or changed. Inference uses the frozen model in eager mode; the project's mandatory training compilation policy is unchanged.

After confidence-weighted temporal compression to 100 frames, the input repair examines only shoulder/hip anchors 5, 6, 11 and 12. A missing (`score=0`) anchor is filled only when the same anchor is present in both immediately adjacent valid frames. Each of its x, y and score values is the arithmetic mean of those two observations. It neither extrapolates endpoints nor fills a gap of two or more adjacent missing frames. It uses no class label or knowledge of whether the missing point is synthetic; other joints are unchanged. The method uses the following compressed frame and thus requires offline processing or at least one compressed-frame delay.

The rule was fixed after screening the 1,420 training-subject calibration samples and before this single 16,306-sample validation traversal. The calibration set consists of two subjects and also supplies the fusion weights; it is not an independent test. Validation comparisons use the same checkpoint, sample order, 32-sample batches, and the same per-person, per-frame, per-body-point random draws (seed 1729) before and after repair. The 5%/10% tests zero xy and score for independent body points 0–22; the 10% mask contains the 5% mask. This artificial pointwise dropout differs from real occlusion, tracking failures, and temporal blocks of missing frames. The clean condition includes naturally absent points and receives the same repair rule.

## Results

| Dataset | Condition | Original gamma=1 Top-1 | Repaired Top-1 | Change (pp) | Corrected / Regressed |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration (1,420) | Clean | 94.3662% | 94.3662% | 0.0000 | 0 / 0 |
| Calibration | Body point 5% | 85.7042% | 92.7465% | +7.0423 | 117 / 17 |
| Calibration | Body point 10% | 73.9437% | 89.1549% | +15.2113 | 230 / 14 |
| Validation (16,306) | Clean | 15,574 = 95.5109% | 15,573 = 95.5047% | -0.0061 | 1 / 2 |
| Validation | Body point 5% | 14,410 = 88.3724% | 15,357 = 94.1801% | +5.8077 | 1,067 / 120 |
| Validation | Body point 10% | 13,048 = 80.0196% | 14,886 = 91.2915% | +11.2719 | 1,973 / 135 |

The validation paired sample-bootstrap 95% intervals for repaired minus original accuracy are [-0.0307, +0.0123] pp for clean, [+5.4091, +6.2124] pp for body 5%, and [+10.7629, +11.7932] pp for body 10% (20,000 replicates). These are fixed-sample intervals and do not estimate between-subject variation. The rule repaired 926 anchor point-frames in clean validation, 263,212 under body 5%, and 471,881 under body 10%. The clean comparison is not proof of exact equality.

The frozen model is substantially more robust to this particular artificial single-frame dropout with the rule, with one fewer correct clean prediction in this validation run. This is evidence for an inference-side robustness candidate, not proof of improved standard accuracy or real-world occlusion performance. Longer outages are not repaired by the rule; an independent calibration-set stress test of temporal blocks is being conducted separately.

## Evidence

- `calibration/summary.json`, `calibration/predictions.csv`, `calibration/paired_summary.json`: training-subject calibration screen.
- `validation/summary.json`, `validation/predictions.csv`, `validation/paired_summary.json`: complete validation and paired sample bootstrap; each row retains its original sample name.
- Evaluation: `tools/evaluate_face_pair_anchor_interpolation.py`; SHA256 of the complete-validation script `1eba4c37f325aedb98679f6dd0b412f84bd1b380d40ef9eab7c1ff546a18c371`.
- Paired statistics: `tools/summarize_face_pair_anchor_interpolation.py`; two CPU boundary tests in `tests/test_face_pair_anchor_interpolation.py` pass.
