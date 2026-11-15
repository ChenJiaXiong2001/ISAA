# Validation max-gap anchor repair

This is a fixed-checkpoint evaluation on the independent `validation` partition: 16,306 samples from 20 subjects. The checkpoint's gamma=1 fusion weights were calibrated on the separate training-subject calibration partition and were not recalibrated here. The exact command was:

```text
python -u tools/evaluate_face_pair_anchor_gaps_maxgap_20261105.py --checkpoint /home/xiong/Desktop/ISAA/outputs/best_face_pair_100_seed2_20261105/20261105-051202-2fcd5590/best.pt --manifest /home/xiong/Desktop/ISAA/outputs/best_face_pair_100_seed2_20261105/20261105-051202-2fcd5590/split_manifest.json --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip --output-dir /home/xiong/Desktop/ISAA/outputs/face_pair_anchor_gap_candidates_validation_20261105 --partition validation --compare-max-gap --batch-size 32 --num-workers 8
```

The evaluator uses the same fixed synthetic mask seed (1729) and conditions as the calibration screen. The corruption is applied after 100-frame temporal compression; starts are sampled independently per anchor/frame/person and expanded forward for gaps. This is a controlled stress test, not a detector log: it does not estimate real occlusion frequency, temporal duration, or detector score failure.

| Condition | Unmodified | Single-frame | Max-2 | Max-5 | Single gain/loss | Max-2 gain/loss | Max-5 gain/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 95.511% | 95.505% | 95.505% | 95.499% | +1/-2 | +1/-2 | +1/-3 |
| Independent 5% | 91.764% | 95.284% | 95.431% | 95.462% | +694/-120 | +728/-130 | +733/-130 |
| Contiguous gap 2 | 92.487% | 92.494% | 95.290% | 95.462% | +1/-0 | +580/-123 | +614/-129 |
| Contiguous gap 5 | 93.794% | 93.794% | 93.788% | 95.358% | +1/-1 | +1/-2 | +369/-114 |
| Contiguous gap 10 | 94.658% | 94.652% | 94.652% | 94.652% | +1/-2 | +1/-2 | +1/-2 |

The max-5 rule is the strongest candidate in this fixed stress test: it adds 3.698 percentage points for independent 5% loss, 2.974 points for 2-frame gaps, and 1.564 points for 5-frame gaps. Subject-bootstrap 95% intervals are [3.314, 4.168], [2.594, 3.665], and [1.281, 2.074] percentage points respectively. Clean input changes slightly downward by 0.012 points, with interval [-0.054, 0.007]. Ten-frame gaps are outside the bound and show no improvement.

The interpolator needs valid endpoints and future frames. Its maximum look-ahead is 1, 2, or 5 compressed frames for the three candidates. At a 30 FPS example rate this is approximately 33, 67, or 167 ms before model computation. No synchronized repair latency benchmark was run in this validation job, so these are protocol delays rather than measured runtime. The full output has `summary.json`, `predictions.csv`, and `subject_paired.json`; the latter reports paired gain/loss and subject-cluster bootstrap intervals.

The result is evidence for robustness under the specified synthetic missing-anchor process. It does not prove that max-5 raises the clean official accuracy, and it should not be described as a deployed detector improvement without a real detector-failure capture or an online latency measurement.
