# Normalized rank fusion LOSO screen

This CPU-only candidate was fixed before evaluation. For each sample and each head, the 60 logits are converted to descending ranks: the highest class gets score 60 and the lowest gets score 1. Exact logit ties receive their arithmetic-average rank. Scores are divided by `60*61/2`, then the two normalized rank vectors are mixed with the held-out fold's gamma=1 class accuracy weights. The final argmax uses the lowest class index on an exact tie. Static probability-domain gamma=1 uses the same held-out fold weights. No official validation labels were read.

| Fit subject | Held-out subject | Static gamma=1 | Rank fusion | Delta | Rank pseudo-NLL |
|---|---:|---:|---:|---:|---:|
| P018 | P009 | 674/708 (95.1977%) | 652/708 (92.0904%) | -3.1073 pp | 3.4236 |
| P009 | P018 | 2327/2483 (93.7173%) | 2305/2483 (92.8313%) | -0.8860 pp | 3.4193 |
| Pooled | Both | 3001/3191 (94.0458%) | 2957/3191 (92.6669%) | -1.3789 pp | 3.4203 |

The rank candidate corrected 43 static errors but introduced 87 new errors. No exact logit ties occurred in either head; the final rank mixture had tied maxima in 5 and 6 samples in the two folds, resolved by the fixed lowest-index rule. Its pseudo-NLL is not a calibrated posterior NLL and is much worse than static probability NLL (0.2480), which is expected from replacing calibrated logit magnitudes with ranks.

This result rejects normalized rank fusion as an accuracy improvement for this checkpoint and calibration protocol. It does not motivate training or a larger rank search. Only two training calibration subjects were available, so the result is a bounded internal screen rather than a new-subject generalization estimate.

Evidence: `summary.json`, `predictions.csv`, and `tools/analyze_face_pair_rank_fusion_subject_cv.py`. Cache SHA-256: `0a12b921db9f6002688f5ccd2687ecfa86706801d6d4a897bec5e0cdb6e2cc8e`.
