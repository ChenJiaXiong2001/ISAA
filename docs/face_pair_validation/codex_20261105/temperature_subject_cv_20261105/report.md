# Leave-One-Subject-Out Temperature Screen

This is a CPU-only analysis of the cached epoch-83 `best-face-pair` checkpoint (`aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`) using only its 3,191 training-subject calibration logits. The cache contains two calibration subjects: P009 (708 samples) and P018 (2,483 samples). No official validation labels were read, and no parameters were searched on a held-out subject.

For each fold, one subject fits scalar HF and BF temperatures by minimizing NLL on the other subject. The same fit subject also computes the classwise gamma=1 accuracy-proportional HF/BF weights. The held-out subject is evaluated with those fixed temperatures and weights. The baseline uses the same class weights with unit temperatures. Positive temperature scaling leaves each head's argmax unchanged, so the calibration-derived accuracy counts remain valid. The 20,000-replicate intervals are fixed-sample bootstrap intervals, not cross-subject confidence intervals.

| Fit subject | Held-out subject | Unit-T Top-1 | Fitted-T Top-1 | Delta | Unit NLL | Fitted NLL | HF T / BF T | Corrected / regressed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P018 | P009 | 95.1977% (674/708) | 94.9153% (672/708) | -0.2825pp | 0.2945 | 0.3191 | 1.4782 / 1.2905 | 0 / 2 |
| P009 | P018 | 93.7173% (2327/2483) | 93.9992% (2334/2483) | +0.2819pp | 0.2347 | 0.2444 | 1.5388 / 1.1833 | 10 / 3 |
| Combined (out-of-subject predictions) | Both subjects | 94.0458% (3001/3191) | 94.2024% (3006/3191) | +0.1566pp | 0.2480 | 0.2610 | -- | 10 / 5 |

The combined net gain of five is exactly the sum of the fold results: P018-fit/P009-test regresses by 2 with no corrections, while P009-fit/P018-test corrects 10 and regresses by 3, for net +7. The fold directions disagree. A subject-stratified paired sample bootstrap interval for the combined +0.1566pp is [-0.0627, +0.4074] pp; it does not estimate variation across new subjects. Fitted NLL increases in both folds. This does not support adding scalar temperature calibration to the inference path or training objective. It also does not establish that temperature scaling is useless with more calibration subjects; the present cache is only two subjects and is a screening result.

Evidence: `summary.json`, `predictions.csv`, and `tools/analyze_face_pair_temperature_subject_cv.py`. The cache source and manifest are recorded in `summary.json`; the exact source is `docs/face_pair_validation/fusion_suite_20261103/cache/best_calibration.pt`.
