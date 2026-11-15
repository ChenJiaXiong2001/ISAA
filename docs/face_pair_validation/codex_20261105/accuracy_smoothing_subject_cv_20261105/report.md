# Gamma=1 Accuracy-Weight Pseudocount Screen

This CPU-only screen reuses the epoch-83 `best-face-pair` training-subject calibration logit cache (checkpoint SHA256 `aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`). It contains 3,191 examples from only P009 (708) and P018 (2,483). No official validation labels were read.

For each leave-one-subject-out fold, HF/BF per-class accuracy weights use the original gamma=1 ratio. The candidate adds the same fixed pseudo-count `alpha` to the two heads' correct counts before forming that ratio: `w=(correct_HF+alpha)/(correct_HF+correct_BF+2*alpha)`. The class weights are fit only on the other subject. Fixed alphas 0.5, 1, 2 and 5 were screened without choosing from held-out performance. Unit temperatures and uniform per-sample fusion remain fixed.

| Alpha | P009 held out (fit P018) | P018 held out (fit P009) | Combined Top-1 | Net gain over alpha=0 | Combined NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 674/708 (95.1977%) | 2327/2483 (93.7173%) | 3001/3191 (94.0458%) | -- | 0.24798 |
| 0.5 | 674/708 | 2327/2483 | 3001/3191 (94.0458%) | 0 | 0.24811 |
| 1 | 674/708 | 2328/2483 | 3002/3191 (94.0771%) | +1 | 0.24824 |
| 2 | 674/708 | 2328/2483 | 3002/3191 (94.0771%) | +1 | 0.24847 |
| 5 | 675/708 | 2327/2483 | 3002/3191 (94.0771%) | +1 | 0.24905 |

The largest pooled difference is one correct sample (+0.0313pp), confidence-weighted calibration NLL is worse for every nonzero alpha, and gains move between folds as alpha changes. This is not evidence of a clean accuracy improvement and does not justify changing the specified unsmoothed gamma=1 method. With just two subjects, the observed one-sample change is a screening observation, not evidence of broader generalization.

Evidence: `summary.json`, `predictions.csv`, and `tools/screen_face_pair_accuracy_smoothing_subject_cv.py`; source cache and ordered member manifest are the same as documented in `temperature_subject_cv_20261105/report.md`.
