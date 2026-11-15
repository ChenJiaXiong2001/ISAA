# Fixed Dynamic-Confidence Screen Across Calibration Subjects

This CPU-only screen uses cached logits from the epoch-83 `best-face-pair` checkpoint (SHA256 `aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`). The only data are the 3,191 training-subject calibration samples: P009 (708) and P018 (2,483). The official validation set was not read.

Each direction fits classwise original gamma=1 accuracy weights on one subject and evaluates the other subject. The static gamma=1 prior remains fixed per class; a confidence-derived sample reliability rescales the HF/BF contributions only at inference. Uniform static fusion is the control. Confidence measures and beta values were fixed before scoring either held-out subject: maximum probability, top-two probability margin, or one minus normalized entropy, each with beta 0.5, 1, and 2. No candidate was selected or tuned on a held-out subject. This is a screen across two people, not a population estimate.

| Candidate | P009 held out (fit P018) | P018 held out (fit P009) | Combined Top-1 | Delta vs static | Combined NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static gamma=1 | 674/708 (95.1977%) | 2327/2483 (93.7173%) | 3001/3191 (94.0458%) | -- | 0.2480 |
| Max probability, beta 0.5 | 673/708 | 2326/2483 | 2999/3191 (93.9831%) | -0.0627pp | 0.2423 |
| Max probability, beta 1 | 673/708 | 2325/2483 | 2998/3191 (93.9517%) | -0.0940pp | 0.2381 |
| Max probability, beta 2 | 671/708 | 2326/2483 | 2997/3191 (93.9204%) | -0.1254pp | 0.2335 |
| Margin, beta 0.5 | 673/708 | 2326/2483 | 2999/3191 (93.9831%) | -0.0627pp | 0.2382 |
| Margin, beta 1 | 671/708 | 2328/2483 | 2999/3191 (93.9831%) | -0.0627pp | 0.2368 |
| Margin, beta 2 | 668/708 | 2330/2483 | 2998/3191 (93.9517%) | -0.0940pp | 0.2395 |
| Normalized entropy, beta 0.5 | 673/708 | 2327/2483 | 3000/3191 (94.0144%) | -0.0313pp | 0.2446 |
| Normalized entropy, beta 1 | 673/708 | 2328/2483 | 3001/3191 (94.0458%) | 0.0000pp | 0.2416 |
| Normalized entropy, beta 2 | 671/708 | 2327/2483 | 2998/3191 (93.9517%) | -0.0940pp | 0.2367 |

None beats static gamma=1 on combined accuracy. Most candidates lose accuracy on both held-out subjects; a few small mixed-direction changes cancel out. Several lower NLL values do not improve top-1, while the temperature screen also increased NLL. This screen gives no reason to replace the current clean-accuracy fusion path with per-sample confidence gating. Since ten alternatives were scored on the same two subjects, the table is descriptive and should not be interpreted as a confirmatory ranking.

Evidence: `summary.json`, `predictions.csv`, and `tools/screen_face_pair_dynamic_confidence_subject_cv.py`; source logits and member manifest are the same training-subject calibration cache cited in the preceding temperature screen.
