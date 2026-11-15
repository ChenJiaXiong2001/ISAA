# Subject-held-out screen of sample-level HF/BF confidence fusion

This CPU-only screen uses the cached epoch-83 checkpoint logits and labels from the training-subject calibration partition. For each of its two subjects, classwise gamma=1 accuracy weights are computed on the other subject, then fixed max-probability, top-two-margin, and normalized-entropy reliability gates are evaluated on the held-out subject. Temperatures remain 1.0. No official validation labels were read. Candidate beta values 0.5, 1, and 2 were fixed before this evaluation, matching the existing fusion-sweep grid.

| Reliability gate | Beta | P009 held-out delta | P018 held-out delta | Pooled correct | Pooled delta | Pooled NLL |
|---|---:|---:|---:|---:|---:|---:|
| max probability | 0.5 | -1 | -1 | 2999/3191 | -0.0627 pp | 0.2423 |
| max probability | 1 | -1 | -2 | 2998/3191 | -0.0940 pp | 0.2381 |
| max probability | 2 | -3 | -1 | 2997/3191 | -0.1254 pp | 0.2335 |
| top-two margin | 0.5 | -1 | -1 | 2999/3191 | -0.0627 pp | 0.2382 |
| top-two margin | 1 | -3 | +1 | 2999/3191 | -0.0627 pp | 0.2368 |
| top-two margin | 2 | -6 | +3 | 2998/3191 | -0.0940 pp | 0.2395 |
| normalized entropy | 0.5 | -1 | 0 | 3000/3191 | -0.0313 pp | 0.2446 |
| normalized entropy | 1 | -1 | +1 | 3001/3191 | 0.0000 pp | 0.2416 |
| normalized entropy | 2 | -3 | 0 | 2998/3191 | -0.0940 pp | 0.2367 |

Static gamma=1 scored 674/708 on P009 and 2327/2483 on P018 (3001/3191 pooled; NLL 0.2480). No dynamic candidate improved pooled top-1. Normalized entropy beta 1 tied top-1; all nine gates reduced pooled NLL, but the gates selected by higher beta traded more top-1 errors for better probability scores. Directions were not stable across held-out subjects: for example, top-two margin beta 2 lost six samples on P009 and gained three on P018. Pooled sample-bootstrap intervals are wide and include zero; they do not estimate performance on new subjects.

This is a small internal screen, not independent confirmation: only two calibration subjects exist, class weights change by fold, and these reliability families were previously explored in the official-validation fusion sweep. It does not justify adding a sample-level gate to the current gamma=1 model. If optimizing probability calibration becomes a separate target, entropy gating may merit a larger subject-held-out study; it has no demonstrated accuracy benefit here.

Evidence: `summary.json`, `folds.csv`, and reproducible CPU script `tools/analyze_face_pair_dynamic_fusion_subject_cv.py`. Cache SHA-256: `0a12b921db9f6002688f5ccd2687ecfa86706801d6d4a897bec5e0cdb6e2cc8e`.
