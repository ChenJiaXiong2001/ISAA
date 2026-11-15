# Frozen body-point ablation

The existing seed2 epoch-100 checkpoint was evaluated without retraining on 4,096 randomly selected XSub validation samples. Every condition uses the same ordered sample list, checkpoint, gamma=1 class weights, and per-point/per-frame/person random draws. At each level, anchor, non-anchor, and all-body ablations use the same underlying draws. Clean samples are compared by sample, not by separate aggregate datasets.

| Condition | HF Top-1 | BF Top-1 | Gamma=1 Top-1 | Valid point-frames hidden |
|---|---:|---:|---:|---:|
| Clean | 88.4766% | 93.0176% | 95.7275% | 0 |
| Anchors 5% | 65.6738% | 91.3574% | 92.0166% | 74,506 |
| Non-anchors 5% | 88.4766% | 88.8916% | 94.9219% | 353,230 |
| All body 5% | 65.6738% | 86.9873% | 88.5498% | 427,736 |
| Anchors 10% | 47.2656% | 89.3799% | 87.8906% | 149,078 |
| Non-anchors 10% | 88.4766% | 84.3750% | 94.1406% | 705,743 |
| All body 10% | 47.2656% | 80.9814% | 79.8584% | 854,821 |

The four anchors are left/right shoulders and hips (indices 5, 6, 11, 12). In the current model their valid points define the hand/face coordinate center and shoulder-to-hip distance scale, while they are also body encoder inputs. A 5% anchor-only mask causes an HF drop of 22.8027 percentage points and gamma=1 drop of 3.7109 points. Paired 20,000-replicate bootstrap intervals for the changes are [-24.2188, -21.4111] and [-4.3945, -3.0518] points respectively. At 10%, the corresponding changes are -41.2109 and -7.8369 points. Non-anchor corruption leaves every HF prediction unchanged, consistent with the input dependency in the implementation.

Equal per-point probabilities do not mean equal corruption severity: the non-anchor set contains 19 points, including indices 1–4 that are not directly used by the current body encoder, and hides substantially more valid point-frames. Anchor masking changes both the body branch and the input reference of the local branches. These measurements do not isolate the two mechanisms; a separate clean-derived-feature oracle intervention is needed. The per-frame independently missing-point simulation does not establish performance under long occlusions or real detector failures.

The model was run in frozen eager inference; this does not change the requirement that training must use torch.compile. `summary.json` records the checkpoint SHA-256, sample digest, random draw settings, and condition totals. `predictions.csv` contains every sample's label and all branch predictions, permitting paired analysis. Script: `tools/evaluate_face_pair_body_ablation.py`.
