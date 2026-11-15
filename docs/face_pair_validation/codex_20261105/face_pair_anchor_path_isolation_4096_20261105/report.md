# Anchor path isolation

This frozen-checkpoint experiment uses the same 4,096 validation samples and per-point random draws as `../face_pair_anchor_ablation_4096_20261105/`. The raw-input `FacePairFusion.forward` torso center, shoulder-to-hip scale, cross-hand distance and direction calculations were reproduced to construct an equivalent 8-channel input. Clean 3-channel and 8-channel inference agreed on all 4,096 samples: maximum absolute difference was exactly 0 for hard-fusion, HF and BF logits, with zero prediction disagreements. The program checks all logits per batch and aborts if they are not close.

The intervention then hides the same valid shoulder/hip anchors in the raw three body channels. The recomputed condition lets the model derive its local features from the corrupted anchors. The preserved condition instead supplies the clean 8-channel hand/face relative coordinates and cross-hand distance/direction, while keeping the body-channel anchor mask identical. The preserved condition is an oracle diagnostic: the clean derived features would not be available during genuine occlusion.

| Anchors hidden per frame | Local derived features | HF Top-1 | BF Top-1 | Gamma=1 Top-1 |
|---:|---|---:|---:|---:|
| 0% | clean | 88.4766% | 93.0176% | 95.7275% |
| 5% | recomputed from corrupted input | 65.6738% | 91.3574% | 92.0166% |
| 5% | preserved clean oracle | 88.4766% | 91.3330% | 95.5078% |
| 10% | recomputed from corrupted input | 47.2656% | 89.3799% | 87.8906% |
| 10% | preserved clean oracle | 88.4766% | 89.4287% | 95.1416% |

With identical anchor masks, preserving clean local features restores every HF prediction to its clean prediction, while BF remains affected by missing body input. At 5%, gamma=1 improves by 143/4096 = 3.4912 percentage points relative to the corrupted/recomputed condition, with paired 20,000-replicate bootstrap 95% interval [2.8564, 4.1504] points; at 10% the increase is 297/4096 = 7.2510 points [6.3965, 8.1299]. These isolate dependence on the combined local derived inputs, not a particular center-versus-scale cause, and are not a deployable accuracy claim.

`summary.json` records actual hidden point counts, checkpoint and sample hashes, and equivalence diagnostics; `predictions.csv` retains all paired predictions. Inference is frozen eager. The training requirement for torch.compile is unchanged.
