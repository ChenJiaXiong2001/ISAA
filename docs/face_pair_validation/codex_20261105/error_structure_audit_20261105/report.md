# Cached error structure audit

This is a post-hoc diagnosis of one frozen epoch-83 checkpoint. It computes the original gamma=1 class weights from the training calibration cache and applies them to the validation logits only to describe errors. It does not search or select a new fusion rule, and it does not alter the checkpoint.

The cache contains 16,306 validation logits. HF is correct on 14,431 samples, BF on 15,126, and the gamma=1 soft fusion on 15,538. Both heads are wrong on 465 samples. In 682 fused errors the true class remains in the fused top five, so a substantial part of the residual is a ranking or margin problem rather than complete absence of class evidence. These counts are from this cache rerun and differ by a few samples from earlier CUDA re-evaluations because TF32/Inductor inference is not bitwise deterministic.

The largest fused confusion pairs are:

| True | Predicted | Count |
|---|---|---:|
| A011 阅读 | A029 玩手机或平板 | 32 |
| A011 阅读 | A012 写字 | 22 |
| A029 玩手机或平板 | A012 写字 | 22 |
| A048 恶心或呕吐 | A045 喝酒 | 20 |
| A031 抓挠 | A032 拍打身体 | 15 |
| A010 拍手 | A034 搓手 | 14 |

The hand-centric A011/A012/A029 cluster is the clearest structural target. A011 has 215/276 fused correct and 274/276 top-five coverage; A029 has 222/275 correct and 269/275 top-five coverage. Their errors are concentrated in each other, while both branches are often individually correct on different samples (A011: HF-only 38, BF-only 26; A029: HF-only 56, BF-only 12). This indicates complementary evidence exists but the current representation/readout does not reliably separate the fine-grained variants. It is consistent with a temporal or cross-region feature bottleneck, not evidence that a new gamma is needed.

The current `FacePairFusion` input provides torso-relative coordinates, score, cross-hand distance and cross-hand direction to independent hand and face encoders. The two encoded features are concatenated only at the final linear HF head; there is no explicit hand-to-face relation channel or hand displacement/velocity channel. A low-cost structural candidate for a short, fixed-protocol training screen is therefore:

1. Keep the existing 3-channel raw input, torso normalization, body branch, and gamma=1 inference unchanged.
2. Add two per-frame hand-to-face geometry channels to the HF local input, such as the vector from each hand wrist/palm centroid to the nearest valid face-token centroid (with a validity mask), or its normalized distance and direction.
3. Add first temporal differences of the hand-relative coordinates for the hand local branch, using zero differences on invalid transitions. Initialize the new channels so the original path remains available, and train from scratch under the mandatory compile protocol.

This candidate is motivated by the A011/A012/A029 confusion and can be falsified with a short paired screen against the unchanged architecture using fixed training-subject calibration and a predeclared clean Top-1 plus body-corruption score. It is not a claim of improvement. A separate, lower-priority candidate is an explicit face-to-body relative geometry channel for the A041/A048 face-centric confusion cluster (A041: 219/273 fused correct; A048: 236/275), but it should not be combined with the hand candidate in the first screen.

Evidence files: `summary.json`, `per_class.csv`, `top_confusions.csv`, and the CPU-only reproducer `tools/audit_face_pair_error_structure.py`. The validation cache was used for diagnosis only; no official labels were used to fit parameters.
