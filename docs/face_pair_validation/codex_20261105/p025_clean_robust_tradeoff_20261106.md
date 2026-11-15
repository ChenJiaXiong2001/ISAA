# p025 clean and anchor-loss tradeoff

This is a descriptive analysis of two 20-epoch, 4,096-training-sample
checkpoints, evaluated on the same 16,306 validation examples. The control
checkpoint SHA256 is `3e61964b164ac426fe1059c4ae0a6efcf6c3a0996c356cd7ffb16859bc44c02f`;
the p025/clean50 candidate is
`f6ce6a0a07ea320017a381f381129e7d6f9a820f3b4c8bcf9f0ff4c34e5dbb4a`.
Each model uses its own training-calibration gamma=1 accuracy weights. All
conditions use paired inputs and matching frame masks, and the same sample
order and labels across conditions are checked by the analysis script.

On clean inputs, the candidate corrects 1,261 control errors but introduces
1,400 errors, for a net decrease of 139/16,306 = 0.8524 percentage points.
The table below partitions exactly those 1,400 clean regressions by what
happens to the same samples under synthetic original-frame anchor loss.

| Anchor loss | Candidate correct, control wrong | Candidate wrong, control correct | Both correct | Both wrong |
| --- | ---: | ---: | ---: | ---: |
| 5% | 98 | 616 | 132 | 554 |
| 10% | 133 | 461 | 172 | 634 |

Only 98/1,400 (7.0%) clean regressions become candidate-only correct at 5%
anchor loss, and 133/1,400 (9.5%) do so at 10%. Conversely, of the 3,392
candidate-only correct examples under 5% loss, 2,261 were already correct
for both models on clean input. Under 10% loss, 2,562 of 3,705 candidate-only
correct examples were in that clean both-correct group. In this fixed set,
the aggregate robustness gain is therefore not mainly explained by the
specific examples that regressed on clean input.

Class-level changes are heterogeneous. Of 60 classes, 22 have a negative
clean net change and a positive 5%-loss net change, while 9 have negative
changes in both conditions. At 10%, the respective counts are 24 and 7.
For A030, the candidate has 23 fewer correct predictions on clean input but
121 and 145 more under 5% and 10% loss. For A010 it has 22, 39 and 30 fewer,
respectively; for A052 it has 20, 29 and 23 fewer. Class-level counts are
descriptive and do not imply a reliable deployment decision for any class.

The source paired CSV records class predictions and correctness only; it has
no logits, probabilities, or confidence values, so confidence dependence
cannot be assessed from these outputs. Clean groups are explicitly defined
using validation labels. Conditioning on those groups can create selection
effects, and these statistics do not establish an individual causal tradeoff
or provide an inference-time routing rule. The official validation set has
also been reused for exploration, and results from these partially trained
models do not establish the performance of a full-length training run.

The input data and complete per-class and per-subject results are in
`/home/xiong/Desktop/ISAA/outputs/precompression_anchor_p025_clean50_vs_control_20ep_anchor_pre_full_20261106/clean_robust_tradeoff_analysis.json`.
The original paired CSV and input hashes are in the same directory. The
analysis can be reproduced without GPU:

```bash
/home/xiong/venvs/isaa-cu126/bin/python -s /tmp/analyze_anchor_clean_robust_tradeoff_20261106.py \
  --evaluation-dir /home/xiong/Desktop/ISAA/outputs/precompression_anchor_p025_clean50_vs_control_20ep_anchor_pre_full_20261106 \
  --output /tmp/p025_clean_robust_tradeoff_recheck_20261106.json
```

The workspace source is `tools/analyze_anchor_clean_robust_tradeoff.py`.
