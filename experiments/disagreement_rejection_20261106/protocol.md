# Fixed disagreement-aware rejection screen

Frozen epoch-83 HF/BF cache, original classwise gamma=1 weights calculated from
the checkpoint's saved routing counts, and the already frozen 50/50 normalized
arithmetic/geometric fusion. Let `confidence` be the fused maximum probability,
and `JSD` the Jensen-Shannon divergence between unit-temperature HF and BF
probability distributions. The one fixed experimental score is
`confidence * (1 - JSD / ln(2))`; higher scores are accepted. This is a fixed
rule, not a parameter fit or a claim that it is optimal.

Before examining validation results, set the score threshold to the 5th and
10th percentile of this score on the 3191 calibration examples. Compare with
the corresponding baseline fused-confidence thresholds calculated from the
same calibration cache using checkpoint weights.
Report coverage and selective accuracy, plus error and correct counts sent to
review. Calibration includes the same two training subjects used to estimate
the classwise fusion weights. The initial cache-derived version is preserved in
`result/`; it differed from checkpoint weights in two classes. This correction
was made before re-running the comparison, not by selecting on validation
outcomes. The reason for the count discrepancy is unconfirmed.

For matched-coverage diagnosis, the baseline rejects the number of validation
examples determined by its fixed calibration threshold. Reject exactly that
many validation examples with the smallest candidate score, breaking ties by
manifest order. No validation labels enter this matching operation, but it
requires a batch of validation-like unlabeled examples and is not a fixed
per-clip threshold. Report paired error capture differences, review swaps,
and 20-subject cluster bootstrap intervals and subject/class effects. Test
the two predeclared review budgets; do not optimize weights or choose a rule
on validation outcomes. CPU cached logits only, no GPU or training.

The validation set has been repeatedly reused, so this is exploration, not
independent confirmation. Subject bootstrap does not account for the estimated
threshold or research selection. Rejection requires a real review path and
does not change overall model accuracy. Synthetic occlusion cannot be tested
from existing predictions because the saved output has no probabilities.
