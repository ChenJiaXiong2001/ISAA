# Soft-route single-subject calibration screen

Use only the cached outputs of the two matched 20-epoch models and their single
P025 calibration subject. Calibration has 351 clean clips from one setup and
three cameras; most classes have six clips. These are neither independent
subjects nor independent scenes. Split strictly by repetition (R001 vs R002)
so all three cameras of one repetition stay in the same fold. Each fold builds
classwise head-correct counts on the other repetition and tests clean clips.

Fix before running the calibration screen: baseline gamma=1 ratios and two
symmetric pseudo-count candidates, alpha=1 and alpha=5, plus equal weights as
a negative/control endpoint. Candidate weight is (H+alpha)/(H+B+2*alpha).
Select alpha=1 or 5 only if its pooled out-of-fold clean correct count exceeds
the baseline by at least 3 AND the gain is nonnegative in both folds; when both
qualify, select the greater pooled gain, then lower NLL, then alpha=1. The
control model's own evaluation is descriptive, and selection uses the
soft-route model only. If neither qualifies, no candidate is selected and no
validation labels will be used to rescue or choose a candidate.

If one candidate qualifies, apply its fixed alpha to each model's saved
checkpoint counts on already captured same-input validation logits; report
both models with their own weight tables on the seven previously evaluated
conditions, paired changes per condition and subjects, no tuning on validation.
Historical evaluation has already used this validation set, so these results
are exploratory and cannot establish independent generalization. Head logits,
labels, members, checkpoint counts and cache identity must be checked; no GPU,
model training or old-output overwrite. Repetition splitting still shares a
single person and setup, and model checkpoints were trained with calibration
route feedback. This is not a new-person generalization test.
