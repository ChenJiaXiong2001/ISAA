# Lagged gamma=1 training: fixed 20-epoch paired screen

The candidate did not meet the preregistered screen: clean own-weight gain
was +0.2392 percentage points (required >= +0.25), and precompression anchor
10% was -0.7973 points (required >= -0.25). Do not promote the candidate as
an improvement on this evidence. This compares 20-epoch models on a 4096-item
training prefix; it does not compare to or improve the full original 95.28%
model.

Both models were initialized from scratch, trained for 20 epochs using the
same training/validation prefix and parameters, and evaluated at last.pt using
identical inputs across all 16,306 validation examples from 20 subjects.
Source archives and split manifests were checked; their recorded split manifest
SHA256 was 5646a14e0ca992a54e16f9dcb3e6755d89bd30961144aec9bad4ce2f4e7590ff.
The CUDA Inductor checks, 20-epoch checkpoints, and two-sample paired integration
smoke check passed. The original face LocalSTBlock.forward eager boundary was
retained. Epoch-one candidate weights had 3 zeros and 11 ones; epoch-two
completed 117 finite loss/gradient updates (numbers 118 through 234).

| Condition | Control % | Candidate % | Own-weight difference pp | Subject bootstrap 95% CI pp | Common-weight difference pp |
|---|---:|---:|---:|---:|---:|
| Clean | 69.5388 | 69.7780 | +0.2392 | [-0.3915, +1.1109] | +0.5397 |
| Point 5% | 43.8121 | 46.1057 | +2.2936 | [+1.5646, +3.6301] | +3.3546 |
| Point 10% | 32.2765 | 34.5394 | +2.2630 | [+1.3428, +4.1963] | +3.1829 |
| Hand/face group 5% | 64.8412 | 66.5951 | +1.7540 | [+0.9619, +3.1326] | +2.2262 |
| Hand/face group 10% | 60.4931 | 62.1734 | +1.6804 | [+0.9070, +3.0728] | +2.3795 |
| Precompression anchor 5% | 54.8142 | 54.6547 | -0.1595 | [-0.6384, +0.6869] | +1.3185 |
| Precompression anchor 10% | 48.8287 | 48.0314 | -0.7973 | [-1.4987, +0.0351] | +0.7175 |

Own-weight comparisons use the gamma=1 class weights computed from each
checkpoint's calibration counts; common-weight comparisons use the control's
weights for both heads. The two learned tables differ in 41 of 60 classes,
with mean absolute difference 0.1015 and maximum difference 1.0. Their
calibration set contained only subject P025. The reversal at anchor 10%
indicates sensitivity to the weights, so the -0.7973 pp result does not by
itself establish that the training objective reduces anchor robustness. The
confidence intervals resample subjects and reflect the evaluated sample; a
single matched training pair does not separate initialization/training
variation from the intervention. Four stochastic-corruption comparisons are
exploratory, not independent confirmation.

An independent pass over the saved per-sample CSV reproduced each condition's
correct totals. Clean control/candidate totals were 11339/11378; for anchor 10%
the control, candidate-own and candidate-common totals were 7962/7832/8079.
The evaluator additionally reconstructed predictions from stored branch logits.

Control last.pt SHA256:
`8905eeabf263de20078fe85e4ba27809ca8c1a287defc6e62832aa985bd35a43`.
Candidate last.pt SHA256:
`daac65f9076f033679f5ff94fd39bef76b14ff0d569ef83731e8bb524b9fb3b4`.
Validation artifacts remain under the server path
`/home/xiong/Desktop/ISAA/experiments/training_route_mismatch_20261106/paired_validation_20261106/`:
`paired_samples.csv` SHA256 `2fa84fd7c200e8266366dbd8a4810e806f029a63675df31cfa48dc31d299a729`;
`paired_branch_logits.npz` SHA256 `0ff957cd295640056b4b9b4a8c1973642f26bf9bff96aec3f4b033c113cd4ca2`;
`summary.json` SHA256 `3049a5bee155a3696d3c8e738ce4ef7a0db142a8416a61c6ed501eae8c48a689`.
The complete calibration outputs, checks, scripts, and logs are retained in
the same experiment directory. No model or prior experiment was replaced.
