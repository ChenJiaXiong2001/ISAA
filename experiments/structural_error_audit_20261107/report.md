# Frozen HF/BF error-structure audit (2026-11-07)

This diagnostic reuses the epoch-83 frozen cache from
`face_pair_stable_20261103` (3,191 calibration and 16,306 validation
samples). Class priors are computed from calibration labels only; validation
labels are used only to describe error strata. No candidate is selected from
this audit and no model is modified.

The original gamma=1 fusion is correct on 15,538/16,306 samples (95.2901%).
The two heads are both correct on 13,716 samples, HF-only on 715, BF-only on
1,410, and both wrong on 465. An oracle that chooses whichever head is
correct reaches 15,841 samples, an upper bound of +1.8582 percentage points.
The existing fusion nevertheless flips 123 HF-only and 186 BF-only samples
to an incorrect result.

The remaining 465 both-wrong samples are usually close ranking failures:
the true class is in at least one head's top-5 for 433 (93.1%), in both
heads' top-5 for 309 (66.5%), and in neither top-5 for only 32 (6.9%). The
fused result rescues only 6 of these samples. This supports testing changes
that improve class ordering or cross-branch representation, rather than
assuming missing information is the dominant cause.

Raw confidence gating is not a free fix. Routing to the head with larger
max-probability gives 94.8853% (−0.4048pp versus gamma=1), and larger top-1
margin gives 94.9221% (−0.3680pp). Among fusion flips, the wrong branch is
often more confident: in HF-only flips, HF confidence exceeds BF in only
10/123 cases; in BF-only flips, HF confidence exceeds BF in 165/186 cases.

Focus classes show the same pattern: A011/A012/A029 have 41/5/39 both-wrong
samples; fused errors with the true class in either head top-5 are 61/15/48.
The largest confusion pairs are A011→A029 (32), A011→A012 (22), and
A029→A012 (22). These are diagnostic evidence for hand-centric interaction
or temporal representation experiments, not proof that either feature will
improve trained accuracy.

Machine-readable outputs are `error_structure_summary.json`,
`error_strata_summary.json`, and `confidence_summary.json`.
