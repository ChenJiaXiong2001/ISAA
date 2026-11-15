# Fixed p025/clean75 short screen

The candidate did not meet the predeclared clean point-estimate criterion and
does not support promotion to a full training run. Against the same-budget
clean control, clean accuracy was 69.6492% versus 70.4035%, a difference of
-0.7543 percentage points (20-subject bootstrap 95% interval [-1.3746,
+0.2386]). These are 20-epoch, 4096-training-sample results, not comparisons
with the fully trained 95.28% model. The interval includes positive values;
it does not establish the true clean difference's sign.

## Matched full-validation comparison

Each condition uses all 16,306 validation samples, with the same tensors fed to
both checkpoints in one paired forward and each model's own frozen gamma=1
training-only calibration weights. All differences are candidate minus clean
control, in percentage points; intervals resample the 20 subjects with sample
weighting, 20,000 draws.

| Condition | Control | Candidate | Difference | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| clean | 70.4035% | 69.6492% | -0.7543 | [-1.3746, +0.2386] |
| anchor before compression 5% | 53.8636% | 68.1528% | +14.2892 | [+12.6831, +15.1981] |
| anchor before compression 10% | 48.8532% | 65.7611% | +16.9079 | [+14.9014, +17.9028] |
| body point 5% | 48.1786% | 60.5667% | +12.3881 | [+11.1893, +14.1390] |
| body point 10% | 38.1639% | 52.1403% | +13.9765 | [+12.8679, +15.8295] |
| hand/face group 5% | 67.3801% | 68.6373% | +1.2572 | [+0.5248, +2.4673] |
| hand/face group 10% | 64.6817% | 67.0428% | +2.3611 | [+1.5301, +3.4599] |

Using the clean control's weights for both models gives clean -0.9138 pp
[-1.5721,+0.2833], anchor 5/10% +14.4241/+17.0857 pp, point 5/10%
+12.6211/+14.3873 pp, and group 5/10% +1.0794/+2.3427 pp. The latter group
10% interval is [+1.5969,+3.4924]. Thirty-three of 60 class HF weights differ
between checkpoints, with mean absolute difference 0.0630. Weights were
calibrated on only subject P025; the shared-weight analysis does not isolate
augmentation effects from training variation or calibration uncertainty.

## Earlier p025/clean50 candidate

For raw-anchor conditions, input and frame-mask digests, ordered members,
labels, control checkpoint and weight hashes were checked before reusing the
older p025/clean50 predictions. Relative to that candidate, clean is +0.1043
pp [-0.4403,+1.0559], anchor 5% -0.3925 pp [-0.9623,+0.2738], and anchor
10% -0.4293 pp [-1.3272,+0.3356]. Both candidates using the older candidate's
weights give clean -0.2514 pp [-0.8124,+0.7809], anchor 5% -0.6133 pp
[-1.1197,-0.0246] and anchor 10% -0.5090 pp [-1.1847,+0.2743]. Thirty-nine
of 60 weights differ, mean absolute difference 0.1091. These candidates ran in
separate forwards: the repeated clean-control predictions differ on 2, 0 and
1 samples for the three conditions, respectively. Those differences indicate
run-level numerical variation and cannot be attributed to augmentation alone.

The earlier standard missing-input results cannot be used for a direct
paired candidate comparison: all four point/group input and validity-mask
hashes differ, even though condition names and recorded seeds match. Clean
hashes do match. The older evaluator's file timestamp predates its result,
but the exact earlier code snapshot and worker count were not recorded, so the
cause is undetermined. Worker RNG and configuration are possible contributors,
not established causes. Same seed does not establish identical inputs. The
guard rejected the reuse; no paired statistics were calculated for those four
conditions. The present candidate versus clean control remains paired because
both received the same tensor within each batch.

## Run and evidence

The model trained from scratch with mandatory Inductor, no whole-model eager
fallback, 20 epochs, 2340 updates, p=0.025 and clean probability 0.75. CUDA
dry-run completed a finite forward, backward and optimizer update. The
existing face LocalSTBlock eager boundary remains. Training took 1243.6 seconds
including first compilation; epoch-20 `last.pt` was selected in advance even
though epoch 18 was the best logged hard-routing checkpoint.

Training: `/home/xiong/Desktop/ISAA/outputs/anchor_p025_clean75_20ep4096_20261106/20261106-220642-5dfe12d3/`.
Anchor and standard paired outputs: `/home/xiong/Desktop/ISAA/outputs/anchor_p025_clean75_vs_clean_anchor_pre_20261106/` and `/home/xiong/Desktop/ISAA/outputs/anchor_p025_clean75_vs_clean_standard_20261106/`.
Each contains predictions, model and input hashes, bootstrap, branch logits,
and shared-weight sensitivity. The raw-anchor directory also contains
`vs_p025_clean50_reused.json`. Full per-class weight differences are in the
JSON sensitivity files. The protocol and pre-training threshold correction are
in `protocol.md`.

The validation set has been used repeatedly during research; subject-cluster
intervals do not correct model-selection bias. Corruptions are synthetic and
do not demonstrate real-camera performance. P007 supplies 5,688/16,306 samples,
so sample-weighted aggregate results are sensitive to its outcomes.
