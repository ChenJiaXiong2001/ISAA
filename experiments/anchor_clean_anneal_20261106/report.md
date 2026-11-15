# Anchor clean annealing: 20-epoch screen

## Design and verification

The candidate uses original-frame anchor dropout at p=0.025 on selected
samples. Clean sample probability is 0.50 for epochs 1-10, then increases by
0.05 each epoch to 1.00 at epoch 20. Both controls and the candidate use the
hand-motion HF/BF model, scratch initialization, 4096 training samples,
20 epochs, batch 32 and identical learning-rate and calibration settings.
`last.pt` at epoch 20 was fixed in advance; no best-checkpoint selection was
used for this comparison.

The A6000 run completed in 891.6 seconds. CUDA Inductor dry-run completed a
forward, backward, and optimizer step; the first real training batch also
completed with finite loss and gradient. The saved checkpoint records epoch 20,
2340 updates, scratch initialization, mandatory Inductor with no whole-model
eager fallback, and the curriculum. The inherited face `LocalSTBlock.forward`
eager boundary is unchanged. Logs show all 20 scheduled probabilities.
Persistent spawn workers were tested for visibility of epoch changes. Existing
control configs, split manifests, and 117-batch warmup were checked. The two
older and new runs' archived common source files are byte-identical.

## Full validation paired results

Each row compares the same 16,306 inputs with frozen training-calibration
gamma=1 weights. Intervals are sample-weighted 20-subject cluster bootstrap
95% intervals; all values are percentage-point changes, candidate minus control.
Each model is a short-run checkpoint, not the 95.28% full-training model.

| Condition | vs clean control | vs fixed p025/clean50 |
| --- | ---: | ---: |
| clean, anchor-pre evaluator | -0.472 [-1.170, +0.163] | +0.386 [-0.077, +0.793] |
| original-frame anchor 5% | +12.400 [+10.916, +13.222] | -2.275 [-3.053, -1.613] |
| original-frame anchor 10% | +13.541 [+12.287, +14.229] | -3.802 [-4.555, -2.635] |
| clean, standard evaluator | -0.472 [-1.170, +0.163] | +0.399 [-0.063, +0.799] |
| body-point 5% | +8.758 [+7.865, +9.808] | -9.567 [-10.407, -8.546] |
| body-point 10% | +8.175 [+7.376, +9.277] | -15.608 [-16.851, -14.517] |
| hand/face group 5% | -0.190 [-1.454, +0.491] | -1.809 [-2.556, -1.145] |
| hand/face group 10% | -0.760 [-1.368, -0.344] | -4.379 [-5.135, -3.454] |

The clean point estimate misses the predeclared -0.25pp threshold, while
group 10% also declines relative to clean control. Relative to fixed p025,
clean improvement is uncertain and missing-data accuracy is lower. This
screen does not support promotion to full training.

Although the first ten epochs use the same prescribed augmentation as the
fixed control, their validation trajectories differ. Their first-batch losses
were 15.07069 and 15.07073; the logs do not establish that all training inputs,
parameters or CUDA calculations were identical. Thus the observed difference
cannot be attributed solely to the curriculum. Validation has also been reused
in earlier exploration, and simulated missing data are not real occlusion.

## Evidence

- Training and console: `/home/xiong/Desktop/ISAA/outputs/anchor_clean_anneal_20ep4096_20261106/20261106-204945-3294cc8e/` and `/home/xiong/Desktop/ISAA/outputs/anchor_clean_anneal_20ep4096_20261106.console.log`.
- Anchor-pre results: `/home/xiong/Desktop/ISAA/outputs/anchor_anneal_vs_clean_anchor_pre_20261106/` and `/home/xiong/Desktop/ISAA/outputs/anchor_anneal_vs_fixed_anchor_pre_20261106/`.
- Standard results: `/home/xiong/Desktop/ISAA/outputs/anchor_anneal_vs_clean_standard_20261106/` and `/home/xiong/Desktop/ISAA/outputs/anchor_anneal_vs_fixed_standard_20261106/`.
- Each result directory contains paired samples, summary, and subject-cluster bootstrap. Cached comparisons checked member, label, corruption-input and frame-mask hashes; separate reruns can differ by a few predictions.
