# Pre-registered second-seed short screen (design only)

This file defines a second matched short run before any second-seed result is
read. No training is started by creating this protocol.

## Objective

Test whether the lagged gamma=1 soft fusion's body-face robustness survives a
new scratch initialization, while preserving the original hard-route control,
the original model, and all earlier runs. The candidate and control use the
same data members, optimizer schedule, compile settings, and seed. The only
intended intervention remains the candidate's previous-epoch gamma=1 soft
fusion during training and readout.

## Fixed run configuration

- Seed: `29` for both control and candidate.
- Epochs: `20`; training prefix: `4096`; batch size: `32`.
- Learning rate `0.025`, warmup `1`, warmup start `0.001`, cosine minimum
  `0.0001`, momentum `0.9`, weight decay `0.0004`, accumulation `1`.
- Calibration starts at epoch `1`; use the same fit/calibration/validation
  manifests and archive as the seed-17 paired screen.
- Start both models from random initialization. Do not load any prior model
  weights.
- `torch.compile`/Inductor remains mandatory with `compile_mode=default`;
  reject `--no-compile`, eager fallback, or a mismatched saved compile record.
- Save immutable outputs under new directories:
  `outputs/soft_route_matched_control_20ep4096_seed29_20261107/` and
  `outputs/soft_route_candidate_20ep4096_seed29_20261107/`.
- Freeze and evaluate `last.pt` at epoch 20. Do not select an epoch after
  examining validation results.

## Evaluation and weight sensitivity

Run same-input paired evaluation on the complete 20-subject validation
partition for clean, body-point dropout 5%/10%, hand/face-group dropout
5%/10%, and original-frame precompression anchor 5%/10%. Use each checkpoint's
gamma=1 training-calibration weights for the primary comparison. Before
looking at validation outcomes, also register a fixed transfer-weight
sensitivity: fuse candidate logits with the control checkpoint's calibration
weights and report it separately. Do not choose a weight table, blend factor,
or corruption-specific rule using validation labels.

## Promotion gates

The candidate must beat the matched control by at least `+0.25` percentage
points on clean, and no listed corruption condition may be below `-0.25` points
at the candidate's own weights. Report 20-subject bootstrap CIs and equal-
subject averages regardless of gate outcome. The fixed control-weight result
is sensitivity evidence only and does not change the primary gate.

Record corrected and regressed counts, per-subject differences, input/mask
hashes, checkpoint/source/config hashes, calibration counts, and compile
smoke evidence. Any protocol or manifest mismatch invalidates the screen and
must be retained as a failure artifact.

## Expected wall time

The seed-17 pair took about 20 minutes of GPU wall time including the two
20-epoch trainings and paired evaluation on the A6000. Reserve roughly
25 minutes for the seed-29 pair, plus at most 10 minutes for integrity checks
and report generation. These are planning estimates, not measured guarantees.
