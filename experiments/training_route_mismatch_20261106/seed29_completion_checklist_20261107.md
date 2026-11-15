# Seed-29 completion checklist (read-only post-run)

Do not start a second process from this checklist. Run checks only after both
seed-29 processes exit and the paired evaluator has produced its artifacts.

## Process and logs

- Confirm the control and candidate PIDs have exited with status 0; retain
  `run_seed29_20261107.log`, `control_train_seed29_20261107.log`, and the
  candidate log.
- Search all three logs for `ERROR`, `Traceback`, `nonfinite`, `eager
  fallback`, and `--no-compile`. A match invalidates the screen until
  investigated.
- Confirm each log records `torch.compile mandatory`, `compile_mode=default`,
  the expected compile scope, and a successful CUDA compile forward/backward
  smoke before epoch 1.

## Checkpoints and saved configuration

- Each new run contains `last.pt`, `best.pt`, `config.json`, `source.zip`, and
  `split_manifest.json`; never substitute `best.pt` for the registered epoch
  20 `last.pt`.
- Load both checkpoints with `weights_only=False` and verify `epoch == 20`,
  `args.seed == 29`, `args.epochs == 20`, `max_samples == 4096`, batch size
  `32`, `lr == 0.025`, warmup `1`, calibration start `1`, and all optimizer
  values in the seed-29 protocol.
- Verify `checkpoint.compile.enabled == true`, mode `default`, and the
  architecture IDs distinguish the hard control from the lagged soft
  candidate. Reject missing or contradictory compile metadata.
- Verify the candidate has a recorded previous-epoch gamma=1 effective route
  table and that the hard-routing diagnostic buffer is not used by its forward
  path.

## Source and split integrity

- Hash each `source.zip`, compare the shared baseline Python files byte for
  byte, and confirm the candidate archive contains only the registered
  soft-route experiment additions.
- Compare control/candidate `split_manifest.json` members and hashes. Require
  disjoint fit/calibration/validation members, identical manifests between
  runs, and `official_validation_used_for_routing == false`.
- Check the saved routing counters sum to the calibration member count and
  match the recorded calibration labels; retain any count mismatch.

## Paired evaluation

- Run the CUDA paired evaluator only with the two new epoch-20 `last.pt`
  files, the shared archive, and `partition=validation`. Confirm the evaluator
  refuses mismatched source, manifest, architecture, compile, or checkpoint
  arguments.
- Require `paired_samples.csv`, `paired_branch_logits.npz`, and `summary.json`;
  verify sample order, labels, logits reconstruction, finite fused outputs,
  and equal input/frame-mask hashes for control and candidate in every
  condition.
- Record own-weight and fixed control-weight comparisons for clean, point 5/10,
  group 5/10, and precompression anchor 5/10. Report corrected/regressed
  counts, 20-subject bootstrap CIs, and equal-subject means.
- Apply the pre-registered primary gate only to candidate own weights:
  clean at least `+0.25` percentage points and no listed corruption below
  `-0.25`. Fixed control-weight results remain transfer-weight sensitivity.

Keep every failed check and all generated logs; do not overwrite seed-17 or
the original epoch-83 artifacts.
