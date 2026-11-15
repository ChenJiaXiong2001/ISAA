# BF temporal interaction experiment

This standalone model adds a masked, low-rank per-frame body/face product and
zero-initialized residual BF logits to the existing `best-face-pair` model.
Each body and face projection starts randomly; the residual readout starts at
zero, so copied common weights produce the baseline outputs exactly. The
readout receives gradients on the first step; projections receive gradients
after the readout moves away from zero. Training itself never loads weights.

The original model, entry, and experiment outputs remain untouched. The new
entry calls the existing scratch-only entry and mandatory Inductor compiler;
the new interaction is compiled alongside the original encoders, pooling,
heads, and fusion. The inherited face `LocalSTBlock.forward` eager boundary is
unchanged. Configuration records this architecture and compilation scope.
The isolated entry registers the distinct model variant in its own process;
the preset, parser, model configuration, and saved checkpoint all identify
`best-face-pair-body-face-interaction`. Checkpoint consumers must explicitly
instantiate this class and check `checkpoint["architecture"]`. Existing
strict base-model loaders must reject it, not silently drop its parameters.

Run from the ISAA root after placing this folder inside `experiments/`:

```bash
python experiments/body_face_interaction_20261106/test_cpu.py
python experiments/body_face_interaction_20261106/check_cuda_compile.py
python experiments/body_face_interaction_20261106/train.py --device cuda --compile --compile-mode default \
  --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
  --epochs 8 --warmup-epochs 1 --batch-size 32 --test-batch-size 32 --num-workers 8 \
  --seed 17 --max-samples 4096 --pair-calibration-start-epoch 1 \
  --pair-head-loss-weight 0.5 --pair-fusion-loss-weight 1.0 \
  --save-dir outputs/face_pair_body_face_interaction_8ep4096
```

The CUDA smoke test compiles the new interaction with `fullgraph=True`, then
uses the project Inductor compilation entry for two complete forward/backward
steps. The GPU test and training must not run concurrently with other trials.
For comparison, use the identical 8-epoch control's data manifests and
predeclare the epoch-8 checkpoint, gamma=1 weights from each training-only
calibration partition, full clean/occlusion evaluation, and subject-cluster
paired intervals. Separate the 4096 epoch-selection samples from the remaining
validation samples; overlapping subjects and prior reuse still limit claims.

## Frozen paired evaluation

Run only after epoch 8 has been saved and the GPU is free. The evaluator fails
unless both `last.pt` checkpoints are epoch 8 and their data, schedule, loss,
and fit/calibration/validation member lists match. Both models receive the
same tensors in each batch; `summary.json` records their input hashes, model
hashes and training-calibration-derived gamma=1 weights. Five conditions use
the same seeds as the existing clean/point/group evaluators.

```bash
python experiments/body_face_interaction_20261106/test_evaluate_cpu.py
python -u experiments/body_face_interaction_20261106/evaluate.py \
  --control outputs/face_pair_loss_control_8ep4096_20261105/20261105-171736-c1c4112c/last.pt \
  --candidate outputs/face_pair_body_face_interaction_8ep4096_20261106/20261106-150107-3305aefd/last.pt \
  --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
  --output-dir outputs/face_pair_body_face_interaction_vs_control_full_20261106
python experiments/body_face_interaction_20261106/summarize.py \
  outputs/face_pair_body_face_interaction_vs_control_full_20261106/paired_samples.csv \
  outputs/face_pair_loss_control_8ep4096_20261105/20261105-171736-c1c4112c/split_manifest.json \
  outputs/face_pair_body_face_interaction_vs_control_full_20261106/stratified_subject_cluster_bootstrap.json
```

The evaluation directory must not already exist. `summarize.py` reports
sample-weighted subject-cluster confidence intervals and corruption DiD for
the full validation set, the epoch-selection prefix, and the remaining
samples. Its `remaining` stratum is not a held-out-subject or external test.
