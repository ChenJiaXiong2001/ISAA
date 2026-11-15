# Precompression anchor paired evaluation

This isolated evaluator compares two frozen `best-face-pair-hand-motion`
checkpoints under the same per-sample input tensors. For each validation member
it applies a deterministic, member-specific dropout to valid shoulder/hip
anchors (RTMW indices 5, 6, 11 and 12) on the original frames, then performs
the normal confidence-weighted temporal compression. Both checkpoints receive
the same clean or corrupted tensor in each batch and retain their own training
calibration counts for gamma=1 HF/BF fusion.

It evaluates the complete 16,306-member validation set at 5% and 10%, records
member and tensor hashes, and writes `paired_samples.csv`, `summary.json`, and
subject-cluster bootstrap input for `summarize.py`. The anchor condition is a
synthetic precompression missingness test; it must not be mixed with ordinary
post-compression joint dropout.

CPU unit checks:

```text
python test_cpu.py
```

GPU command (run only when the server GPU is free):

```text
python -u evaluate.py --control CONTROL.pt --candidate CANDIDATE.pt \
  --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
  --output-dir OUTPUT --batch-size 256 --num-workers 8
python summarize.py OUTPUT/paired_samples.csv CONTROL_DIR/split_manifest.json \
  OUTPUT/stratified_subject_cluster_bootstrap.json
```
