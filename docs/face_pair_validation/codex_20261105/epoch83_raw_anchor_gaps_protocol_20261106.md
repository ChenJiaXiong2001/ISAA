# Epoch 83: fixed max5 under original-frame anchor gaps

This is an exploratory, fixed-rule evaluation of checkpoint
`best_face_pair_stable_100_20261103/20261103-052121-a73ca490/best.pt`,
SHA256 `aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`.
Its historical full-validation gamma=1 accuracy was 15,536/16,306 = 95.2778%.
HF/BF class weights come from the checkpoint's training-subject calibration
counts, without using validation labels to fit new weights.

On each original sequence, independently sample starts for each of four
shoulder/hip anchor joints and each person, then hide runs of 2, 5, or 10
frames. The start probabilities are respectively 0.025, 0.01, and 0.005;
overlap and truncation mean realized run lengths need not equal their nominal
values. A member-specific deterministic mask is generated with seed 1729.
Each damaged original sequence is processed by the usual confidence-weighted
temporal compression. The unchanged compressed input and a repaired copy are
evaluated on the same checkpoint. Repair always uses the same max5 rule:
interpolate score-zero shoulder/hip anchors over at most five compressed
frames, between valid endpoints, without crossing an invalid frame. It does
not receive the synthetic mask or nominal gap length. The identical pair is
also evaluated for the clean input.

The two scripts are `tools/evaluate_face_pair_raw_anchor_gaps.py` and
`tools/summarize_face_pair_raw_anchor_gaps.py`. Their server copies have
SHA256 `4ebbd770c2089b04e19578dff1d020dce16f6370f4a1e776c6a1625e6ed18415`
and `70d6182a1560d4e29b8fb507b136350b2662eae01dd95a2814e423af7130822b`,
respectively. Training is not performed by these scripts. Evaluation uses
frozen CUDA eager inference and checkpoint TF32 settings.

## Training-side calibration

The complete calibration partition contains 3,191 samples from P009 and P018.
The fixed procedure produced the following repair-minus-unmodified results:

| Condition | Difference (pp) | Corrected | New errors |
| --- | ---: | ---: | ---: |
| clean | +0.0313 | 1 | 0 |
| original-frame gap 2 | +3.8233 | 159 | 37 |
| original-frame gap 5 | +2.9144 | 122 | 29 |
| original-frame gap 10 | +0.0940 | 4 | 1 |

Data and paired predictions are in
`/home/xiong/Desktop/ISAA/outputs/epoch83_raw_anchor_gaps_max5_calibration_20261106/`.
These two subjects also supplied the checkpoint's fusion calibration counts;
their cluster-bootstrap intervals are not a reliable estimate of variation
across new subjects. The nominal gap conditions have distinct mask draws and
do not represent a controlled comparison of equal actual missingness.

## Full validation command

The commands below run only after the GPU is free. The complete official XSub
validation partition is evaluated in manifest order; `--max-samples 0` is the
default. An existing output directory causes an error instead of replacement.

```bash
env PYTHONPATH=/home/xiong/Desktop/ISAA/experiments/face_pair_handface_geometry_20261105/code \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/xiong/venvs/isaa-cu126/bin/python -s -u /tmp/evaluate_face_pair_raw_anchor_gaps_20261106.py \
  --checkpoint /home/xiong/Desktop/ISAA/outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490/best.pt \
  --manifest /home/xiong/Desktop/ISAA/outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490/split_manifest.json \
  --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
  --output-dir /home/xiong/Desktop/ISAA/outputs/epoch83_raw_anchor_gaps_max5_validation_20261106 \
  --partition validation --batch-size 32 --num-workers 4 --device cuda

env PYTHONPATH=/home/xiong/Desktop/ISAA/experiments/face_pair_handface_geometry_20261105/code \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/xiong/venvs/isaa-cu126/bin/python -s /tmp/summarize_face_pair_raw_anchor_gaps_20261106.py \
  --evaluation-dir /home/xiong/Desktop/ISAA/outputs/epoch83_raw_anchor_gaps_max5_validation_20261106 \
  --output /home/xiong/Desktop/ISAA/outputs/epoch83_raw_anchor_gaps_max5_validation_20261106/subject_bootstrap.json
```

The evaluator saves checkpoint, ordered-member, clean/condition input, and
frame-mask hashes, per-sample predictions and correction/regression counts.
The summary resamples 20 complete subjects 20,000 times. The official
validation set has already been used for historical model and repair
exploration, so this is not an independent or prospectively registered test.
Synthetic anchor gaps do not establish performance for real detector errors.
