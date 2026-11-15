# Fixed p025/clean75 short training screen

Hypothesis: exposing only 25% of training examples to p=0.025 independent
original-frame shoulder/hip anchor dropout may reduce the clean accuracy cost
seen with p025/clean50 while retaining some missing-anchor robustness. This is
an exploratory hypothesis, not a claim that it must work.

Train the existing hand-motion HF/BF architecture from scratch with mandatory
torch.compile/Inductor, 20 epochs, 4096 training samples, batch 32, seed 17,
warmup 1 and the existing matched optimizer, learning-rate schedule, and
training-only calibration. The only intended training change from the clean
control is p=0.025 and clean sample probability 0.75. Compare the preselected
epoch-20 `last.pt` against both the clean and p025/clean50 checkpoints on the
same full validation input tensors, with each checkpoint's frozen gamma=1
training-calibration weights. Check manifests, configurations, checkpoint
digests and input hashes. Before the run, execute an actual CUDA compiled
forward/backward/optimizer dry-run. The existing face LocalSTBlock eager
boundary remains; no whole-model eager fallback is allowed.

The fixed screening rule versus the clean control is clean difference at least
-0.25 percentage points with a subject-cluster 95% interval including zero,
and at least +5 percentage points with positive interval lower bounds for both
5% and 10% original-frame anchor dropout. Report differences versus the fixed
p025/clean50 control and body-point and hand/face-group dropout at 5%/10%,
including clean-relative differences. The screening rule does not establish
non-inferiority or justify a full run by itself. Simulated corruption and this
repeatedly used validation set are not an independent real-world test.

The training entry is `train_best_face_pair_hand_motion_100.py`, with
`--archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip --device cuda
--compile --compile-mode default --epochs 20 --warmup-epochs 1 --num-workers 8
--seed 17 --max-samples 4096 --log-interval 20 --pair-calibration-start-epoch 1
--precompression-anchor-dropout-probability 0.025
--precompression-anchor-clean-sample-probability 0.75` and a unique save dir.
Add `--dry-run` for the CUDA compile smoke test. Full evaluation uses
`evaluate.py --condition-set anchor-pre` and `--condition-set standard`; each
is run with both controls and a new output directory. Bootstrap is by subject
(20,000 resamples, sample-weighted) and does not correct selection bias.
