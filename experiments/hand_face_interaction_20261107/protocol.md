# HF temporal hand-face residual interaction: pre-registered short screen

## Hypothesis

The frozen epoch-83 audit leaves a 1.8582 percentage point oracle gap between
the current HF/BF fusion and choosing the correct head. Of the 465 samples
where both heads are wrong, 433 still contain the true class in at least one
head's top-5. A small cross-branch interaction residual may improve these
boundary rankings without relying on an uncalibrated confidence gate.

## Isolated implementation

Create a new scratch-only `FacePairFusion` variant. Keep the existing body,
hand and face encoders, both classifier heads, gamma=1 training-calibration
fusion, and three supervised losses unchanged. After the existing per-frame
`hand_t` and `face_t` pooling, project each with a bias-free linear layer to
32 channels, multiply the projected features elementwise, mask by
`hand_valid & face_valid`, and average over valid time/person positions. A
zero-initialized bias-free `Linear(32, 60)` maps the interaction to an HF
logit residual. Add this residual to the HF head before the inherited fusion.
The baseline paths are unchanged; no dynamic confidence gate or routing-table
change is introduced. A separate velocity ablation is deferred until this
interaction-only result is known.

## Fixed short screen

- NVIDIA A6000, CUDA/PyTorch environment used by the project.
- 8 epochs, 4096 maximum training samples, batch 32, warmup 1, seed 17.
- Same archive, split, fit/calibration/selection members, corruption masks,
  optimizer, schedule, and loss weights as the matched control.
- Candidate and control both start from random initialization. The candidate
  must not load the epoch-83 checkpoint or another candidate checkpoint.
- The run must use `torch.compile` with the verified Inductor backend. The
  interaction residual is compiled with the neural path; the inherited face
  `LocalSTBlock.forward` eager boundary remains unchanged. Compilation errors
  are fatal and cannot fall back to whole-model eager execution.

## Gate, fixed before training

Evaluate the paired epoch-8 `last.pt` checkpoints with gamma=1 weights computed
from each run's training calibration partition. Candidate-control must satisfy:

1. clean accuracy gain at least `+0.25pp`;
2. point and hand/face-group corruption at 5% and 10% each no worse than
   `-0.25pp`;
3. at least one corruption condition improves by `>=+1.00pp` and has a
   paired subject-bootstrap 95% interval whose lower endpoint is above zero.

Only a candidate passing all three rules can receive an independent short
recheck. The recheck uses the same protocol and a fresh random initialization;
the second initialization is a stability audit, not a separate seed study.
No 100-epoch run starts before the fixed gate and recheck pass.

## Required checks and artifacts

Run CPU tests for masking, zero residual initialization, finite gradients and
checkpoint variant rejection. Run real CUDA compiled forward/backward and two
optimizer updates. The saved configuration, model variant, compile scope and
checkpoint metadata must agree and must record the random-scratch origin.
Preserve all previous models, entries and outputs. Record paired samples,
branch logits, per-condition summaries, subject bootstrap intervals and SHA256
hashes under this experiment directory.

## Implementation record

The isolated draft is in `model.py`, with the process-local entry point in
`train.py`. `check_cuda_compile.py` is the mandatory synthetic CUDA check; it
requires a compiled forward/backward and two optimizer updates and records the
Inductor scope in `cuda_compile_smoke.json`. `evaluate.py` performs the paired
epoch-8 clean and corruption evaluation against the seed-17 matched control.
No candidate GPU run is started until the matched control has released the
GPU.
