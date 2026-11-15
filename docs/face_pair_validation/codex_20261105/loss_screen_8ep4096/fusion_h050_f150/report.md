# Fusion-loss weight 1.5 short screen

## Protocol

- Model: `best-face-pair`, random scratch initialization, no checkpoint.
- A6000 CUDA run with mandatory `torch.compile`/Inductor (`compile=true`, `required=true`).
- `pair_head_loss_weight=0.5`, `pair_fusion_loss_weight=1.5`.
- 8 epochs, `max_samples=4096` per split, batch/test batch 32, warmup 1, seed 17 as a fixed control value.
- Split counts were fit 3,745, calibration 351, validation 4,096. The run completed in 694.6 seconds.

## Result

The best and final validation hard-routing Top-1 was **57.1045%** at epoch 8 (2,339/4,096). Validation hard-routing Top-1 by epoch was 10.45%, 18.65%, 24.85%, 31.81%, 46.66%, 48.27%, 54.59%, and 57.10%. HF/BF single-head accuracy at epoch 8 was 44.09%/47.12%. These figures are not the target gamma=1 accuracy-proportional probability fusion; the gamma=1 checkpoint reevaluation was 2,360/4,096 = 57.6172%.

The run completed without a non-finite failure. The saved configuration records compiled scope, random scratch initialization, and the required Inductor backend. HF logit absolute maxima were high early (864.6 at epoch 1) and remained about 200 at epoch 8; BF was 42.1 at epoch 1 and 43.1 at epoch 8. This is a numerical-risk signal to monitor, not evidence of a failed run because the fused loss and gradients remained finite.

## Interpretation

The matched 8-epoch/4,096-sample control is documented in the parent directory. Its hard-routing result is 56.1279% and its gamma=1 result is 57.2754%; the gamma=1 short-screen difference is +0.3418 percentage points with paired 95% interval -0.9521 to +1.6357 percentage points. The separate full-training candidate was stopped during epoch 2 and its files were retained.

Evidence files: `config.json`, `status.json`, `epochs.csv`, and `full.log` in this directory. Remote run: `/home/xiong/Desktop/ISAA/outputs/face_pair_loss_screen_8ep4096_20261105/20261105-170247-f9145a1e`.
