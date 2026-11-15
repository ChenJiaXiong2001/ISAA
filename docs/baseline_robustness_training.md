# Baseline Robustness Training

This scratch run uses the canonical pooled 100-frame baseline with 20% training dropout. Body points 0:23 are masked independently per frame and person. The 40 finger points form ten four-point groups and the 68 face points follow the existing six face-token groups; each group is masked together with 20% probability per frame. Hand roots 91 and 112 are preserved. Validation data remains unaugmented. Coordinates and confidence scores are masked before the baseline's derived torso-relative and cross-hand distance/direction features are built. These are independent probabilities, not a hard cap on the fraction hidden in each sample.

The run lasts 150 epochs with batch cosine over the full 150 epochs and the original five-epoch warmup. The baseline architecture, 100-frame confidence-weighted compression, batch size, optimizer settings, seed, random initialization, and required `torch.compile` configuration are inherited from `train_best_weighted_100_cosine.py`. The dedicated entry writes to `outputs/best_weighted_100_robustness_ntu60_xsub/` and does not load baseline weights. Earlier experiment outputs are preserved.

Start on the CUDA server with:

```bash
bash tools/train_baseline_robustness_server.sh --device cuda
```

The compile mode follows the canonical baseline (`reduce-overhead`); compilation failures are fatal. Saved run configuration records the joint-dropout probability. For robustness evaluation, compare the selected checkpoint on clean validation data and separately on predefined joint-dropout corruption levels. Do not augment validation data during checkpoint selection.
