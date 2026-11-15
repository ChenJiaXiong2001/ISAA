# Gamma=1 Soft Fusion Paired Check

The original screen's epoch metrics are the model's hard class routing, not the requested accuracy-proportional gamma=1 mixture. Both epoch-8 checkpoints were therefore reevaluated on the same ordered 4,096 validation samples with the saved training-subject calibration counts, fixed unit temperatures, and compiled CUDA inference. Neither checkpoint was retrained. The common validation digest is `7546414388a725274af69c6971840b8c285c396fd56f76f232f9a55e12617795` and the ordered calibration and validation manifests match.

| Training fusion loss | Hard routing Top-1 (training log) | Gamma=1 Top-1 | Gamma=1 Top-5 |
|---|---:|---:|---:|
| 1.5 | 57.1045% | 2,360/4,096 = 57.6172% | 89.1113% |
| 1.0 | 56.1279% | 2,346/4,096 = 57.2754% | 88.9160% |

For gamma=1, the 1.5 model corrects 370 samples that the control misses and misses 356 samples that the control gets right: net +14, or +0.3418 percentage points. The 20,000-replicate paired bootstrap 95% interval is [-0.9521, +1.6357] percentage points. This interval includes zero, and the 8-epoch experiment does not establish an improvement in the target mixture.

Each checkpoint uses its own saved calibration counts and therefore its own gamma=1 class weights, as prescribed by the method. Although the calibration samples and formula match, 43 of 60 HF weights differ, with mean absolute difference 0.1662 and maximum 1.0. The small calibration set and the two separately trained checkpoints limit attribution of the observed difference to the loss weight alone. The recorded 1.5 BF head differs by one correct sample between training-time and reevaluation, consistent with evaluation numerical variation; no exact cause was established here.

`soft_paired_samples.csv` records all 4,096 paired correctness outcomes. The original evaluator's `errors.csv` contains the union of samples misclassified by any head or soft mixture; missing samples are correctly classified by all three methods. `compare_soft.py` reconstructs the full list, verifies both reported soft correct totals and the ordered manifests, and generates `soft_paired_summary.json`. Complete evaluation artifacts and class weights are in the two `face_pair_loss_soft_eval*` directories; `split_manifest.json` is retained under each training variant.
