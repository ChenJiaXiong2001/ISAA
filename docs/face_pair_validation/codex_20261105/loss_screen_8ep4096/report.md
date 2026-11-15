# Paired fusion-loss short screen

Both runs use exactly the same `best-face-pair` scratch-training protocol: seed 17 as a fixed control value, 8 epochs, 4,096 samples per split, fit/calibration/validation counts 3,745/351/4,096, batch size 32, warmup 1, and mandatory CUDA `torch.compile` with the Inductor backend. The only changed value is the fusion-loss weight.

The following `val_acc` values are the training entry's per-class **hard-routing** results, not the target gamma=1 accuracy-proportional probability fusion.

| Epoch | fusion=1.5 hard route | fusion=1.0 hard route | Difference (pp) |
|---:|---:|---:|---:|
| 1 | 10.4492% | 10.6445% | -0.1953 |
| 2 | 18.6523% | 21.5332% | -2.8809 |
| 3 | 24.8535% | 26.0742% | -1.2207 |
| 4 | 31.8115% | 32.0068% | -0.1953 |
| 5 | 46.6553% | 41.4307% | +5.2246 |
| 6 | 48.2666% | 47.6318% | +0.6348 |
| 7 | 54.5898% | 54.1504% | +0.4395 |
| 8 | **57.1045%** | 56.1279% | **+0.9766** |

Both runs completed with finite metrics and the required compile configuration. The fusion=1.5 run reached 57.1045% hard-route accuracy at epoch 8; the strict fusion=1.0 control reached 56.1279%, a +0.9766 percentage-point short-screen difference. The hard-route advantage appears after epoch 4 rather than at initialization.

The two epoch-8 checkpoints were separately reevaluated using the target gamma=1 accuracy-proportional probability fusion. Fusion=1.5 obtained **2,360/4,096 = 57.6172%**; the default control obtained **2,346/4,096 = 57.2754%**, a **+0.3418 percentage-point** difference. It corrected 370 control errors while introducing 356 errors. The 20,000-replicate paired bootstrap 95% interval is **-0.9521 to +1.6357 percentage points**, so the short screen does not establish an improvement. The ordered validation and calibration members match, but each checkpoint uses its own saved calibration counts: 43 of 60 HF weights differ, with mean absolute difference 0.1662 and maximum 1.0. Full paired correctness, reproducible calculations and the weight comparison are in `soft_paired_samples.csv`, `compare_soft.py` and `soft_paired_report.md`.

The gamma=1 difference is a weak short-training signal, not a full-dataset generalization result. The separate 100-epoch fusion=1.5 run was stopped during epoch 2 after this interval showed insufficient evidence to justify the much longer run. Its epoch-1 `best.pt`, `last.pt`, configuration, status and logs remain on the server at `/home/xiong/Desktop/ISAA/outputs/face_pair_fusion15_100_20261105/20261105-173053-4b4d64fa`; `status.json` still says `running` because the process received SIGTERM. No full-training accuracy or corruption claim is made. The large early HF logit scale in the fusion=1.5 run (absolute maximum 864.6 at epoch 1 of the short screen, about 200 at epoch 8) remains a numerical quantity to monitor even though the run stayed finite.

Evidence is in `fusion_h050_f150/` and `control_h050_f100/` (`config.json`, `status.json`, `epochs.csv`, `full.log`), with machine-readable values in `summary.json`.
