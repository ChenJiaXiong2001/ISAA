# Grouped dropout: paired early corruption screen

## Result

No reliable improvement is established. On the same 1,024 validation examples, grouped-dropout training has lower clean Top-1 (12.8906% vs 13.5742%). Under 10% group corruption it has one more correct prediction (13.0859% vs 12.9883%); 48 formerly wrong cases become correct and 47 formerly correct cases become wrong. These models are only four epochs into training and their accuracy is low, so the screen cannot predict fully trained performance or justify a long run by itself.

| Input condition | Clean training Top-1 | Grouped 0.1 training Top-1 | Grouped corrects / regresses | Clean training Top-5 | Grouped training Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 13.5742% | 12.8906% | 48 / 55 | 45.8008% | 44.2383% |
| Body point dropout 5% | 14.2578% | 12.8906% | 46 / 60 | 45.0195% | 45.4102% |
| Body point dropout 10% | 13.4766% | 12.1094% | 50 / 64 | 43.7500% | 44.6289% |
| Finger/face group dropout 5% | 13.4766% | 12.8906% | 46 / 52 | 45.2148% | 44.1406% |
| Finger/face group dropout 10% | 12.9883% | 13.0859% | 48 / 47 | 44.1406% | 44.5313% |

Both checkpoints were trained from scratch under the same 4-epoch, 1,024-example, seed 17, mandatory `torch.compile` protocol, with training dropout 0 versus grouped 0.1. Both saved best checkpoints are from epoch 3 and were evaluated with gamma=1 calibration-accuracy proportional HF/BF weights. Each condition uses a fixed seed, `shuffle=False`, the same worker configuration, and the same first 1,024 validation samples. The SHA-256 of the actual batched feature and frame-mask tensors is identical across models for each condition; all 1,024 sample indices and labels match. This verifies identical realized inputs for the paired comparison, beyond merely assuming RNG reproducibility. Conditions use different fixed seeds, so they are not nested subsets of one missing-mask draw.

The `clean/` and `grouped/` directories contain summaries, per-condition predictions and confusion matrices, and per-class accuracies. The evaluator is `tools/evaluate_face_pair_corruption.py`; its paired-input checksum and predictions were added for this screen. Server copy: `tools/evaluate_face_pair_corruption_paired_20261105.py` under the stable experiment code directory.

## Interrupted Full Candidate

The grouped 0.05 full-training candidate was stopped with SIGINT at user-directed research prioritization: the earlier screen had not established a robustness benefit, there was no same-stage full-training control, and the 100-epoch run was expensive. Server process PID 2189979 exited with `KeyboardInterrupt`. Its independent output directory is `/home/xiong/Desktop/ISAA/outputs/face_pair_grouped_dropout005_20261105/20261105-170130-2e7b5d8e`; `best.pt`, `last.pt`, logs, config and status are retained. The last fully completed epoch was 4; best hard-routing validation Top-1 was 75.1625%, and interruption occurred during epoch 5 validation. Status explicitly reports `interrupted`, last completed epoch 4. Local snapshots are `interrupted_run_config.json`, `interrupted_run_status.json` and `interrupted_run_epochs.csv`. This unfinished run provides no basis for comparing final accuracy or robustness.
