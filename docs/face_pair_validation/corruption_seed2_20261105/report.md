# Face-pair corruption robustness

Checkpoint: `best_face_pair_100_seed2_20261105/20261105-051202-2fcd5590/best.pt`, epoch 100, `best-face-pair`, 16,306 XSub validation samples. The checkpoint is frozen. All conditions use the saved training-subject accuracy-proportional HF/BF probability mixture (`gamma=1`); corruption is applied only by `RTMWZipDataset`, followed by the normal derived-feature reconstruction.

| condition | Top-1 | Top-5 | change from clean |
|---|---:|---:|---:|
| clean | 95.5109% | 99.5033% | 0.0000 pp |
| independent body-point dropout 0.05 | 88.6300% | 98.0191% | -6.8809 pp |
| independent body-point dropout 0.10 | 79.5535% | 95.2410% | -15.9573 pp |
| grouped finger/face-token dropout 0.05 | 94.7197% | 99.3561% | -0.7911 pp |
| grouped finger/face-token dropout 0.10 | 93.2908% | 99.1108% | -2.2200 pp |

The model is substantially more sensitive to independent body-point loss than to complete finger/face groups disappearing. This is robustness evidence, not a clean-accuracy improvement claim. Existing training-time dropout runs should therefore be evaluated on corruption validation before being judged as robustness improvements; their clean Top-1 results alone do not establish that benefit.

Largest per-class drops were A029/A036 at point dropout 0.05 and A036/A012 at point dropout 0.10. For grouped dropout, A011/A029 were the most affected at 0.05 and A029/A011 at 0.10. Detailed confusion matrices and class accuracies are in the accompanying CSV files; the machine-readable source is `summary.json`.
