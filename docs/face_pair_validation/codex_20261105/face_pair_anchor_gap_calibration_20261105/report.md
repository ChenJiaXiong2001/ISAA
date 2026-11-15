# Anchor gap stress test

This frozen model evaluation uses the checkpoint's 1,420 training-subject calibration samples and its fixed gamma=1 class weights. Those same samples contributed the weights, so this is development evidence, not an independent test. Conditions were fixed before observing results: independent 5% anchor loss or fixed-length gaps of 2, 5 and 10 frames, with start probabilities 0.025, 0.01 and 0.005 per anchor/frame/person respectively. They use shared random starts, producing approximately 5% masked valid anchor point-frames after overlaps and boundary truncation. Corruption is applied after temporal compression, and does not represent a measured detector failure process.

The tested rule fills only an isolated zero-score anchor with the mean XY and score of valid adjacent frames; it never extrapolates or fills a run of two or more missing frames. Results are gamma=1 Top-1:

| Condition | Actual masked | Synthetic points repaired | Unmodified | Repaired | Change |
|---|---:|---:|---:|---:|---:|
| Clean | 0 | 0 | 94.3662% | 94.3662% | 0 pp |
| Independent 5% | 23,919 (5.0292%) | 20,962 (87.64%) | 89.7183% | 94.0141% | +4.2958 pp |
| Gaps of 2 | 23,482 (4.9373%) | 0 | 90.4930% | 90.4930% | 0 pp |
| Gaps of 5 | 23,021 (4.8404%) | 0 | 91.9014% | 91.9014% | 0 pp |
| Gaps of 10 | 22,788 (4.7914%) | 0 | 92.8873% | 92.8873% | 0 pp |

For independent missing frames, repair fixes 77 samples and introduces 16 errors, net +61/1,420, with paired 20,000-replicate bootstrap 95% interval [3.0282, 5.6338] percentage points. Under the simulated contiguous gaps, the method restores none of the artificially hidden point-frames. It does fill 55–60 naturally absent point-frames depending on the condition, but that does not change the aggregate gamma=1 correct count here. The chosen gaps, independent across anchors, are a controlled stress test; they do not establish real occlusion frequency, temporal duration, or benefit for pre-compression or low-but-nonzero scores.

`summary.json` includes model/sample hashes, exact actual hidden and repaired point counts, and HF/BF/gamma=1 results. `predictions.csv` contains paired sample-level predictions. The evaluator is `tools/evaluate_face_pair_anchor_gaps.py`; the repair itself is imported from `tools/evaluate_face_pair_anchor_interpolation.py`.
