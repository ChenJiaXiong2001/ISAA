# Raw-XY ensemble calibration screen

The fixed candidate was checked on the two-subject training calibration
partition using paired cached logits. It mixes 75% of the epoch-83 original
HF/BF 50/50 arithmetic/geometric consensus with 25% of the independently
trained raw-XY model's gamma=1 arithmetic probability.

| condition | correct / total | accuracy | corrected | regressed |
|---|---:|---:|---:|---:|
| original consensus | 3022 / 3191 | 94.7038% | — | — |
| 75% original + 25% raw-XY | 3015 / 3191 | 94.4845% | 5 | 12 |
| raw-XY arithmetic alone | 2972 / 3191 | 93.1369% | 44 | 94 |

The candidate change is -0.2194 percentage points. P009 changes from
675/708 to 671/708 (-0.5650pp); P018 changes from 2347/2483 to 2344/2483
(-0.1208pp). An independent float64 NumPy implementation reproduced all
counts. Cache members, labels, and checkpoint digests were checked; each model
uses its own checkpoint-saved classwise accuracy-ratio counts. The original
checkpoint is epoch 83; raw-XY is epoch 85.

This is a calibration-only CPU screen, not an independent test. Both models
and their routing weights have prior use of these two subjects, and the
candidate requires two complete networks, increasing inference cost. It does
not test missing-input robustness. The result provides no reason to run a
full validation GPU evaluation or pursue this ensemble.
