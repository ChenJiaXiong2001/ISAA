# Precompression anchor dropout: long-sequence screen

The 34 clips are all clips longer than 100 frames from a previously sampled
256-clip training-subject calibration list. The frozen seed2 checkpoint and its
gamma=1 calibration weights were used. A fixed member-specific random seed
independently hides 5% of valid shoulder/hip anchor points before temporal
compression; 843 original point-frames were hidden. All methods use the same
samples, labels, mask, and corrupted input.

| Input | No repair | max-gap 5 | threshold 0.95 + max-gap 5 | threshold 0.98 + max-gap 5 |
| --- | ---: | ---: | ---: | ---: |
| Clean, correct / 34 | 33 | 33 | 33 | 25 |
| Precompression dropout, correct / 34 | 32 | 33 | 33 | 26 |

On clean data, threshold 0.98 introduced eight errors; under dropout it
corrected one and introduced seven. Threshold 0.95 and max-gap alone both
corrected one dropout error without introducing errors. This small screen
provides a negative signal for 0.98 and no evidence of additional improvement
from threshold 0.95. It does not establish population-level robustness or
performance on real detector failures; the checkpoint weights were calibrated
using the same training-subject partition. Max-gap 5 needs up to five future
frames.

`summary.json` records the checkpoint, ordered members, source list, input,
and mask hashes; `predictions.csv` contains the paired outputs. The source
member list is archived at
`../face_pair_anchor_confidence_calibration256_20261105/long_members_precompression_20261106.csv`.
Member selection tests passed on the server; separate parity checks verified
clean tensors, frame masks, labels, and cross-batch predictions.
