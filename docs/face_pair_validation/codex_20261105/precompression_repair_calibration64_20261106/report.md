# Precompression anchor dropout: 64-sample calibration screen

This is an exploratory CPU screen, not a full validation result. The first 64
members of the training-subject calibration manifest were evaluated with the
frozen seed2 face-pair checkpoint and its gamma=1 calibration weights. The
same calibration partition supplied those weights. Fourteen clips exceeded
100 original frames and were temporally compressed; the remaining clips were
padded. A deterministic member-specific seed applies independent 5% dropout
to valid shoulder/hip anchor points before the existing confidence-weighted
compression. All repair methods use the same corrupted input and labels.

| Input | No repair | max-gap 5 | threshold 0.95 + max-gap 5 | threshold 0.98 + max-gap 5 |
| --- | ---: | ---: | ---: | ---: |
| Clean, correct / 64 | 60 | 60 | 60 | 43 |
| Precompression dropout, correct / 64 | 60 | 61 | 60 | 42 |

On clean inputs, the 0.98 candidate corrected 2 examples but introduced 19
errors; under dropout it corrected 2 and introduced 20. The 0.95 candidate had
no net gain under dropout (one correction and one new error). The unthresholded
max-gap repair corrected one dropout error and introduced none. These counts
show a strong adverse signal for 0.98 on this prefix, but cannot establish
population-level effects or generalize to real detector failures. The observed
dropout pattern is synthetic, and the candidate uses up to five future frames.

In the 14 clips actually compressed, pre-dropout correct counts were 12 for
both no repair and max-gap, 11 for threshold 0.95 plus max-gap, and 9 for
threshold 0.98 plus max-gap. In the 50 shorter clips the corresponding counts
were 48, 49, 49, and 33. The compressed subgroup is too small for a reliable
estimate of the low-positive-score failure mode.

The paired tool checks the source manifest digest and records member, input,
mask, and checkpoint hashes in `summary.json`; predictions are in
`predictions.csv`. Separate CPU checks on four long and four short clips found
that clean tensors, masks, and labels match the original dataset exactly;
batch-size 4 and 8 predictions matched on an eight-sample smoke run.
