# Geometric fusion calibration-weight screen

Epoch-83 calibration logits contain 3191 samples from only P009 and P018.
No official validation data were read. The predeclared candidate adds one
pseudocount to each head's classwise correct counts, retaining the fixed 50%
arithmetic / 50% geometric mixture and unit temperatures. Each fold fits its
weights on the other subject.

| Held-out subject | Original gamma=1 correct | Smoothed correct | Original NLL | Smoothed NLL |
| --- | ---: | ---: | ---: | ---: |
| P009 | 674/708 | 674/708 | 0.245551 | 0.245598 |
| P018 | 2342/2483 | 2341/2483 | 0.206907 | 0.206915 |

The candidate provides no evidence to replace the original weights. P009 and
P018 independently fitted HF weights differ by 0.0549 on average across the
60 classes, median 0.0415, maximum 0.2566. The maximum is A048 (12 P009 and
42 P018 examples); the sample counts are too small to identify the cause.
Within-subject setup weights differ from their respective whole-subject weights
by approximately 0.03-0.05 on average, but individual setup/action cells are
sparse and overlap the whole-subject reference. Setup IDs do not establish
chronological time, so this is not evidence of temporal drift.

Previous arithmetic-fusion screens already tested four pseudocounts and nine
dynamic confidence gates without clear top-1 improvement. This one fixed
candidate tests the actual geometric mixture, without validation-set tuning.
Both calibration subjects participated in the original routing calibration;
two subjects do not establish generalization. Output including all classes,
setups, hashes and fold counts is on the server at
`/home/xiong/Desktop/ISAA/outputs/geometric_weight_stability_loso_20261106/summary.json`.
