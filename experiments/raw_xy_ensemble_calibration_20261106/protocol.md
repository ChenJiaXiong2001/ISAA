# Fixed raw-XY ensemble calibration screen

Test one predeclared inference ensemble on the cached training-calibration
partition, CPU only. The reference is epoch-83 original HF/BF with its saved
class accuracy-ratio counts and the fixed 50% arithmetic/50% geometric
consensus. The candidate combines 75% of that reference probability and 25%
of the separately trained raw-XY checkpoint's gamma=1 arithmetic probability,
using that checkpoint's own saved accuracy-ratio counts. Temperatures are 1.

Both calibration caches must match their own checkpoint SHA256 and have
identical ordered-member hashes and labels. Verify the original checkpoint
against the known epoch-83 digest and derive each set of weights from the
checkpoint, not from re-evaluated cache predictions. Report pooled and per
subject accuracy, paired corrections/regressions, disagreement and costs.
Only the existing training-calibration partition is read; it has already been
used for each checkpoint's classwise weights and consists of two subjects.
It is not an independent estimate of generalization. The candidates and
checkpoint histories have also undergone prior exploration. No validation
labels are used to change the candidate in this screen.

Inference would require both complete networks. Compare its accuracy benefit
to the increased compute and memory before considering real deployment; this
screen does not measure latency or robustness under missing inputs.
