# Person-slot occupancy diagnostic

Read only the 3,191 epoch-83 training-calibration members and the corresponding
cached logits, CPU only. Reconstruct the frozen checkpoint's gamma=1 weights
and its 50/50 arithmetic/geometric consensus. Compute second-slot body
occupancy as the fraction of valid compressed frames having at least one
positive-confidence body input point in person slot 1. Slot numbers are archive
positions; they do not prove distinct stable identities. Use the body indices
from the model. Check the ordered member digest, cache checkpoint digest, labels,
and clean-input and frame-mask hashes. Record each sample's occupancy and model
prediction. Fixed descriptive groups are zero, (0, 50%), and at least 50%.

Report overall, per-subject, and per-action results; sparse groups are not used
as causal comparisons. Also split action IDs 50-60 from 1-49 as a coarse
interaction-action indicator, with the caveat that IDs alone do not identify
real identities or presence of bystanders. This is a diagnostic of naturally
recorded inputs, not an intervention or an independent evaluation. Calibration
subjects generated the frozen weights. A relationship between second-slot
occupancy and accuracy can be confounded by action, subject, recording and
detection quality. Do not remove or downweight people based on this scan.
