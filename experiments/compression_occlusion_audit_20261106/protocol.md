# Audit fixed-duration occlusion versus compression

Join existing calibration-only inputs and predictions by exact member ID. Use
the original-frame lengths from the complete calibration confidence scan and
the previously generated paired same-side 5-original-frame occlusion results.
Freeze natural compression bins before examining the joined results: up to
100 frames (no compression), 101-199 frames (bins of 1-2 original frames),
and 200 or more frames (bins of at least 2 original frames). Report sample
and subject counts, the raw occluded fraction 5/T, clean and occluded
arithmetic/geometric accuracy and their paired drops, without selecting
parameters. These groups differ in clip content and subjects as well as
duration, so their differences are not causal evidence about compression.

Independently determine the exact output bin lengths and the number of bins
wholly or partially overlapped by each recorded interval using integer floor
boundaries. Whole-bin overlap hides the selected region in every raw frame;
partial overlap is mixed in a bin. Report counts by duration group and
subject. Existing scores and predictions are reused, no new inference or GPU.
This is the existing deterministic synthetic occlusion protocol and does not
represent real failure rates. Two calibration subjects are insufficient for
population uncertainty; no validation labels or results enter this audit.
