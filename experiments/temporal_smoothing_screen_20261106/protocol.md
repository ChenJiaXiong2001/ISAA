# Fixed confidence-weighted temporal smoothing mechanism screen

Use 256 evenly spaced entries from the epoch-83 3191-member training-calibration
manifest, selected without labels. Reuse the same-side
five-original-frame synthetic occlusion protocol, exact member-derived side and
window, confidence-weighted compression and auto score normalization. Compare
three-frame [1, 2, 1] confidence-weighted XY smoothing to unsmoothed compressed
XY, separately on clean and occluded inputs. Apply only when a point exists in
all three consecutive valid frames, preserve all score and missing masks, and
leave frame endpoints unchanged. No label or model inference selects the rule.
This is different from filling missing values; it does require a future frame.

Treat the unoccluded compressed XY as a reference, not ground truth. Compare
the mean XY displacement from that reference for (a) compressed point-frames
whose raw bin intersects the injected occlusion window, for the selected body
and hand region, and which remain present in both inputs, and (b) all other
shared-valid point-frames. Also report the displacement introduced when the
same smoother is applied to the clean input. Score means represent compression
and are mostly close to one, not measured geometric accuracy. Report point-frame
and clip-level statistics; highly correlated point-frames are not independent.
Use original body-length-normalized distances when the clean shoulder and hip
are both valid, and report eligible counts. Synthetic intervention and this
single proxy cannot establish action-recognition or real-world improvement.
