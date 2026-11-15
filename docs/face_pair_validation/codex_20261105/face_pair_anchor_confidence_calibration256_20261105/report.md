# Anchor confidence sample

This read-only CPU diagnostic draws 256 samples from the checkpoint's training-subject calibration manifest using a fixed sampling generator. `samples.csv` contains the selected member names, original frame counts and compression flags; `summary.json` includes both manifest and selected-member SHA-256 digests. It does not sample official validation data.

Sequences contain 41–189 original frames, with median 72. Only sequences longer than 100 frames are compressed by the current loader: 34/256 (13.2813%) in this sample. Shorter sequences are retained and padded later. Within a compression bin, XY coordinates are confidence-weighted and output score is mean confidence. Thus a zero score in one original frame need not remain zero after compression if the bin also contains positive scores.

The score population is the four shoulder/hip anchors for every valid frame/person that has at least one positive anchor, before padding. This excludes empty people and invalid frames but includes zero-valued anchors among otherwise present people. After the loader's automatic score normalization:

| Stage | Total anchor point-frames | Zero | Positive | Positive below 0.5 | Positive 1st percentile | Positive median |
|---|---:|---:|---:|---:|---:|---:|
| Original frames | 87,584 | 54 | 87,530 | 0 | 0.9179 | 0.9808 |
| After compression where applicable | 85,076 | 54 | 85,022 | 2 | 0.9183 | 0.9809 |

FacePairFusion uses score > 0 as validity for anchors; any positive anchor contributes equally to the center, regardless of confidence. The interpolation under investigation only acts on exact zero scores. These statistics are limited to 256 calibration clips and four anchor points from this archive; they are not a distribution for other joints, cameras, detectors or real-world occlusions. Synthetic post-compression zeroing and pre-compression corruption represent different input conditions.

Reproduce with `tools/analyze_face_pair_anchor_confidence.py`; `summary.json` includes exact counts and quantiles.
