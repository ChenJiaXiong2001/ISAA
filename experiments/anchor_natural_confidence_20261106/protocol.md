# Natural low-confidence anchor scan

Scan all 3191 training calibration members only, in manifest order. Read raw
and confidence-weighted-compressed detector inputs and no prediction labels.
The four anchors are shoulders/hips (5, 6, 11, 12); valid person-frames have
at least one positive anchor and a valid frame. Report positive scores at or
below 0.5 and 0.9, zeros, affected clips and subject-level totals, with raw
and compressed values distinguished. These two fixed thresholds were examined
in the previous 34-long-clip mechanism analysis and are not new discoveries.

For compressed inputs compare existing max5 zero-gap interpolation to zeroing
positive anchors at/below the fixed threshold followed by the same max5.
Count newly flagged points restored by interpolation or left missing, clips
affected, torso-center changes, and changes to shoulder/hip scale on comparable
person-frames. No class labels, model inference or GPU are used. Torso center
and scale changes only quantify input sensitivity and are not ground-truth
error or evidence of improved accuracy. All measurements concern the supplied
RTMW archive, not a general detector failure rate.

The dataset uses `score_normalization=auto`: it applies sigmoid to a sample's
scores when their maximum exceeds 1, otherwise clips to [0,1]. Compression
uses confidence-weighted XY and mean confidence over each bin; clips of at
most 100 frames are unchanged. Score populations include valid frame/person
combinations with at least one positive shoulder/hip anchor, including zero
anchors in those combinations; no empty people or padded frames are counted.
