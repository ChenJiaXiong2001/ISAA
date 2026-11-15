# Person-slot occupancy on training calibration

The frozen epoch-83 geometric consensus was paired with compressed input for
3,191 training-calibration clips. Slot 1 is the second archive position, not a
verified person identity. A slot is present in a frame if any model body input
point has positive confidence; the fraction is measured over valid compressed
frames. The members, clean tensor and frame-mask hashes match the prior scan.

| Slot 1 visible fraction | Samples | Correct | Accuracy |
|---|---:|---:|---:|
| 0 | 2490 | 2340 | 93.9759% |
| >0 and <50% | 130 | 126 | 96.9231% |
| >=50% | 571 | 556 | 97.3730% |

Of the 571 clips with >=50% visibility, 567 are actions A050-A060, the NTU60
interaction group; none of these interaction-group samples are in the other
two bands. Action mix therefore strongly confounds the aggregate differences.
These data do not show that the second slot harms accuracy; the second person
may carry required interaction information. Slot continuity and whether a
detected person is a bystander were not established.

A possible fixed, label-independent intervention would remove slot 1 only
when its compressed visibility is >0 and <50%, leaving all other clips intact.
On these cached calibration predictions, that affects 130 clips, of which only
four are errors and 126 are currently correct. Assuming exactly unchanged
predictions for the untouched clips, its **maximum possible net clean gain on
this partition** is four clips, 4/3191 = 0.1254pp; any new errors reduce it.
This is a theoretical bound, not a prediction the intervention works. The
restriction uses a complete sequence and may not be immediately applicable to
streaming. With this small upside and possible losses, this scan does not
justify a GPU inference experiment or removing a person in production.

The two calibration subjects also supplied the checkpoint's weights; the data
are not an independent performance test. `audit.py` independently checks the
CSV counts, action and subject IDs, occupancy groups, member digest and
previously recorded input/mask digests. Per-action and per-subject results are
in `summary.json`, with per-sample values in `samples.csv`.
