# Calibration weight stability screen

Use only epoch-83 training-subject calibration logits and their manifest. The
candidate is fixed in advance: add one pseudo-correct count to each head per
class, then form the gamma=1 ratio. Retain unit temperatures and the fixed 50%
normalized arithmetic / 50% weighted geometric fusion. Compare against the
same mixture with the unmodified gamma=1 ratio. Fit class weights on one
calibration subject and test on the other, then reverse the fold. Report top-1,
NLL, corrected/regressed samples, classwise subject-weight differences and
setup-level weight variability. No official validation data or hyperparameter
selection. Two calibration subjects are insufficient for a population claim;
setup ID describes recording setup, not chronological time.
