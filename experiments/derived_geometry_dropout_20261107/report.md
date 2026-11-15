# Derived geometry dropout scratch screen

This isolated candidate applies training-only Bernoulli dropout with
probability 0.025 to the hand path's derived cross-hand distance and direction
channels. Raw coordinates and confidence remain intact. Evaluation runs with
dropout disabled. The production `FacePairFusion` model and all existing
entries are untouched.

The CUDA smoke test passed with mandatory Inductor compilation and two finite
optimizer updates. Both 4-epoch runs used random initialization, seed 17 or
29, 4,096 training samples, the same split and loss protocol, and no warm
start.

| seed | clean | point 5% | point 10% | group 5% | group 10% |
|---:|---:|---:|---:|---:|---:|
| 17 | +1.453pp [0.838, 2.379] | +1.110pp [0.117, 1.660] | +0.736pp [-0.097, 1.223] | +1.938pp [1.161, 2.725] | +2.539pp [1.701, 3.413] |
| 29 | +1.490pp [0.602, 2.217] | +1.637pp [0.238, 2.531] | +2.502pp [1.469, 3.591] | +1.785pp [0.650, 2.547] | +1.748pp [0.884, 2.259] |

Intervals are 20,000 complete-subject bootstrap draws over the 16,306-sample
paired validation comparison. Both independent short screens pass the fixed
clean and robustness gates. A full 100-epoch validation run has now been started after both gates passed.

Artifacts are under the corresponding remote output directories:

- `outputs/derived_geometry_dropout_control_4ep4096_seed17_20261107/`
- `outputs/derived_geometry_dropout_candidate_4ep4096_seed17_20261107/`
- `outputs/derived_geometry_dropout_paired_4ep4096_seed17_20261107/`
- `outputs/derived_geometry_dropout_control_4ep4096_seed29_20261107/`
- `outputs/derived_geometry_dropout_candidate_4ep4096_seed29_20261107/`
- `outputs/derived_geometry_dropout_paired_4ep4096_seed29_20261107/`

## Full training promotion

Both independent 4-epoch scratch screens passed the fixed gate. A necessary 100-epoch
random-scratch validation was started with seed 29, full fit/calibration/validation
partitions, mandatory Inductor compilation, and training-only dropout p=0.025.
Output: `/home/xiong/Desktop/ISAA/outputs/best_face_pair_derived_geometry_dropout_100_20261107/`.
No formal checkpoint was loaded.
