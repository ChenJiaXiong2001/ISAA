# Mask-aware derived geometry frozen screen

The screen uses the formal epoch-83 `best-face-pair` checkpoint:

- checkpoint SHA256: `aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`
- validation manifest SHA256: `82fd60410fd04e381c65d3bb9f24f18f7993124ee2581902942fbfb4a2ecddc0`
- samples per condition: 16,306
- conditions: clean plus point/group dropout at 5% and 10%
- CUDA: RTX A6000, PyTorch 2.14 cu126, Inductor `default`, `suppress_errors=false`

The production `FacePairFusion` and its entry points were not changed. The
isolated variants are loaded with the formal checkpoint for inference-only
screening. `geometry_only` masks cross-hand distance/direction whenever the
four torso anchors are incomplete. `center_scale` additionally replaces the
partial torso reference with center zero and scale one.

## Results

`geometry_only` is neutral to slightly negative:

| condition | formal base | mask-aware | delta |
|---|---:|---:|---:|
| clean | 95.28394% | 95.27781% | -0.00613pp |
| point 5% | 86.16460% | 86.15847% | -0.00613pp |
| point 10% | 74.79455% | 74.78842% | -0.00613pp |
| group 5% | 93.98381% | 93.98381% | 0.00000pp |
| group 10% | 92.41997% | 92.41997% | 0.00000pp |

The subject-bootstrap intervals are compatible with zero for all corruption
conditions. `center_scale` is rejected as unsafe: clean falls by -0.21464pp
(95% CI [-0.30223, -0.06260]pp), point 5% falls by -76.07016pp, point 10%
falls by -69.06047pp, group 5% by -0.16558pp, and group 10% by -0.17172pp.
The large point-dropout failure shows that replacing a partial torso frame's
coordinate reference destroys the learned coordinate distribution. Neither
variant is promoted to training.

Artifacts:

- `outputs/derived_geometry_mask_frozen_geometry_20261107/summary.json`
- `outputs/derived_geometry_mask_frozen_center_20261107/summary.json`
- `evaluate_frozen.py`
- `model.py`

These are exploratory frozen-checkpoint results on a repeatedly used
validation set and synthetic corruption masks. They are negative evidence for
this particular mask-aware transformation, not evidence against all possible
derived-feature dropout or consistency training.
