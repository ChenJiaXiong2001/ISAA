# Seed-29 soft-route paired validation (2026-11-07)

Validation used epoch-20 `last.pt` checkpoints with identical input batches, frame masks, split manifest, and gamma=1 fusion reconstruction. The candidate was scratch-trained for 20 epochs with `torch.compile` mandatory/default on CUDA.

## Own training weights

| condition | control % | candidate % | delta pp | subject bootstrap CI95 pp | corrected/regressed | equal-subject delta pp |
|---|---:|---:|---:|---|---:|---:|
| clean | 70.8083 | 70.0724 | -0.7359 | [-1.4971, +0.4309] | 1299/1419 | -0.7182 |
| point_0.05 | 43.3215 | 50.9567 | +7.6352 | [+6.0566, +9.9825] | 2432/1187 | +7.7497 |
| point_0.1 | 33.4662 | 40.4820 | +7.0158 | [+5.6237, +9.5158] | 2290/1146 | +8.0995 |
| group_0.05 | 66.2640 | 65.8531 | -0.4109 | [-1.3261, +0.9427] | 1446/1513 | -0.4954 |
| group_0.1 | 61.9159 | 62.5291 | +0.6133 | [-0.6087, +1.9478] | 1626/1526 | -0.0311 |
| anchor_pre_0.05 | 52.7597 | 54.7161 | +1.9563 | [+0.2888, +3.1650] | 1777/1458 | +0.6957 |
| anchor_pre_0.1 | 44.8240 | 48.9636 | +4.1396 | [+2.3946, +5.5294] | 2071/1396 | +3.0576 |

## Fixed-control weight sensitivity

| condition | delta pp | subject bootstrap CI95 pp | corrected/regressed |
|---|---:|---|---:|
| clean | -1.5638 | [-2.2924, -0.6829] | 1223/1478 |
| point_0.05 | +7.8069 | [+6.2272, +10.1072] | 2456/1183 |
| point_0.1 | +7.2857 | [+5.8934, +9.7522] | 2299/1111 |
| group_0.05 | -0.2882 | [-1.1461, +0.9083] | 1425/1472 |
| group_0.1 | +1.2940 | [-0.0148, +2.6576] | 1643/1432 |
| anchor_pre_0.05 | +2.7413 | [+0.8947, +3.8356] | 1831/1384 |
| anchor_pre_0.1 | +4.8816 | [+3.2537, +6.2786] | 2160/1364 |

## Pre-registered gate

- clean own-weight gate (>= +0.25 pp): **FAIL** (-0.7359 pp)
- all listed corruption own-weight gate (>= -0.25 pp): **FAIL** (group 5% is -0.4109 pp)
- overall candidate decision: **FAIL; do not promote**

Fixed-control rows are transfer-weight sensitivity only, not independent efficacy evidence.
