# Fixed control-weight sensitivity

This analysis reuses the paired validation branch logits from the 20-epoch
hard-route control and lagged soft-route candidate. The candidate logits are
fused with the control checkpoint's class-wise gamma=1 calibration weights;
the weights are read from the control training calibration counters and were
fixed before reading validation outcomes. The comparison is against the
control prediction using the same control weights. Validation labels are used
only to count the reported outcomes and bootstrap intervals.

The protocol already declared this as a common-weight sensitivity. It is a
transfer-weight robustness result, not independent generalization evidence:
both checkpoints came from one matched short run and the same validation
partition, and the transfer weights are from the matched control checkpoint.

The CSV provenance is recorded in
`fixed_control_weight_bootstrap_20261107.json`; it uses 20-subject,
20,000-draw multinomial bootstrap intervals.

| Condition | Difference (pp) | 95% subject bootstrap CI (pp) | Equal-subject difference (pp) |
|---|---:|---:|---:|
| clean | +0.5397 | [-0.1327, +1.2778] | +0.7547 |
| point 5% | +3.3546 | [+2.5173, +4.9389] | +4.3971 |
| point 10% | +3.1829 | [+2.1106, +5.3824] | +4.7808 |
| group 5% | +2.2262 | [+1.3683, +3.6973] | +3.1677 |
| group 10% | +2.3795 | [+1.5174, +3.8574] | +2.9942 |
| anchor-pre 5% | +1.3185 | [+0.7688, +2.5506] | +2.4236 |
| anchor-pre 10% | +0.7175 | [-0.0758, +1.9791] | +1.6726 |

The robust-condition gains are consistent with the candidate's stronger
body-face branch, while the clean and long-anchor intervals cross zero. These
numbers support testing the transfer-weight rule in a separately held-out
subject evaluation; they do not justify promoting the candidate on this
partition.
