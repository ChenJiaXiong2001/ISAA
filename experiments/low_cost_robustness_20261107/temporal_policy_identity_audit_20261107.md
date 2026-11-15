# Temporal policy identity audit (2026-11-07)

This is a read-only audit of the frozen temporal-policy post-processing. It
does not change a training entry point, model, checkpoint, or running
experiment, and it does not start training.

The remote analyzer in `analyze_temporal_remote.py` reads the canonical CSV and
then only checks that each of the six temporal CSVs has 16,306 rows. It takes
`prediction` from row `i` and compares it with the canonical label at row `i`.
The corruption evaluator writes `sample_index,label,prediction`, so the
analyzer silently ignores even the available `sample_index` and candidate
`label`. A reordered, truncated-and-padded, or wrong-partition CSV could
therefore produce plausible accuracies while being joined to the wrong truth.

The isolated guard
[`temporal_policy_identity_guard.py`](temporal_policy_identity_guard.py)
closes this gap. For every clean, point-5%, and point-10% CSV under both
`cw_singlepass` and `none_center`, it requires:

- exactly 16,306 rows and the exact `sample_index = 0..16305` sequence;
- candidate labels identical to the canonical label sequence;
- predictions in the 60-class range;
- canonical sample uniqueness and `Axxx - 1 == label` consistency;
- raw frame lengths read from the RTMW ZIP (or an ordered lengths CSV), with a
  deterministic length SHA; and
- a SHA-256 for the canonical file and every candidate CSV.

If a `split_manifest.json` is available, `--manifest` additionally requires
the canonical sample order to match its validation member list. It records both
the manifest's recorded member SHA and the SHA derived from that list and
reports any self-hash mismatch. The guard then recomputes the threshold-100
policy (`cw` when raw length is above
100, `none_center` otherwise), corrected/regressed counts, and subject-level
directions. `--expected-summary` compares `n`, control accuracy, policy
accuracy, and gain against the recorded JSON and fails on any mismatch.

Use this command on the server after copying this one file; it is a short
read-only pass over the existing prediction files and ZIP:

```text
python temporal_policy_identity_guard.py \
  --canonical /home/xiong/Desktop/ISAA/outputs/canonical_p10_pair_full_20261106/predictions.csv \
  --cw-dir /home/xiong/Desktop/ISAA/outputs/temporal_corruption_cw_singlepass_full_20261106 \
  --none-dir /home/xiong/Desktop/ISAA/outputs/temporal_corruption_none_center_full_20261106 \
  --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
  --expected-summary /home/xiong/Desktop/ISAA/outputs/temporal_policy_subject_bootstrap_20261107.json \
  --output /home/xiong/Desktop/ISAA/outputs/temporal_policy_identity_guard_20261107.json
```

Add `--manifest /path/to/split_manifest.json` when the producing checkpoint's
manifest is available. Do not pass a recorded manifest SHA as a substitute for
the member list: the guard is intended to expose any mismatch.

## What is already independently supported locally

The local full temporal-input screen contains 65,224 rows: 16,306 rows for
each of `confidence_weighted`, `none_start`, `none_center`, and `none_end`.
The sample and label sequences match the local canonical prediction CSV for
all four modes, and every label agrees with the `Axxx` action encoded in its
sample path. The local file SHA-256 values are:

| artifact | SHA-256 |
| --- | --- |
| canonical `predictions.csv` | `01346c94ab8c0f70107c021829078a2603ec3250488090dd3fea141f58d0ed11` |
| full temporal-input-screen `predictions.csv` | `a149e944a34d24d84a233f5d993fd264e2766d2b362b2c5d16fe8ab2265c59f9` |

The ordered member SHA is
`82fd60410fd04e381c65d3bb9f24f18f7993124ee2581902942fbfb4a2ecddc0`, matching
the formal validation metadata and local manifests. The ordered label SHA is
`0f103e951cdf0aee7b2aa2647162b831215e032a6e053b89bb1648a4a19749de`.

The local archive length pass found 16,306 matching NPZ members, lengths 36--300
frames, and 4,326 clips above 100 frames. Its ordered
`sample\tlength` SHA is
`980613b8bf64577d4f71a94d9e4916ea4ecf09bc3149cae412d0b2c80fb2e174`, matching
the recorded temporal length bins (11,980 at 0--100; 3,433 at 101--150; 722 at
151--200; 171 at 201--300).

The local temporal screen reproduces the clean comparison exactly:

| condition | `cw` correct | `none_center` correct | difference |
| --- | ---: | ---: | ---: |
| clean | 15,574 | 15,379 | +1.1958788 pp |

This establishes that the clean +1.2 pp arithmetic is reproducible for the
local frozen screen. The same guard was then run read-only on the A6000 server
against the six remote temporal-corruption CSVs and the RTMW archive. The
machine-readable result is
[`temporal_policy_identity_guard_result_20261107.json`](temporal_policy_identity_guard_result_20261107.json),
with `status=PASS` and no warnings. All six files contain 16,306 rows; their
`sample_index` sequences and labels match the canonical sequence, and each
file's SHA-256 is recorded in that result.

The guard reproduced the recorded policy metrics exactly:

| condition | control correct | policy correct | gain |
| --- | ---: | ---: | ---: |
| clean | 15,379 | 15,574 | +1.1958788 pp |
| point 5% | 14,211 | 14,452 | +1.4779836 pp |
| point 10% | 12,729 | 12,972 | +1.4902490 pp |

The validated archive length range is 36--300 frames, with 4,326 clips above
the threshold 100. Because the guard found no identity or hash violation, the
three frozen results are numerically and procedurally reproducible for this
validation partition. They remain synthetic-corruption comparisons rather than
evidence from a new camera-domain split.
