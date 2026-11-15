# Temporal input construction screen

This screen tests whether the frozen seed2 face-pair checkpoint depends on confidence-weighted compression or on the deterministic 100-frame crop location. It does not train, change model weights, recalibrate routing, inject synthetic corruption, or modify the default training path.

The evaluator uses the exact validation member order and digest from the checkpoint manifest, the checkpoint's own gamma=1 accuracy-proportional calibration weights, and raw x/y/score inputs. It evaluates four fixed modes in one run:

| Mode | Temporal construction |
|---|---|
| `confidence_weighted` | Existing default: complete sequence compressed into 100 confidence-weighted bins; no second crop |
| `none_start` | No compression; deterministic first 100-frame crop |
| `none_center` | No compression; deterministic centered 100-frame crop |
| `none_end` | No compression; deterministic last 100-frame crop |

For each mode, the output reports full validation and the first 4,096 members of that same validation manifest. It records HF, BF, and gamma=1 soft Top-1 metrics, exact member/input digests, checkpoint and archive hashes, and all predictions. The input digest covers each member name, tensor dtype/shape/bytes, label, and frame mask in manifest order.

The evaluator is `tools/evaluate_face_pair_temporal_input_screen.py`. CPU protocol tests are in `tests/test_face_pair_temporal_input_screen.py`. GPU execution is intentionally separate from this protocol note so the measured result can record the actual device and wall time.
