# Face-pair loss-weight screening

This is a low-cost loss adjustment screen, not a seed study. Both runs use the same fixed seed 17 only to hold the data order and initialization constant while changing the loss coefficient. The protocol is 1,024 samples per split, four epochs, batch 32, one warmup epoch, calibration routing from epoch 1, `lr=0.025`, gradient clipping 5, CUDA Inductor compilation required, and random scratch initialization.

The objective is

```text
loss = fusion_weight * CE_fused + head_weight * (CE_HF + CE_BF)
```

| Variant | Head weight | Fusion weight | Best validation fused Top-1 | Last validation fused Top-1 | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `loss_h075_f100` | 0.75 | 1.0 | 11.4258% | 11.3281% | completed |
| `loss_h100_f100` | 1.0 | 1.0 | 12.1094% | 11.6211% | completed |

The earlier matched screen with `head=0.5, fusion=1.0` reached 12.5977% on the same short protocol. Therefore increasing auxiliary head supervision to 0.75 or 1.0 shows no early advantage. The result is only a screening signal because the split contains 890 fit, 134 calibration, and 1,024 validation samples and only four epochs; it does not justify a full 100-epoch run for either variant.

Both runs completed with `compile=true`, `required=true`, backend `inductor`, finite losses and gradients, and no NaN/Inf abort. Full artifacts are in each variant directory: `config.json`, `status.json`, `epochs.csv`, and the captured log.
