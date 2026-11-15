# ISAA 研究结论（2026-11-07）

本轮研究围绕 canonical `best-face-pair`、冻结推理策略和一个随机 scratch 训练候选展开。训练候选保留原模型、旧入口和旧实验；训练入口强制 `torch.compile`/Inductor，拒绝 `--no-compile`、整模型 eager fallback 和 checkpoint warm start。

## 结论排序

### 保留：冻结 0.5 geometric fusion

在正式 xsub60 validation（16,306 条、20 个 subject）上，把 arithmetic probability pool 与 normalized weighted-geometric pool 以 0.5/0.5 混合：

| 条件 | 相对 arithmetic | subject bootstrap 95% CI |
|---|---:|---:|
| clean | +0.3557pp | [+0.234, +0.530] |
| point 5% | +0.675pp | [+0.283, +1.136] |
| point 10% | +0.092pp | [-0.330, +0.416] |
| group 5% | +0.497pp | [+0.275, +0.894] |
| group 10% | +0.442pp | [+0.282, +0.716] |

xset60 与原 xsub60 validation 的 6,023 条交集（不是独立测试）复核为 arithmetic 95.5006%，geometric 95.7828%，提升 +0.2823pp。

### 保留为长序列策略：threshold=100 temporal policy

对 raw frame length `>100` 使用 confidence-weighted temporal compression，否则使用 `none`。在 xsub validation 上 clean +1.1959pp，point 5% +1.4780pp，point 10% +1.4902pp；identity guard、样本顺序、labels 和 hash 均通过。xset 交集上 canonical 从 none 的 93.9233% 提升到 95.5006%，但该交集不是独立测试。

### 保留为缺失检测鲁棒性策略：max5 anchor interpolation

只在有效帧、前后端点存在、无 invalid frame 且 gap 不超过 5 时插值，不外推。正式 validation clean 约中性（-0.0123pp）；synthetic precompression anchor dropout 下，5% 缺失提升 +3.8513pp，gap2 +2.974pp，gap5 +1.564pp，gap10 基本无收益。与 geometric fusion 的全量 compiled summary 已完成：5% repaired geometric +0.4354pp（CI [+0.3303,+0.5928]），10% +0.4293pp（CI [+0.2946,+0.6315]）。这些 corruption 是合成协议，validation 被重复使用，不能当作真实采集盲测。

冻结模型 benchmark 显示 max5 repair-only 为 batch1 median 0.646ms、P95 0.656ms；batch32 median 0.643ms、P95 0.652ms。repair+forward 相对 unmodified 约增加 0.65–1.00ms/batch。该 benchmark 不含 detector、archive loading、压缩和 look-ahead 等端到端成本。

## 已完成完整训练验证但淘汰的方向

### derived geometry dropout scratch

两个独立 4-epoch screen 曾显示约 +1.45/+1.49pp clean，因此按预注册 gate 启动了 100-epoch random-scratch 训练。终态审计通过：`status=completed`、100 epochs、best epoch 87、best validation 93.8857%，随机初始化，无 init/resume checkpoint，Inductor compile guards、数据分区、routing、strict load 和 finite tensors 全部通过。

但正式 paired evaluator（16,306 条、五条件、20,000 次 complete-subject bootstrap）显示相对 stable control：

| 条件 | candidate own routing | candidate control-fixed routing |
|---|---:|---:|
| clean | -0.8279pp，CI [-1.2730,-0.0647] | -0.9138pp |
| point 5% | -1.8643pp，CI [-2.4408,-0.7315] | -2.1219pp |
| point 10% | -2.1281pp，CI [-2.8346,-1.0153] | -2.3120pp |
| group 5% | -0.6501pp，CI [-1.2128,+0.3964] | -0.8402pp |
| group 10% | -0.6930pp，CI [-1.3344,+0.5754] | -0.9076pp |

control-fixed routing 没有回升，因此不继续冻结、滞后或平滑该 routing，也不保留 dropout 训练候选。xset validation-intersection arithmetic 为 candidate 94.7202% vs canonical 95.5006%，仍低 0.7803pp；geometric mix 后为 94.9859% vs 95.7828%。

## 其他已排除方向与瓶颈判断

- `score^p` temporal compression、raw-XY、center/scale geometry、mask-aware geometry、rank fusion、soft confusion/KL、动态 confidence gate、时间平滑、occupancy 和自然置信度修复没有稳定的 clean/鲁棒性正证据。
- 8-epoch hand-face temporal interaction 的 corruption 收益伴随 clean 约 -0.47pp，未通过 clean gate。
- 当前瓶颈更像双分支表示和跨分支排序，而不是简单概率权重；结构诊断中 both-wrong 仅约 465/16,306，但 oracle upper bound 仍约 +1.858pp，说明还有跨头互补空间。

## 验证状态

- derived-dropout CPU smoke：通过。
- weighted regression tests：7 项通过，1 项按 CUDA 条件跳过。
- derived-dropout checkpoint checker：`status=passed`。
- paired evaluator、xset arithmetic/geometric evaluator、max5 full summary 和延迟 benchmark 均完成。
- xset60 结果全部标记为 `validation-intersection; not independent`；不能将其称为独立测试。

当前最合理的实现优先级是：canonical + 0.5 geometric fusion 作为准确率默认策略；threshold=100 作为长序列策略；max5 作为检测到 anchor 缺失时的鲁棒性修复。继续启动新的完整训练没有现有证据支持。
