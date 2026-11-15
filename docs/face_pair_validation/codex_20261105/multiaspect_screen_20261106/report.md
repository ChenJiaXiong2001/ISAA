# 原准确率比例加权 HF/BF 模型：多方面短训筛选（2026-11-06）

本轮把 gamma=1 的原准确率比例加权 HF/BF 融合作为固定基线。所有候选从随机初始化训练，训练入口保留强制 `torch.compile`/Inductor；短训只用于筛选方向，不能把 4,096 前缀或 8 epoch 的绝对准确率外推到 100 epoch 的 95.28% 全量结果。

## 已完成的低成本筛选

| 候选（8 epoch，4096 训练前缀） | clean 差值（pp） | 点缺失 5% 差值（pp） | 点缺失 10% 差值（pp） | 结论 |
|---|---:|---:|---:|---|
| 手运动增强，clean 保留 75% | +0.399 | +20.226 | +20.011 | 保留，鲁棒性强，clean 小幅正向 |
| 手运动增强，clean 保留 90% | -0.613 | +17.791 | +16.540 | 淘汰，clean 下降 |
| 手脸几何 | +0.693 | +6.936 | +7.887 | 保留，clean 与鲁棒性同时改善 |
| 手脸几何 + 0.5% dropout | -0.025 | +4.115 | +7.181 | 暂不扩训，clean 无收益 |

以上均来自同一验证成员、同一输入张量和 gamma=1 软融合的冻结配对评估；差值是 candidate-control，不能用不同协议的绝对值直接比较。

## 融合与输入消融

已有缓存 logits 的 112 组融合组合显示：gamma=1 原准确率比例方案仍是正式默认。gamma=2 的最高观察值只比 gamma=1 高 0.0491 pp，且不是按训练内校准预先选出的稳定方案；因此不把它替换为正式方案。对同一 seed-2 checkpoint，confidence-weighted 压缩的 soft 融合为 94.4824%，none/start、none/center、none/end 分别为 93.3594%、93.4082%、93.5303%，说明当前 100 帧置信度压缩仍是低成本且有效的输入路径。

## 当前 20 epoch 复核

为检验短训筛选不依赖单个早停点，已完成 p=0.05、clean 保留 75%、seed=17、4096 样本、20 epoch 的候选训练；随后按完全相同协议启动 p=0 对照训练。两者都固定使用 `last.pt` 第 20 轮作为配对输入，避免用验证集挑选 checkpoint。候选和对照完成后执行全量 16,306 条验证及点/组缺失条件，并做 subject-cluster bootstrap。

## 研究决策

在 20 epoch 配对结果出来前，不启动 100 epoch 长训。若 p=0.05/clean75 在 clean 与缺失条件都保持正向，再进入一次独立 seed 的短训复核；若 clean 退化，则保留手脸几何为下一候选。所有结论以冻结配对输入、训练内 calibration 权重和全量验证为准。

## 20 epoch 手运动候选的冻结结论

20 epoch 对照与候选使用相同 seed、同一 fit/calibration/validation 成员和同一随机初始化协议；评估固定读取两组第 20 轮 `last.pt`，全量 16,306 条验证样本，gamma=1 权重只来自各自训练内 calibration。

| 条件 | control | p=0.05/clean75 | candidate-control | subject bootstrap 95% CI |
|---|---:|---:|---:|---:|
| clean | 70.3974% | 69.6431% | −0.7543 pp | [−1.7535, +0.4922] |
| point dropout 5% | 48.1479% | 63.1608% | +15.0129 pp | [+13.4785, +16.4988] |
| point dropout 10% | 38.5257% | 55.2557% | +16.7300 pp | [+15.2796, +18.0747] |
| group dropout 5% | 67.1961% | 68.1958% | +0.9996 pp | [−0.0081, +1.8321] |
| group dropout 10% | 64.5468% | 66.1658% | +1.6190 pp | [+0.8532, +2.6127] |

相对 clean 的鲁棒性差分为 point 5% **+15.7672 pp**、point 10% **+17.4844 pp**、group 5% **+1.7540 pp**、group 10% **+2.3734 pp**。因此该增强确实学到了缺失恢复，但 clean 结果未证明提升，暂不进入 100 epoch 长训。

配对摘要与 20,000 次 subject bootstrap 位于 `20ep/summary.json` 和 `20ep/bootstrap.json`；两组 checkpoint 的编译/随机 scratch/数据划分检查均通过。

当前正在运行手脸几何分支 seed=2 的 8 epoch 配对筛选，用于验证已有 clean +0.693 pp 结果是否依赖 seed；若 clean 仍非负，再考虑短训扩展，仍不直接启动长训。

补充：同一两组 20 epoch checkpoint 的压缩前 anchor 缺失复评也完成。clean 差值 −0.760 pp（subject bootstrap 95% CI [−1.755,+0.483]）；anchor-pre 5% 差值 +14.608 pp（CI [+12.730,+15.754]）；anchor-pre 10% 差值 +18.294 pp（CI [+15.755,+19.623]）。这与 point/group 评估一致：增强主要改善缺失恢复，不能证明 clean 提升。

## 几何分支独立 seed 复核

手脸几何分支 seed=2 的 8 epoch 配对筛选与同协议 control 均通过强制编译和 scratch 检查，但 4,096 条固定验证前缀不支持继续：clean **−3.3447 pp**（bootstrap CI [−4.9561, −1.7578]），point dropout 5% **−1.2695 pp**（CI [−2.8809, +0.3662]），point dropout 10% **−3.8086 pp**（CI [−5.2979, −2.3438]）。这说明先前 seed=17 的 +0.693 pp clean 结果不稳定，几何分支不进入长训。

因此当前没有证据支持把任何新分支替换原 95.28% gamma=1 模型：p05/clean75 只提升缺失鲁棒性并牺牲 clean，几何分支跨 seed 不稳定。后续若继续，应优先围绕原模型的训练稳定性/数据增强强度做更小范围筛选，而不是直接 100 epoch。
