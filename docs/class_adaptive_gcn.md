# 按动作学习通道宽度的完整 GCN

入口为 `train_class_adaptive_gcn.py`，配置为 `configs/class_adaptive_gcn.json`。原有最佳模型与 `train_full_gcn.py` 保持独立。本版本以约 93.08% 的 wide-relative 模型作为冻结蒸馏教师，学生使用完整 GCN。尚未全量训练，不继承教师的准确率。

## 模型与通道分配

身体 CTR-GCN、左右手共享 CTR-GCN、面部 ST-GCN 都保留 10 层。每个分支分为 4/3/3 层的三个阶段，最大宽度为 64/128/256，以 8 通道为一个组。每个动作有身体、手、面部共九个独立宽度，无需提前划分“小、中、大”动作类别。

训练和调用流程为：

1. **共享权重超网训练，默认 40 epoch。** 每批依次运行最大宽度、最小宽度、随机宽度，平均三个分类/蒸馏损失，同时训练轻量路由器。只有最大宽度更新公共 BN。
2. **类别门控学习，默认 10 epoch。** 冻结骨干，训练每类每阶段的宽度概率和路由器。软门控提供候选宽度，宽度惩罚鼓励减少通道。此阶段用标签产生软门控，内部选择的标签条件损失只用于候选学习；部署准确率始终用不读取标签的模型衡量。
3. **真实窄网络搜索。** 在内部搜索受试者上验证候选，并做有界贪心的减半/减 8 搜索。每类使用正样本以及易混淆和随机负样本，保留全部类别输出。相对最大宽度检查整体准确率、目标类别召回、误报率和 CE，所有容差均与最初最大宽度比较，防止逐步积累退化。默认至多 12 轮，每轮可能验证多个配置；不保证全局最优。
4. **实际路由微调，默认 15 epoch。** 使用预测路由选宽，训练时也不以真实类别代替路由结果。最后在内部选择受试者上校准路由温度和阈值，再固定模型做官方评估。

运行时，路由器读取身体、手和面部的时间统计量：高置信度调用预测动作的宽度；中等置信度取 top-k 类别的逐阶段宽度最大值；低置信度使用共享配置。没有独立 BN 校准的 top-k 联合配置也回退到共享配置。一个批次按实际宽度分组，每个样本只执行一次。

窄网络直接切片卷积权重，真实减少 GCN 的输入、输出及关系通道；CTR 多尺度时间卷积的四个分支均保留。最终特征补零恢复原融合接口，投影和分类器保持最大维度。软门控学习阶段仍运行最大宽度，不代表计算量已经减少。

## 开始训练

在 Ubuntu 的项目目录中，确认新的源码、配置、数据和教师 checkpoint 已放到对应位置后运行：

```bash
python train_class_adaptive_gcn.py --device cuda
```

默认数据为 `data/ntu60_skeletons_rtmw.zip`，协议 NTU60 XSub，窗口 64，最多 2 人，batch size 32；默认教师：

```text
outputs/body_local_hand_ctr_wide_crosshand_direction_zip/20261008-220235-1777a79b/best.pt
```

如教师在其他位置：

```bash
python train_class_adaptive_gcn.py --device cuda \
  --teacher-checkpoint /path/to/best.pt
```

也可用已训练的完整 GCN 初始化学生：

```bash
python train_class_adaptive_gcn.py --device cuda \
  --init-checkpoint outputs/full_gcn_ntu60_xsub/20261010-193341-e9cf699c/best.pt
```

教师仍为原最佳模型，两者角色不同。`--no-teacher` 是明确的无蒸馏消融，不会自动在缺失教师时启用。默认从头训练学生；有旧全量训练 checkpoint 的初始化需在实验报告中说明其见过哪些训练受试者。

**默认开启 `torch.compile`，模式保留原来的 `reduce-overhead`，显式关闭 CUDA Graph capture。** 最大/最小固定路径、软门控固定路径和已选宽度按单个 GCN block 编译；预处理、门控、路由及融合在 eager 中运行，避免把完整融合网络展开成巨大计算图。随机宽度和搜索候选使用 eager，避免为大量临时配置重复编译。编译配置缓存默认上限 64，超过后使用 eager；相同 block/输入输出宽度共享编译路径。

**Inductor 开启，CUDA Graph capture 始终关闭**，包括显式指定 `--compile-mode reduce-overhead` 时。这解决了 sandwich 多次反向、梯度累积及左右手共享骨干导致的 capture 缓冲覆盖问题；不能用每次 backward 前清空梯度的方法绕过，否则会丢失前面宽度的梯度。日志打印每个首次编译 block 的开始/结束以及第一批各宽度 forward/backward 进度。后向内核在第一次 backward 时编译，仍可能需要等待。

日志和 `config.json` 记录实际设置。`--no-compile` 可用于诊断，`--compile-backend eager` 仅检查编译 API；它不代表 Inductor/CUDA 性能。服务器复现多宽度和跨 microbatch 累积的 CUDA 回归检查：`python tools/check_adaptive_compile.py --device cuda --full-size`，它只用合成输入执行两个优化器更新，不启动全量训练。

阶段分开运行时，保持相同配置、随机种子、数据和受试者划分：

```bash
python train_class_adaptive_gcn.py --device cuda --stage supernet
python train_class_adaptive_gcn.py --device cuda --stage search \
  --init-checkpoint outputs/class_adaptive_gcn_ntu60_xsub/<supernet-run>/supernet.pt
python train_class_adaptive_gcn.py --device cuda --stage route \
  --init-checkpoint outputs/class_adaptive_gcn_ntu60_xsub/<search-run>/searched.pt
```

每次运行生成独立目录。阶段续接加载权重和通道表，重新建立优化器，不是中断后的逐步精确恢复。`supernet.pt` 保存已恢复的超网/门控最佳权重；`searched.pt` 保存搜索和 BN 校准结果。若门控 epoch 为 0，搜索仍能通过贪心缩宽运行。

## 数据划分与公平对照

只从官方训练部分按受试者划分：默认 80% 受试者拟合权重，剩下 20% 再分成搜索和选择受试者。拟合、搜索、选择成员名单保存在 `width_partition.json`，续接必须完全一致。BN 校准只用拟合部分的无增强样本，默认 4 个 batch，可用 `--bn-calibration-batches` 增加。

官方 val 只在模型、宽度表、BN 和阈值固定后进行最大、共享、自适应三种评估，不参与当前流程的选宽或早停。`--skip-official-evaluation` 可保存未评官方集的最终模型。教师曾在完整官方训练部分训练，因此内部搜索/选择受试者并非教师从未见过的独立集合，此事实写入 checkpoint；严格的内部泛化实验需要另训只见拟合受试者的教师。

共享宽度对照：

```bash
python train_class_adaptive_gcn.py --device cuda --experiment shared \
  --save-dir outputs/shared_adaptive_gcn_ntu60_xsub
```

对照保留相同完整骨干、训练划分和蒸馏流程，仅最终使用同一宽度配置。比较动作级召回、混淆、macro-F1 和实际耗时，再决定是否接受类别自适应。

`conv_cost_proxy` 是按宽度平方估算的相对卷积成本，**不是实测 MACs、显存或 GPU 加速比**。评估计时包含数据加载和初次编译，不能当作纯推理延迟。正式延迟实验需使用相同硬件和批量，预热后测量完整路由、分组、BN 切换和 GCN。

## 结果与调用

运行目录包含 `best.pt`、`last.pt`、阶段最佳权重、`class_widths.json/csv`、每类搜索记录、共享搜索记录、路由校准、官方评估、受试者划分、逐批/逐轮日志及源码快照。`class_widths.csv` 可直接查看每个动作的九个宽度。

详细评估（默认 compile）：

```bash
python tools/evaluate_best.py --device cuda \
  --checkpoint outputs/class_adaptive_gcn_ntu60_xsub/<run>/best.pt \
  --width-mode adaptive
```

把 `--width-mode` 改为 `shared` 或 `max` 可分别复测，输出逐类准确率、混淆矩阵和错误样本。自适应评估还输出实际宽度比例、路由置信度和回退率。

```python
from isaa.adaptive_width_training import load_adaptive_checkpoint

model = load_adaptive_checkpoint("/path/to/best.pt", device="cuda")
# x: B x 3/8 x T x 133 x M; mask: B x T
output = model(x, mask, return_routing=True)
logits, widths = output["logits"], output["widths"]
```

各 GCN 独立保存在 `branches/body_gcn.pt`、`hand_gcn.pt`、`face_gcn.pt`，包含共享完整权重、该分支类别宽度和 BN 校准统计，可单独加载、微调：

```python
from isaa.models.adaptive_branch import load_adaptive_branch

branch = load_adaptive_branch("/path/to/branches/hand_gcn.pt", device="cuda")
features, output_mask = branch(hand_features, hand_mask, class_index=0)
# features 为补零后的最大输出维度，窄 GCN 已真实执行。
branch.train()
features, output_mask = branch(hand_features, hand_mask,
                               widths=(32, 64, 128), update_bn=True)
# 用拟合数据更新该宽度的 BN 后保存；fit_batches 为 (features, node_mask) 序列。
branch.calibrate_bn(fit_batches, widths=(32, 64, 128))
branch.eval()
branch.save("hand_gcn_finetuned.pt")
```

独立分支输入已经过原模型预处理：身体 `B x 3 x T x 22`；单手 `B x 6 x T x 21`（相对 xy、置信度、跨手距离和方向）；面部 `B x 3 x T x 6`（压缩 token），节点 mask 为 `B x 1 x T x V`。这些分支本身不含动作分类器或原始 133 点预处理。独立微调后的模型须重新校准相应宽度的 BN 并更新 `normalization_stats` 再部署。

无数据 CPU 检查：

```bash
python train_class_adaptive_gcn.py --dry-run --device cpu --no-compile \
  --stage-maxima 8 16 32 --num-classes 3 --num-workers 0
python -m unittest tests.test_class_adaptive_gcn -v
```

本方案借鉴类别通道重要性、共享权重选宽和动态容量分配；关键区别是保留每个动作的九维宽度表，再由预测路由调用。搜索是有预算的经验选择，不能称为证明了每个动作的最优通道数。
