# HF/BF 对称一致性损失短筛

## 目的

验证是否应在 Hand-Face（HF）和 Body-Face（BF）分类头之间加入对称 KL 一致性损失。该实验不是随机种子研究；`seed=17` 只用于与已有 4 epoch/1024 样本短筛控制数据划分。默认权重仍为 0，旧入口不启用该项。

## 协议

- 模型：`best-face-pair`，从随机初始化训练，不加载旧 checkpoint。
- 数据：XSub，`max_samples=1024`；拟合 890、校准 134、验证 1024。
- 训练：4 epoch，batch 32，warmup 1，`lr=0.025`，`max_grad_norm=5`。
- 新项：`0.5 * (KL(HF || BF) + KL(BF || HF))`，权重 `0.05`。
- 编译：`torch.compile` + Inductor，`compile=true`，`suppress_errors=false`；日志记录的编译范围保持身体/手部编码、masked pooling、两分类头和概率融合，保留既有 face `LocalSTBlock` 边界。

## 结果

一致性损失实验最佳为第 3 轮，验证硬路由 Top-1 **11.4258%**（117/1024）；第 4 轮为 **11.3281%**。同协议默认目标（`head=0.5, fusion=1.0, consistency=0`）最佳为 **12.5977%**，因此新项在该指标下降 **1.1719 个百分点**。这并非原准确率比例 γ=1 软融合结果。训练、校准和验证日志均为有限值，没有 NaN/Inf；KL 项从第 1 轮训练均值 1.5101 降至第 4 轮 0.8629。验证阶段 HF/BF 最大 logit 在第 4 轮分别约 106.5/42.6，仍存在分支尺度不均衡。

## 决策

当前短筛的硬路由结果没有显示优势，且不能证明该项是否削弱两个分支的互补性或是否改善 γ=1 融合。因此暂不投入 100 epoch 完整训练。源码保留为显式可选参数，默认值为 0，不改变原方案或旧实验。

证据文件：`config.json`、`status.json`、`epochs.csv`、`console.log`；汇总见 `summary.json`。
