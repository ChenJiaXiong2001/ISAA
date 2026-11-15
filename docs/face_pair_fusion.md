# Hand–Face / Body–Face 双分类器

新增 `best-face-pair`，编码器沿用当前 pooled wide-relative baseline：Body CTR-GCN、左右手共享 Hand CTR-GCN 和 Face ST-GCN。手部保留 torso-relative xy、置信度以及双手距离/方向的六通道输入。Face 编码共享一次，分别参与两个分类器。

- Hand–Face：左右手共享编码后汇总，48 维手部特征与 48 维脸部特征拼接，独立输出全部动作类别。
- Body–Face：192 维身体特征与 48 维脸部特征拼接，独立输出全部动作类别。
- 每个分类头各自 softmax，按类别路由表选择概率，再统一归一化。最终预测为最高类别概率。推理不需要真实类别。

路由表仅由训练协议内部保留的约 10% 被试校准数据生成，这部分不参与反向传播。默认从第 5 个 epoch 起，比较两个分类器在每个动作上的校准准确率，选择较高的一侧；相等或缺少样本时各占 0.5。正式验证数据不参与路由估计。两个头各自接受交叉熵监督，并加入融合输出的主损失，避免未被当前路由选中的头停止学习。

新实验从随机初始化开始，拒绝基线权重初始化。旧模型、旧入口和其他实验保留。

## 数值稳定性

新方案默认基础学习率为 `0.025`，梯度范数裁剪为 `5`；通用入口同样应用此默认值，并拒绝非正或非有限的裁剪阈值。保留原三项损失的权重：融合 CE + 0.5 × Hand–Face CE + 0.5 × Body–Face CE，不额外归一化。每个 batch 记录损失分项、分类头分数幅值与裁剪前梯度范数；出现非有限损失或梯度时保存诊断现场并报错停止。

融合在 float32 对数域计算两个头的概率与平均值，再按 `1/0/0.5` 选择并全局归一化。普通数值下与原概率融合等价，同时保留概率下溢时的梯度。两头最高概率类别都被路由舍弃时，归一化前先减最大值，避免大的公共负偏置造成精度损失。该计算仍在 `torch.compile` 的编译范围内。

## 编译范围

训练默认并强制 `torch.compile`，专用和通用入口均拒绝 `--no-compile`。实际编译 Body/Hand 编码器、池化、两个组合特征、两个分类头以及概率融合。原 baseline 的 Face `LocalSTBlock.forward` 编译边界保留，因此该 Face block 内部仍为 eager。新增神经计算不增加 `compiler.disable`。编译不可用或出错时明确失败，不回退整模型 eager。

```bash
bash tools/train_face_pair_server.sh --device cuda --epochs 100
python -s tools/check_face_pair_compile.py --output docs/face_pair_validation/cuda_compile.json
```

CUDA 验证工具读取真实 RTMW 训练数据，运行 Inductor 前向、反向和优化器更新，检查 body/hand/face 与两个分类头梯度有限且非零，并记录两个新分类头和概率融合实际位于编译图中的证据。另检查 eval 一致性、路由表更新以及全缺失输入。

验证沿用正式训练的 TF32 设置。编译与 eager 结果分别检查原始分数和 softmax 概率，并要求每个输出的 top-1 决策一致；报告保存最大/平均分数差值及最大概率差值，便于复核实际数值误差。
