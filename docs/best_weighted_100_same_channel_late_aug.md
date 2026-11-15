# 池化100帧/100轮：手部同通道 + 后期渐进增强

新增入口 `train_best_weighted_100_same_channel_late_aug.py`。原池化100轮入口和旧实验不变。模型从随机初始化训练，手部 CTR-GCN 通道与躯干完全一致：

```text
body = 48,48,48,48,96,96,96,192,192,192
hand = 48,48,48,48,96,96,96,192,192,192
```

训练默认 100 epoch、100 帧置信度加权压缩、warmup+batch cosine、SGD、weight decay `0.0007`、全局梯度裁剪 `5`。前 55 轮保持干净输入；训练准确率 EMA 连续 3 轮达到 98.5% 后启用增强，最迟第 70 轮强制启用，并在第 55–70 轮逐渐增加到：旋转 ±10°、缩放 ±10%、坐标噪声 0.003、关节丢弃 10%。增强在原始坐标上执行，派生的躯干相对和双手距离/方向特征随后重新计算。

```bash
python train_best_weighted_100_same_channel_late_aug.py --device cuda
```

训练强制使用 `torch.compile`，专用入口拒绝 `--no-compile`；编译失败直接报错。实际编译范围为躯干/双手编码、池化、融合和分类器，原面部 `LocalSTBlock` 边界保留。输出目录为 `outputs/best_weighted_100_same_channel_late_aug_ntu60_xsub/`。

观察日志：

```bash
tail -f outputs/best_weighted_100_same_channel_late_aug_ntu60_xsub/<run>/console.log
grep -E "late_augmentation|Training epoch|epoch=|checkpoint:|failed|Traceback" outputs/best_weighted_100_same_channel_late_aug_ntu60_xsub/<run>/console.log
```
