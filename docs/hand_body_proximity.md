# 手部-身体邻近交互模型

严格遵循“基于最佳模型，只做新增”的版本为 `best-hand-body-proximity`，见 [严格增量方案](best_hand_body_proximity.md)。下文保留此前独立的完整 GCN 实验变体说明。

这是在现有模型基础上新增的 `hand-body-proximity` 变体。原有模型、默认模型、训练脚本和 checkpoint 格式继续保留；不选择这个变体时，行为不变。

## 输入和坐标

- 输入使用 RTMW-133 的原始三通道 `x/y/score`。
- 四个躯干点（5、6、11、12）计算公共中心和尺度。
- 所有 133 个节点都生成躯干中心化、按躯干长度缩放的相对坐标；身体、手部、面部骨干只使用这些相对坐标和 score。
- 绝对坐标只进入躯干上下文编码，用于保留整体位置和尺度变化。
- 每只手保留完整 21 点，左右手独立计算，不建立左右手之间的交互边。

## 稀疏交互

每个时间帧先用相对坐标筛选左手/右手到 91 个非手节点的距离。节点按头脸、躯干、左右臂、左右腿、左右脚八个区域组织。只有进入半径 `radius_on` 的边才编码；已有边要超过 `radius_off` 才断开，避免边界抖动。重复的手腕观测边会被排除，远距离节点不会进入神经边编码。

边特征包含两端节点特征、相对位移、距离、单位方向、相对速度、距离变化、置信度和有效性。消息先按手部节点和目标区域归一化，再由躯干条件化的区域门控融合。没有邻居时交互向量为零。

## 运行

默认配置在 `configs/hand_body_proximity.json`，专用入口会复用原训练参数：

```bash
python train_hand_body_proximity.py --device cuda
python train_hand_body_proximity.py --dry-run --device cpu --window-size 9 --num-workers 0 --no-compile
```

也可以从主入口显式选择：

```bash
python main.py --model-variant hand-body-proximity \
  --gcn-config configs/full_gcn.json \
  --interaction-config configs/hand_body_proximity.json \
  --archive data/ntu60_skeletons_rtmw.zip --split xsub60 \
  --feature-mode raw --device cuda
```

`radius_on` 和 `radius_off` 使用归一化躯干长度为 1 的坐标单位，必须满足 `0 < radius_on <= radius_off`。模型保存实际骨干和交互配置，`tools/evaluate_best.py` 会从 checkpoint 自动恢复配置。

`forward(..., return_interaction=True)` 除 logits 外还返回稀疏边索引、距离、边权、区域门控、相对坐标、节点有效性、躯干尺度以及 batch/person 元数据，可用于动作片段的交互诊断。几何筛选是动态稀疏分支，当前通过 `torch.compiler.disable` 保持兼容；完整 GPU 编译性能和识别准确率需要实际训练后再评估，不能直接继承旧模型的准确率。
