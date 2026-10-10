# 基于最佳模型的严格增量方案

本次使用 `best-hand-body-proximity`，基底结构为当前最佳模型 `body-local-hand-ctr-wide-relative`（旧模型统一复测 Top-1 93.0946%）。新实验从头独立训练，不加载原最佳权重。实现位于 `isaa/models/best_hand_body_proximity.py`，继承最佳模型的结构并直接执行其完整前向流程，再叠加新增特征。

## 原路径全部保留

- 身体、共享双手 CTR-GCN、面部 ST-GCN 的结构和参数定义全部保留，训练权重重新初始化。
- 手部原有六通道全部保留：躯干相对 x/y、score、双手之间的距离、方向 x/y。
- 原身体绝对坐标、手部/面部相对坐标、原节点池化、时间/人物平均、手脸融合与分类器全部保留。
- 原模型入口、默认模型和此前的各个实验变体继续可用。

“绝对坐标只在躯干使用”适用于新增的躯干上下文分支；严格增量要求下，不移除原身体分支已有的绝对坐标。

## 新增路径

用四个躯干点（5、6、11、12）计算公共中心及尺度，为全部 133 个节点额外生成归一化相对坐标、速度和有效性。每只手使用完整 21 点，筛选它与其他 91 个非手节点的距离。

只有靠近的节点对进入神经边编码；进入半径默认为 0.4，断开半径为 0.48，单位是归一化躯干长度。远处节点仍参加低成本距离筛选，但不计算其交互边特征。重复手腕观测之间的边排除。新增路径不重复建立双手交互边，双手距离与方向仍由完整保留的原路径提供。

近邻边使用相对位移、距离、方向、相对速度、距离变化和置信度等特征，按头脸、躯干、左右臂、左右腿、左右脚八个区域聚合，再做躯干条件门控和时间融合。另增加躯干绝对位置、整体运动、尺度的上下文特征。

最终为 `原特征 + 新交互投影 + 新躯干投影`，通过原线性分类器分类。代码通过线性分解实现这一结果，分类器 bias 只加一次。原有骨干及分类器随机初始化，新增投影零初始化；零初始化不涉及加载旧权重。训练中所有分支一起更新，训练后的准确率需要实测。

## 使用

从头独立训练，不传入 `--init-checkpoint`：

```bash
python train_best_hand_body_proximity.py --device cuda
```

仅验证运行：

```bash
python train_best_hand_body_proximity.py --dry-run --device cpu --window-size 9 --num-workers 0 --no-compile
```

Windows 本机可用 `py -3.10` 替代 `python`。脚本默认 NTU60 XSub，输出到 `outputs/best_hand_body_proximity_ntu60_xsub`，支持普通训练参数覆盖。模型接受原始三通道输入或原最佳模型的八通道缓存；新增几何始终从前三个原始通道计算。交互配置可通过 `--interaction-config configs/hand_body_proximity.json` 指定。

此入口不接受 `--gcn-config`，以保持原骨干结构。本次独立实验不使用 `--init-checkpoint`；旧有显式加载接口继续保留，但默认不启用。该接口仅加载模型权重，不恢复 optimizer 和 epoch。评估沿用 `tools/evaluate_best.py --checkpoint ...`。

`forward(..., return_interaction=True)` 返回 logits 和诊断数据，包括邻近边、距离、权重、区域门控、相对坐标，以及 `baseline_logits` 和 `added_features`。

此前的 `hand-body-proximity` 是独立的完整 GCN 相对坐标实验变体，仍保留，具体见 `docs/hand_body_proximity.md`。它不是这里的严格增量方案。

## 验证范围

测试覆盖保留原结构与六通道手部输入、可选加载接口的输出一致性、原始/缓存/缺失输入、新旧分支梯度、从头独立训练、保存、评估和再次加载。CPU 小数据训练验证不能证明全量识别率提升；完整训练、GPU 性能和编译性能仍需实际评估。
