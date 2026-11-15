# 手到脸几何与运动通道短训研究

> 2026-11-06 后续复核：下文保留原始短筛及当时的建议。后续同预算配对短筛的 clean 差值为 -3.3447pp，未复现该收益，当前不支持进入长训；详见 [多方面短训筛选](../multiaspect_screen_20261106/report.md)。

## 研究问题

误差审计显示 A011/A012/A029 等手部细粒度动作混淆明显，现有 HF 分支没有显式手到脸几何或手部一阶运动通道。本候选只给 HF 局部编码增加手到最近有效 face token 的方向、归一化距离、face validity 和方向差分；BF 分支、face 分支、原 gamma=1 概率域融合和训练损失权重保持不变。

## 训练协议

- 随机初始化，未加载旧 checkpoint。
- seed 17，8 epoch，4096 条验证子集，fit/calibration/validation = 3745/351/4096。
- 原始 RTMW ZIP，100 帧，`confidence_weighted`，batch 32，warmup 1。
- HF/BF head loss 0.5，fusion loss 1.0；gamma=1 使用训练内 calibration 的 HF/BF 原准确率比例。
- A6000 实际 CUDA，PyTorch 2.14.0+cu126，强制 `torch.compile`/Inductor；checkpoint checker 验证 compile guards、随机初始化、数据拆分和有限值。

## 结果

严格相同的前 4096 条验证成员和逐条件 corruption seed 下，geometry 与既有无增强 control 的配对结果为：

| 条件 | control | geometry | 差值 | paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| clean | 57.2998% | 59.1309% | +1.8311pp | [+0.5371,+3.1250] |
| 身体点缺失 5% | 43.6768% | 50.5371% | +6.8604pp | [+5.4932,+8.2275] |
| 身体点缺失 10% | 30.5664% | 39.0869% | +8.5205pp | [+7.2021,+9.8389] |

geometry 自身相对 clean 的下降为 8.5938pp（5%）和 20.0439pp（10%）。同一 checkpoint 在完整 16,306 条验证集上的冻结结果为 clean 54.8264%、5% 缺失 45.6274%、10% 缺失 34.7418%；完整集结果用于检查子集抽样差异，不与前 4096 条配对差值混用。

## 判断

这是目前首个在同一 8 epoch/4096 配对短筛中同时提高 clean 和身体点缺失鲁棒性的训练候选，值得进入必要的完整训练验证。短训不能证明 100 epoch 后仍有同等增益，且首次 geometry 编译使 epoch1 评估显著变慢；后续完整训练应沿用相同 compile 范围、随机 scratch 和固定选择规则，不能加载该短训 checkpoint。

## 证据文件

- `summary_prefix4096.json`：geometry 三条件冻结评估。
- `paired_summary.json`、`paired_samples.csv`：同输入逐样本比较和 bootstrap。
- `summary_full.json`：完整 16,306 条验证集冻结评估。
- `checkpoint_validation.json`：checkpoint/config/compile/split 验收。
- 服务器训练目录：`/home/xiong/Desktop/ISAA/outputs/face_pair_handface_geometry_8ep4096_20261105/20261105-194403-4baa62cd`。
