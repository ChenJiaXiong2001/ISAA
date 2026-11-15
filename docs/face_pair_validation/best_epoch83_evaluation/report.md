# Face-pair 最新 best 完整评估

2026-11-03（UTC+09:00），服务器训练已正常完成 100 轮；最新 `best.pt` 来自 epoch 83。原文件 SHA256 为 `aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`。

## 完整验证结果

在原 NTU60 xsub60 正式验证集 16,306 条样本上重新运行，无增强、confidence-weighted 100 帧、最多 2 人、batch 32。验证集成员及顺序与训练保存的 manifest 完全一致。使用该 best 自带路由，没有用正式验证集重新校准。以下单头与等权平均均来自同一个联合训练检查点。

| 方法 | 正确数 | Top1 | Top5 | 宏 F1 | CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 逐动作路由融合 | 15,396 | 94.4192% | 99.2763% | 94.4432% | 0.21655 |
| HF：Hand + Face | 14,429 | 88.4889% | 98.4239% | 88.4620% | 0.42719 |
| BF：Body + Face | 15,128 | 92.7757% | 99.2395% | 92.8639% | 0.27198 |
| 两头全部等权概率平均 | 15,510 | 95.1184% | 99.4542% | 95.1459% | 0.20896 |

逐动作路由比 BF 单头提高 1.6436 个百分点、多判对 268 条。等权平均比逐动作路由提高 0.6991 个百分点、多判对 114 条：修正路由错误 298 条，同时新增错误 184 条。因此两个头有互补收益，但当前硬路由在这个检查点上不如等权平均。95.1184% 是对既有检查点的融合对照，并未按该方法重新选择最佳 epoch。

best 的路由为 17 类 HF、35 类 BF、8 类等权平均，来源为训练被试内部的 3,191 条校准样本（被试 9、18）。逐类选定的是该类别的概率来源；所有类别分数共同参与最终 argmax，因此某类的融合召回率不必等于所选单头的召回率。

## 主要薄弱动作

| 动作 | 路由 | 路由融合 Top1 | 等权平均 Top1 |
| --- | --- | ---: | ---: |
| A011 阅读 | HF | 77.54% | 76.81% |
| A041 打喷嚏或咳嗽 | HF | 77.66% | 81.32% |
| A048 恶心或呕吐 | HF | 81.45% | 86.55% |
| A029 玩手机或平板 | HF | 81.82% | 80.00% |
| A017 脱鞋 | HF | 85.35% | 94.14% |

例如 A017 在正式验证集上 BF 为 94.51%、HF 为 65.93%，但训练内部校准选择 HF，说明该类别的校准选择没有在正式验证集上延续。融合最多的混淆是阅读→玩手机/平板（36 条）、恶心/呕吐→胸痛（28 条）、写字→阅读（19 条）。完整 60 类结果见 [per_class_accuracy.csv](per_class_accuracy.csv)，四种方法的逐样本错误并集见 [errors.csv](errors.csv)。

## 核验及数值差异

- CUDA Inductor `torch.compile` 完整评估通过，耗时约 113 秒（含首次编译）。Body/Hand 编码、池化、两个分类头及概率融合编译；Face LocalSTBlock 保留原 baseline eager 边界，无整模型 eager 回退。
- 16,306 条样本全部完成，四种方法的混淆矩阵、逐类支持数、错误导出总数一致，输出有限且概率归一化；评估前后路由缓冲区完全相同。
- 路由融合正确数与训练记录完全一致。HF 正确数比原记录少 3 条（−0.0184 个百分点），BF 多 2 条（+0.0123 个百分点）；保留 TF32 和原编译设置，未强制确定性。这些重评数值差异在 [summary.json](summary.json) 中明确记录，未将其当作逐条预测完全复现。
- best/last 的 CPU 严格加载、模型/优化器有限性、强制编译配置及数据分离检查通过，见 [checkpoint_validation.json](checkpoint_validation.json)。本次评估没有更新权重、温度或路由，没有重新启动训练。

## 文件及复现

评估脚本：`tools/evaluate_face_pair.py`，SHA256 为 `e901624cb97ccdc2675b62cf7bff14014b86f64c691d5fea9fa7d7a030b586a0`。默认强制使用编译，无关闭编译参数；输出目录须不存在。

```bash
/home/xiong/venvs/isaa-cu126/bin/python -s \
  /home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/code/tools/evaluate_face_pair.py \
  --checkpoint /home/xiong/Desktop/ISAA/outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490/best.pt \
  --output-dir /home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/verification/best_epoch83_evaluation_rerun
```

服务器原始结果目录为 `/home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/verification/best_epoch83_evaluation`。本目录同步保留 summary、四份混淆矩阵、逐类结果、错误样本、评估日志、训练配置/状态、完整轮次和 best 路由。
