# 按动作校准准确率比例加权评估

2026-11-03（UTC+09:00），使用已训练完成的 epoch 83 `best.pt`，不重新训练、不更改检查点。SHA256：`aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`。

每个动作 c 的 HF/BF 准确率来自该检查点保存的训练内部校准计数（3,191 条校准样本，被试 9、18），正式验证集不参与权重计算：

```text
w_HF[c] = acc_HF[c] / (acc_HF[c] + acc_BF[c])
w_BF[c] = 1 - w_HF[c]
两者准确率之和为零时，各取 0.5。
q[c] = w_HF[c] * softmax(HF)[c] + w_BF[c] * softmax(BF)[c]
p[c] = q[c] / sum(q)
最终输出 argmax(p)。
```

通过 log_softmax、logaddexp 和 logsumexp 实现上述概率算术加权，避免数值下溢。示例：喝水 HF/BF 权重约为 50.96%/49.04%；踉跄为 38.10%/61.90%。60 类的完整权重及准确率见 [per_class_accuracy.csv](per_class_accuracy.csv)。

## 正式验证结果

相同 NTU60 xsub60 验证集，共 16,306 条样本；raw RTMW、confidence-weighted 100 帧、最多 2 人、batch 32、无增强。数据成员、顺序及 SHA256 与原训练 manifest 一致。

| 方法 | 正确数 | Top1 | Top5 | 宏 F1 |
| --- | ---: | ---: | ---: | ---: |
| HF 单头（本次） | 14,429 | 88.4889% | 98.4239% | 88.4620% |
| BF 单头（本次） | 15,128 | 92.7757% | 99.2395% | 92.8639% |
| 按动作准确率比例加权（本次） | 15,536 | **95.2778%** | 99.4726% | 95.3009% |
| 等权概率平均（先前完整评估） | 15,510 | 95.1184% | 99.4542% | 95.1459% |
| 原逐动作硬路由（先前完整评估） | 15,396 | 94.4192% | 99.2763% | 94.4432% |

比例加权较等权平均增加 26 条正确（+0.1595 个百分点），较原硬路由增加 140 条正确（+0.8586 个百分点）。与先前逐样本结果比较，比例加权修正等权平均错误 54 条，同时新增错误 28 条。三种融合结果均来自同一 epoch 83 检查点，没有为软权重重新选择最佳 epoch，也没有搜索权重或在正式验证集上重新校准。

## 核验与文件

完整 CUDA Inductor 编译评估成功，耗时约 134 秒（含首次编译）。Body/Hand 编码、池化、两个分类头、新增软融合均参加编译；Face LocalSTBlock 保留原 baseline eager 边界，没有整模型 eager 回退。原路由缓冲区和新软权重在评估前后完全一致。

三个方法均完成全部样本，输出有限，最大概率归一化误差约 3.58e-7；混淆矩阵、各类支持数和错误样本导出计数一致。本次 HF/BF Top1、Top5 与先前重评完全一致；相对训练日志仍保留 HF 少 3 条、BF 多 2 条的浮点运行差异，详见 [summary.json](summary.json)。

新增脚本为 `tools/evaluate_face_pair_accuracy_weighted.py`，SHA256：`0b28df670613e83adc5f4a3f6cf18fb7bc29cc7e12b282cd82fe509846e86902`；本地与服务器一致。仅新增评估入口，没有修改训练入口或模型文件。

```bash
/home/xiong/venvs/isaa-cu126/bin/python -s \
  /home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/code/tools/evaluate_face_pair_accuracy_weighted.py \
  --checkpoint /home/xiong/Desktop/ISAA/outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490/best.pt \
  --output-dir /home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/verification/best_epoch83_accuracy_weighted_rerun
```

输出目录必须不存在。服务器本次结果目录为 `/home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/verification/best_epoch83_accuracy_weighted`，本目录同步保存 summary、逐类权重、三份混淆矩阵、错误样本和评估日志。
