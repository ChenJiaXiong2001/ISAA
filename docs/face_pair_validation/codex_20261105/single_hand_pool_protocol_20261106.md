# 单手缺失时的有效手数归一化：冻结校准协议

## 问题与固定候选

原 `FacePairFusion` 将两手逐帧池化结果固定除以 2，即使某帧恰好只有一只有效手。候选仅改为按该帧有效手数取均值：两手有效时和原值相同；一手有效时使用该手完整特征；两手均无效时输出零并保留无效 mask。身体、面部、双手距离/方向、分类头和 checkpoint 的 γ=1 准确率比例权重保持原样。候选为独立的 `tools/face_pair_single_hand_experiment.py`，不更改原模型或权重。

## 预设比较

仅使用指定 checkpoint 对应 manifest 的训练校准分区，按 manifest 原顺序逐样本对比原模型和候选；这批样本也用于生成 checkpoint 的融合权重，因此结果用于筛选而非独立验证。固定干净、左手和右手各三个遮挡条件：原始帧连续 5 帧、20 帧和全序列，每次只遮挡人物槽位 0 的 21 个手点，压缩前将坐标和分数置零；窗口起点由样本名的固定 SHA256 确定，同一时长的左右遮挡采用同一起点。短于窗口的序列遮挡全长。原始数据原有缺失可能导致遮挡无效，压缩可使短缺口不再体现为完全缺失；记录压缩后人物槽 0 帧数，并以各存在人物、按实际手部编码器 mask 传播后的池化位置帧数记录自然单手缺失和各遮挡覆盖。当前手部 CTR 各层 stride 均为 1，`_downsample_mask` 原样传递，时间卷积虽混合邻帧特征但不扩张 mask；若未来 stride 改变，统计按代码同样执行 max-pooling bin。

所有条件共享相同模型权重、输入张量与帧 mask；使用 checkpoint 的训练校准计数固定 γ=1 权重，不使用标签决定权重或遮挡。每种条件导出全部成员的标签、原/候选 soft、HF 和 BF 预测，核验 BF 数值保持一致，并记录输入、帧 mask 合并摘要、checkpoint 和校准清单 SHA256。比较候选相对原方案的 clean 差值、遮挡准确率差值、修正与新增错误、按受试者结果，以及遮挡相对 clean 的差中差；两名校准受试者不足以估计跨受试者泛化。不同条件并非等价于真实遮挡发生率。

冻结评估仅在 A6000 空闲且不影响其他短训时运行，使用 Inductor 编译两个前向，保留原有 `face LocalSTBlock.forward` eager 边界，编译失败直接报错。先做真实 CUDA compiled 前向对照和数值检查，完成后才运行完整校准分区；不要在该筛选阶段使用官方验证集选择候选。该实验不训练，也不加载权重用于新训练。

命令示例（指定同一运行的 checkpoint 和 manifest）：

```bash
python tools/evaluate_face_pair_single_hand_pool.py \
  --checkpoint /path/to/best.pt --manifest /path/to/split_manifest.json \
  --archive /path/to/ntu60_skeletons_rtmw.zip --output-dir /new/output/path \
  --batch-size 32 --num-workers 4 --smoke-samples 2
```

烟测通过后使用另一新输出目录，省略 `--smoke-samples` 以遍历全部训练校准成员。烟测只检查代码与编译，不用于判定准确率。

在 GPU 忙碌时，可用 `--cpu-smoke --smoke-samples 2 --num-workers 0` 显式执行至多 8 条真实校准样本的 CPU eager 流程烟测；输出明确标注此模式且不作为 CUDA 编译或性能证据。真实 GPU 烟测和完整评估始终要求 Inductor，不能静默回退。

`tests/test_face_pair_single_hand_pool.py` 检查两手、单手、零手的池化边界、遮挡范围和确定性、原模型与候选在两手有效时的 3/8 通道等价性，以及单手缺失时 BF 保持一致和输出有限。上述是评估准备，尚无模型准确率结果；固定半和也可能是分类头已经适应的分布，归一化不保证改善。
