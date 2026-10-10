# 自动动作手部需求与身体优先条件计算

新增模型：`body-local-hand-ctr-wide-relative-class-routed`。

身体 22 点先计算一次，独立身体分类头输出动作概率。动作需求表的分数与全部类别概率加权，判断是否调用双手 CTR-GCN。低置信度默认调用；手部完全缺失时跳过。面部分支固定执行，跳过手部的样本使用经过直接分类监督的身体＋面部分类头。选中的样本先 gather，再执行手部分支；未选中的样本不执行 `_run_hand`，也不构造跨手距离/方向特征。

## 从原最佳模型训练

```bash
python main.py \
  --model-variant body-local-hand-ctr-wide-relative-class-routed \
  --init-checkpoint outputs/body_local_hand_ctr_wide_crosshand_direction_zip/20261008-220235-1777a79b/best.pt \
  --archive data/ntu60_skeletons_rtmw.zip \
  --split xsub60 --num-classes 60 --device cuda \
  --batch-size 32 --test-batch-size 32 --num-workers 8 \
  --epochs 10 --lr 0.01 --warmup-epochs 2 --lr-steps 7 9 \
  --no-compile
```

流程自动完成：

1. 导入原 wide-relative checkpoint，冻结原模型参数和 BatchNorm 统计，完整调用路径保持原模型计算方式。
2. 在训练协议内部按人物身份留出 20% 的 subjects。其余 subjects 用于训练身体分类头、身体＋面部分类头和专用面部投影，损失为直接分类监督与完整路径蒸馏。
3. 用内部留出的 subjects 选择两个辅助分类头损失之和最低的 epoch，保存 `heads_best.pt`。不根据官方验证集选择路由头。
4. 使用该 checkpoint 在内部校准集比较完整路径与身体＋面部路径，自动生成各类 `hand_requirement.json` / CSV。
5. 安装需求表，执行真正条件计算，最后评估官方验证集并保存可直接推理的 `best.pt`。

保存的需求分数是 `sigmoid((平均无手部CE - 平均完整CE - margin)/temperature)`。样本数不足的类别保守地设为需要手部。需求表未安装时也保守调用所有可用手部。

默认参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--hand-route-threshold` | 0.5 | 概率加权需求达到阈值时调用手部 |
| `--body-confidence-threshold` | 0.8 | 身体最大类别概率低于阈值时默认调用 |
| `--body-probability-temperature` | 1.0 | 身体概率温度 |
| `--routing-calibration-fraction` | 0.2 | 训练集内部按 subject 留出的比例 |
| `--hand-requirement-temperature` | 0.1 | 将损失收益映射为需求分数的温度 |
| `--hand-requirement-margin` | 0 | 要求手部提供的最小平均损失收益 |
| `--hand-requirement-min-samples` | 5 | 需求估计的最低类别样本数 |
| `--body-loss-weight` / `--no-hand-loss-weight` | 1.0 | 两个辅助头的直接分类损失 |
| `--routing-distill-weight` | 0.5 | 每个辅助头的蒸馏损失权重 |

动态样本分组使用 eager 执行；该版本不会套用旧训练的 CUDA graph 编译路径。调用比例表示动作识别网络中手部 CTR-GCN 的样本调用比例，不能直接等同于总 FLOPs 降幅或 RTMW 姿态提取节省。

## 需求估计的数据口径

需求表不会读取官方 `val` 标签。复用原 best 时，其骨干可能已经见过内部留出的 subjects，因此这里是辅助分类头的留出估计，不能声称整个模型都在未见数据上校准。该情况写入 `routing_partition.json` 和需求表 metadata：`initial_backbone_calibration_exposure=not_verified`。

严格验证可以从头训练：移除 `--init-checkpoint`，增加 `--no-class-hand-freeze-backbone`，使用原 65 轮训练计划。此时整个模型只拟合内部 fit subjects，metadata 标记骨干校准曝光为 `excluded`。跨折实验应另外组织；该入口不会自动宣称执行了多折交叉验证。

## 验证与逐类别评估

```bash
python main.py --model-variant body-local-hand-ctr-wide-relative-class-routed \
  --dry-run --device cuda --no-compile

python tools/evaluate_best.py --checkpoint <新运行目录>/best.pt \
  --device cuda --num-workers 8 --hand-mode adaptive

python tools/evaluate_best.py --checkpoint <新运行目录>/best.pt \
  --device cuda --num-workers 8 --hand-mode all

python tools/evaluate_best.py --checkpoint <新运行目录>/best.pt \
  --device cuda --num-workers 8 --hand-mode none
```

三种评估分别写入 `best_details_adaptive`、`best_details_all`、`best_details_none`。包含整体和逐类别准确率、混淆矩阵、错误样本、手部调用率、身体置信度和概率加权手部需求。

测试包含原最佳完整路径的一致性、混合 batch 仅执行选中样本、无手部路径完全跳过手部网络、身体只计算一次、辅助分类头梯度、冻结 BN、缺失手部、需求收益统计，以及小型 ZIP 的训练→生成需求表→条件评估端到端流程。
