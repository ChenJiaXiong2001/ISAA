# 原最佳模型：100帧置信度加权压缩

入口：`train_best_weighted_100.py`。使用原最佳 `BodyLocalHandCTRWideRelativeFusion`，保留身体CTR-GCN、共享双手CTR-GCN、面部ST块、池化、融合分类器及双手距离/方向特征。它与 `train_best_no_pool_weighted_100.py` 的区别是保留原最佳池化结构。

默认NTU60 XSub，65轮，batch/test batch 32，SGD学习率0.1，warmup 5轮，衰减节点35/55，seed 1。独立保存到 `outputs/best_weighted_100_ntu60_xsub/` 下的运行目录；从随机初始化训练，不加载原最佳权重。

长于100帧的序列按连续区间划成100组，每帧恰好进入一组。每个关节、每个人物分别计算 `sum(score * coordinate) / sum(score)`，输出置信度为组内普通平均。全零置信度的坐标为零。不足100帧保留原帧顺序、坐标和节奏，只补齐并标记无效帧。训练和验证使用相同压缩，无随机时间裁剪。这里的权重来自关键点置信度，不是学习得到的注意力。

```bash
python train_best_weighted_100.py --device cuda
```

强制 `torch.compile`；禁止 `--no-compile`。编译范围包括原身体/手部编码器、池化、融合和分类器；面部 `LocalSTBlock` 保留原有eager边界。编译错误直接报错，不静默退回整模型eager。`--dry-run` 仅验证eager形状，不代表编译训练验证。

```bash
ISAA_TEST_DEVICE=cuda python -m unittest tests.test_best_weighted_100 -v
python tools/evaluate_best.py --checkpoint outputs/best_weighted_100_ntu60_xsub/<运行目录>/best.pt --device cuda
```

评估自动读取检查点中的100帧加权压缩参数。旧64帧入口、模型和已训练权重保持独立；新协议的准确率须待重新训练后实测。
