# 池化 + 100帧压缩：100轮逐批余弦训练

新增入口 `train_best_weighted_100_cosine.py`。该入口对应项目的 canonical baseline：原池化模型、双手距离/方向、100帧置信度全序列压缩、NTU60 XSub、batch/test batch 32、SGD/Nesterov、momentum 0.9、weight decay 0.0004、seed 1。已完成的统一评估为 Top-1 94.6400%（第98轮最佳；训练记录为94.6461%）。随机初始化，不接受初始化或恢复检查点。旧入口及旧实验保留。

```bash
python train_best_weighted_100_cosine.py --device cuda
```

默认100轮，前5轮每个训练批次线性预热：首批0.001、预热末批0.1。后95轮逐批单周期余弦衰减：余弦首批0.1、训练最终批0.0001，无重启。保留预热是沿用原训练设置；起始/最低学习率是新实验默认值，并非已验证的最佳值。

K取实际 `len(train_loader)`，S=100K，W=5K；当前批次零基编号u为已完成的SGD更新数。预热段 `0.001 + (0.1-0.001)*u/(W-1)`；余弦段 `0.0001 + (0.1-0.0001)/2 * (1+cos(pi*(u-W)/(S-W-1)))`。每批前设置LR，SGD成功后计数；验证不推进。固定梯度累积1，旧35/55节点不适用，配置的milestones为空。支持显式调整预算/预热供验收使用；不足两步的余弦阶段明确报错。

独立输出 `outputs/best_weighted_100_cosine_ntu60_xsub/<运行目录>/`。`batches.jsonl`记录每批实际LR和零基optimizer_update；`epochs.csv`记录train_lr_first/train_lr_last/train_lr_mean，lr列为该轮末批值。config保存调度版本、实际K、S/W和所有LR参数；checkpoint保存已完成更新数、调度配置和SGD动量。当前入口不提供恢复功能，避免将旧权重当作新实验起点。source.zip包含训练入口与调度源码。

默认强制torch.compile，专用及通用余弦入口均拒绝--no-compile，编译错误明确报错。实际编译范围为原身体/手部编码、池化、融合、分类器；原面部LocalSTBlock的eager边界保留。学习率控制在Python循环中，未新增神经计算的编译排除。

```bash
python -m unittest tests.test_batch_cosine tests.test_best_weighted_100 tests.test_best_action_hand_body -v
ISAA_TEST_DEVICE=cuda python -m unittest tests.test_batch_cosine.BatchCosineTests.test_cuda_compiled_training -v
```

服务器须等待已有训练结束，再运行真实CUDA编译前向、反向和SGD更新验收；通过后启动完整训练。验收失败时不启动完整训练，不关闭compile。
