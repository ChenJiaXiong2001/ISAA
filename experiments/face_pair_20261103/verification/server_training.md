# Face–Hand / Face–Body 服务器验证记录

记录日期：2026-11-03。已完成隔离同步、服务器回归测试、真实 CUDA 编译前后向及检查点验证。正式 100 epoch 训练已完成首轮训练、校准、验证及保存，并继续运行。

## 实现与数据协议

新增 `best-face-pair`，在当前 pooled wide-relative baseline 上共享 Body、Hand、Face 编码器，保留双手距离/方向六通道输入：

- Hand–Face：48 维 hand 与 48 维 face 拼接，独立分类。
- Body–Face：192 维 body 与 48 维 face 拼接，独立分类。
- 两个分类头均输出全部动作类别，并分别接受交叉熵监督；融合输出另有主损失。
- 从训练被试内部保留约 10% 被试作为校准集，不参与反向传播。默认从第 5 个 epoch 开始，根据两个头在校准集上每个动作的 top-1 准确率选择该动作使用的概率；平局或无样本时各占 0.5，再对全部类别概率归一化。推理不需要真实动作标签。
- 正式验证集不用于路由选择。新实验随机初始化，不加载旧最佳模型权重。

部署使用独立目录 `/home/xiong/Desktop/ISAA/experiments/face_pair_20261103`，同步源码 SHA256 记录于 [source_manifest.json](source_manifest.json)。

## 强制编译与实际范围

专用入口、通用入口和预设均强制启用 `torch.compile`，拒绝 `--no-compile`。训练器使用 Inductor，设置 `suppress_errors=False`、`fail_on_recompile_limit_hit=True`、`triton.cudagraphs=False`；编译不可用或失败时明确报错，不切换整模型 eager。

实际编译范围为 Body/Hand 编码器、masked pooling、两个组合特征的拼接、两个分类头和概率融合。Face `LocalSTBlock.forward` 保留原 baseline 的 eager 边界；新增神经计算没有增加 `compiler.disable`。校准准确率计数和 epoch 间路由 buffer 更新在模型前向之外；训练器在 GPU 累计准确率计数，再将结果搬到 CPU 生成报告。

## 服务器回归测试

详见 [unittests.log](unittests.log)，共 30 项：29 项通过，1 项条件跳过。

| 测试组 | 结果 | 主要覆盖 |
| --- | --- | --- |
| `test_face_pair_fusion` | 10 项通过 | 双头梯度、逐类选择、平局/缺类、缺失输入、强制编译、随机初始化约束、原骨干与双手特征保留 |
| `test_best_action_hand_body` | 13 项通过 | 原模型输入与参数、编译约束、前后向、交互层及检查点回归 |
| `test_best_weighted_100` | 6 项通过，1 项条件跳过 | 原预设、旧入口、编译约束、时间压缩协议 |

条件跳过项为 `test_cuda_compiled_training_config_checkpoint_and_evaluation`，日志原因是需要 `ISAA_TEST_DEVICE=cuda`。新实验的 CUDA 验证另行执行，结果如下。

## 独立 CUDA 编译前后向与图捕获验证

详见 [cuda_compile.json](cuda_compile.json) 和 [cuda_compile.graphs.txt](cuda_compile.graphs.txt)。在 NVIDIA RTX A6000、CUDA 12.6、PyTorch `2.14.0+cu126` 上读取真实 RTMW 数据，输入形状为 `[2, 3, 100, 133, 2]`。

- Inductor 编译前向、反向及优化器更新通过，两个分类头均实际更新。损失为 10.01044，总梯度范数为 63.44547。
- Body、Hand、Face、Hand–Face 分类头、Body–Face 分类头的梯度范数分别为 3.10735、0.95875、31.50491、34.31526、42.60123，均有限且非零。
- 记录 6 个编译图，证据确认两个新分类头、类别路由权重和概率归一化位于编译图中。
- 等权融合及更新路由两种设置下，compiled/eager 输出均通过数值一致性检查，三个输出的 top-1 全部一致。所有检查中的最大分数绝对误差约 0.004435，最大概率绝对误差约 0.00000632；使用与正式训练一致的 TF32 设置。
- 路由更新一致性和全缺失输入有限性检查通过。验证用时约 538.34 秒，CUDA 峰值分配约 4.13 GiB。

## 两轮真实 CUDA 训练验证

环境：PyTorch `2.14.0+cu126`。输出目录：

```text
/home/xiong/Desktop/ISAA/experiments/face_pair_20261103/verification/smoke_outputs/20261103-035722-64d478ad
```

详见 [smoke.log](smoke.log)。本次仅用于验证完整训练流程：fit 224 条、calibration 32 条、validation 256 条，校准被试为 `[19]`；2 个 epoch 共完成 14 次优化器更新。小数据验证提前激活路由以覆盖更新和保存流程，正式预设仍从第 5 个 epoch 开始。

- 第一个训练 batch（包括首次编译）约 343.1 秒；首轮校准约 111.6 秒。
- 第二轮训练 7 个 batch 共约 1.8 秒，记录的损失与梯度均有限；第二轮验证约 0.6 秒。
- 两轮均完成校准、验证和 `last.pt` / `best.pt` 保存。第二轮验证融合准确率为 0.0273，仅是短流程验证数值，不代表正式收敛性能。
- 日志中的 calibration 融合指标在当轮路由更新前计算；两个头的逐类正确数用于随后更新路由。正式验证在更新后执行。

## 检查点验证

[checkpoint_validation.json](checkpoint_validation.json) 的状态为 `passed`：

- `last.pt` 和 `best.pt` 均可严格加载，模型和优化器状态全部有限。
- 两者均记录 epoch 2、14 次更新、32 个路由校准样本及启用的路由状态。
- 模型包含 1,078,960 个参数，优化器状态包含 935 个参数条目。
- 保存配置验证了随机初始化、强制编译、校准与正式验证分离；fit/calibration/validation 成员及 fit/calibration 被试不存在交叉。
- 两个检查点均记录 Hand–Face 选择 1 类、Body–Face 选择 0 类、平局/未见类平均 59 类。这是小校准集产生的验证状态，不能据此推断各动作最优分支。

| 文件 | SHA256 |
| --- | --- |
| `best.pt` | `0f2d9aabc26d1ec93b258414ad9b532da19a38e072368b0cefeab08c4b653b24` |
| `last.pt` | `37af43db04921844c6c60e6a861aa8a8986e267f7bfe1c62f6f04f231f229234` |
| `config.json` | `45fe3e084e91653e04f379385f1fd6350993b40c3f9b2c2d430ec88bda57cfcf` |
| `routing_latest.json` | `8dd2006444c561a3bd328017ef2f454e38309bc08d643dcf5b67d4bc96396cc0` |

## 正式 100 epoch 训练

启动记录见 [launch_full.json](launch_full.json)：2026-11-03 04:21:11（UTC+09:00）启动，PID `353073`，GPU 为 NVIDIA RTX A6000。

```bash
/home/xiong/venvs/isaa-cu126/bin/python -s -u \
  /home/xiong/Desktop/ISAA/experiments/face_pair_20261103/code/train_best_face_pair_100.py \
  --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
  --device cuda --compile --compile-mode default \
  --batch-size 32 --test-batch-size 32 --log-interval 10 \
  --save-dir /home/xiong/Desktop/ISAA/outputs/best_face_pair_100_20261103 \
  --epochs 100 --num-workers 8
```

- 代码目录：`/home/xiong/Desktop/ISAA/experiments/face_pair_20261103/code`
- 日志：`/home/xiong/Desktop/ISAA/experiments/face_pair_20261103/verification/full.log`
- 输出父目录：`/home/xiong/Desktop/ISAA/outputs/best_face_pair_100_20261103`
- 实际运行目录：`/home/xiong/Desktop/ISAA/outputs/best_face_pair_100_20261103/20261103-042113-d2c1b13f`
- 模型文件：上述运行目录中的 `best.pt` 和 `last.pt`；逐类选择表为 `routing_latest.json` / `routing_latest.csv`。

实际配置见 [full_config.json](full_config.json)：随机初始化，100 帧 confidence-weighted 输入，SGD，batch 32，5 轮 warmup，逐 batch cosine，基础学习率 0.1。fit 36,436 条、内部 calibration 3,191 条、正式 validation 16,306 条；校准被试 `[9, 18]`。首轮训练按 `drop_last=True` 使用 36,416 条，完成 1,138 次更新。

首轮结果见 [epochs.csv](epochs.csv) 和 [full.log](full.log)：

| 阶段 | 损失 | 融合准确率 | Hand–Face 准确率 | Body–Face 准确率 | 用时 |
| --- | --- | --- | --- | --- | --- |
| train | 5.51531 | 28.98% | 22.26% | 20.79% | 309.70 秒 |
| calibration | 3.95763 | 48.54% | 34.10% | 40.71% | 166.82 秒 |
| validation | 3.99066 | 48.63% | 35.48% | 39.38% | 42.35 秒 |

正式首轮的 [检查点核验](full_epoch1_checkpoint.json) 为 `passed`：`last.pt`、`best.pt` 均可严格加载，模型与 SGD 状态有限，保存了 1,138 次更新和 935 个优化器状态条目；编译约束、随机初始化、数据划分以及路由 buffer / epoch JSON 一致性均通过。前四轮按预设对全部 60 类等权融合，第五轮开始逐类选择。

[运行状态快照](full_status.json) 采于 2026-11-03 04:30:59（UTC+09:00）：PID `353073` 正常运行，已到第 2 轮第 276/1138 步、累计 1,414 次更新。已记录的 2,024 个批次损失和 1,414 个梯度范数全部有限，最新 batch 损失 3.43805、梯度范数 4.21711；GPU 利用率 100%，显存约 13,578 MiB。该状态是采集时点记录，100 轮训练仍在进行。

可在服务器只读检查持续进度：

```bash
tail -f /home/xiong/Desktop/ISAA/experiments/face_pair_20261103/verification/full.log
```
