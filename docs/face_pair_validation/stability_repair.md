# Face pair 第 7 轮故障与修复

## 初版故障证据

原实验目录 `/home/xiong/Desktop/ISAA/experiments/face_pair_20261103` 和输出 `/home/xiong/Desktop/ISAA/outputs/best_face_pair_100_20261103/20261103-042113-d2c1b13f` 保留。原进程已退出，训练状态为 `failed`；最后完整保存的是 epoch 6，最佳正式验证准确率为 epoch 4 的 71.08%。

只读诊断见 [failure_diagnosis.json](failure_diagnosis.json)。epoch 6 检查点的模型与优化器均有限，但 HF/BF 分类头权重最大绝对值已分别达到 186.58/197.81。批次记录显示：

| 位置 | batch loss | 裁剪前梯度范数 |
| --- | ---: | ---: |
| epoch 3 step 132 | 3.001 | 249.20 |
| epoch 6 step 1031 | 56.04 | 393.06 |
| epoch 7 step 444 | 28.46 | 9,061.81 |
| epoch 7 step 446 | 175,277,120 | 12,108,023 |
| epoch 7 step 1095 | 4.296 × 10²¹ | 4.593 × 10¹⁶ |
| epoch 7 step 1096 | 非有限，明确报错退出 | 未执行更新 |

原学习率调度符合配置，第 7 轮约 0.0999，无调度跳变。原新方案 `max_grad_norm=0`，实际未裁剪，异常梯度直接进入 SGD；随后至少一个辅助分类头出现巨大 CE，最终前向非有限。原融合概率有 clamp，其有限时的融合 CE 约不超过 88，无法解释记录中的巨大总损失。旧日志没有三个 CE 分项，无法仅凭它确定首个异常所属编码器；不将未经验证的输入或 BN 猜测视为根因。

## 最小修复

仅针对新增 face-pair 方案：学习率默认降为 0.025，梯度裁剪默认 5 并拒绝关闭；三项损失及权重保持。概率融合改用 log-softmax、logaddexp 和 logsumexp，归一化前移除公共最大值。新增分项损失、logits 幅值与梯度记录，非有限现场保存后仍明确失败，不跳过 batch、不掩盖 NaN。

继续强制 Inductor 编译，禁止整模型 eager 回退。Body/Hand、池化、组合、两个新分类头与融合均参与编译；Face LocalSTBlock 保留原 baseline 的 eager 边界。旧模型、双手距离/方向与旧实验不改。新修复实验从随机初始化开始，不加载失败检查点或旧最佳权重。

## 修复实验验证

修复后源码独立部署至 `/home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103`，158 个源码文件逐文件 SHA256 核验通过。旧失败实验和输出均保留。

- 服务器测试：新增 face-pair 14 项、极值数值 3 项、原 `best-action-hand-body` 13 项、原 weighted 组 7 项；36 项通过，weighted 组 1 项因专门要求 `ISAA_TEST_DEVICE=cuda` 而条件跳过。日志见 [unittests_stability.log](unittests_stability.log)。
- 真实数据 CUDA Inductor 验证见 [cuda_compile_stability.json](cuda_compile_stability.json)：编码器和两个头编译前向/反向通过，两个头更新；总梯度裁剪前范数 63.4447，裁剪后为 5.0000。捕获图包含 `log_softmax`、`logaddexp`、`logsumexp`。±10000 logits 的硬路由融合也通过 fullgraph Inductor 前后向，概率归一化，两个头梯度有限且非零。
- 真实数据两轮短流程训练通过，完整训练、校准、验证和检查点保存完成；CPU 严格加载检查通过，模型及优化器均有限。详见 [smoke_stability.log](smoke_stability.log) 和 [smoke_checkpoint_stability.json](smoke_checkpoint_stability.json)。

正式 CUDA 训练从随机初始化启动于 2026-11-03 05:21（UTC+09:00），PID `386071`。20 轮时状态仍为 `running`，`last.pt` 完整保存 epoch 20 / 22,760 次更新，`best.pt` 严格加载为 epoch 19；两者模型和优化器状态均有限，校准路由分别为 7 HF / 51 BF / 2 平均类。第 20 轮验证准确率为 83.96%，该时点最佳验证准确率为第 19 轮的 **87.54%**。详见 [epoch20_checkpoint_validation.json](epoch20_checkpoint_validation.json)、[epochs_stability.csv](epochs_stability.csv) 和 [full_config_stability.json](full_config_stability.json)。

用户要求观察到第 20 轮后，训练继续正常运行；此前状态采集已达到 epoch 24 完整保存、epoch 25 train step 560/1138，状态为 `running`，无 `failure.pt`。截至该快照，各批次三项 CE、HF/BF logits 幅值、裁剪前梯度均有限；裁剪阈值为 5。第 20 轮及更后验证准确率：epoch 20 `83.96%`、21 `86.99%`、22 `84.99%`、23 `82.79%`、24 `87.43%`。

实时查看日志：

```powershell
ssh -i C:\Users\15637\.ssh\isaa_codex xiong@100.123.254.107 "tail -f /home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/verification/full.log"
```

按 `Ctrl+C` 只退出日志查看，不影响服务器进程。正式运行目录为 `/home/xiong/Desktop/ISAA/outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490`。

## 100 轮训练完成

训练于 2026-11-03 13:55（UTC+09:00）正常完成 100 轮，状态为 `completed`，共完成 113,800 次更新。最终 `last.pt` 为 epoch 100，正式验证准确率 93.9961%；`best.pt` 为 epoch 83，记录的融合准确率 94.4192%，HF 88.5073%，BF 92.7634%。best 路由为 17 类 HF、35 类 BF、8 类等权平均。

完整训练结束后的 CPU 检查通过：best/last 均严格加载，模型及优化器全部有限，配置、强制编译策略、训练/校准/验证分离和逐轮路由记录一致。验证记录见 [checkpoint_validation.json](best_epoch83_evaluation/checkpoint_validation.json)，完整轮次见 [epochs.csv](best_epoch83_evaluation/epochs.csv)。

对 epoch 83 best 的完整 CUDA 编译重评得到融合 Top1 94.4192%，正确数与训练一致；HF 88.4889%、BF 92.7757%。同一检查点两头等权概率平均为 95.1184%，较逐动作硬路由多判对 114 条。详见 [完整评估报告](best_epoch83_evaluation/report.md)。
