# 单手有效归一化：真实数据 CPU 流程烟测

使用原 seed2 checkpoint 和其训练校准清单的前 2 个成员，执行干净及左右手各 5 帧、20 帧、全序列的 7 种条件；每个条件均对同批输入运行原模型和独立候选，BF 输出一致且所有输出有限。该烟测使用显式 CPU eager，只验证流程，不是 CUDA 编译验证或准确率证据。

checkpoint SHA256 `25037c807a45e8ec647f89b58557f63c9a68c89ce28a62949bc5b99ce6e47432`；完整校准清单 SHA256 `9b56a5b88334edfb127434815ba4bf09a2fa9249efc27be4423c65bd8783aba7`；前 2 个成员 SHA256 `b9b4f149168595c54760e9169443594b5e74571161c5bf3a6cd6a8b613ff3ad9`。7 组输入+mask 哈希、池化位置统计与模型输出见 `summary.json` 和 `predictions.csv`。干净条件两样本的人物槽位 0 总计 182 个两手有效池化帧，5 帧遮挡生成 10 个单手帧，20 帧生成 40 个，全序列生成 182 个；当前手编码器 6 层 stride 均为 1，mask 不扩张。样本极少，这些数字只证明流程覆盖相应边界。

服务器输出 `/home/xiong/Desktop/ISAA/outputs/single_hand_pool_cpu_smoke_20261106`，复现命令见协议 `../single_hand_pool_protocol_20261106.md`，添加 `--cpu-smoke --smoke-samples 2 --num-workers 0`。后续完整评估前需在 A6000 上先完成真实 Inductor 编译烟测。
