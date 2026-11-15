# 单手池化候选：epoch 83 CPU 流程验证

指定原模型 checkpoint SHA256 `aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7`；训练校准成员清单 SHA256 `68354a1f6bb0f6046b7ea720a07db89513e29ff8b7717ad36ecb028745d7cf11`。前两名样本的干净及左右手各 5 帧、20 帧、全序列遮挡均完成，输出有限且原模型与候选 BF logits 一致。各条件输入与帧 mask 的 SHA256、池化位置可见性和逐样本预测分别记录在 `summary.json` 和 `predictions.csv`。

此结果使用 CPU eager，仅验证数据流、权重加载和遮挡覆盖，不代表 CUDA 编译验证或性能证据。全序列遮挡下各有一个样本由正确变为错误；仅有两个样本，不能用于判断效果。完整校准筛选需要先完成真实 CUDA Inductor 烟测，不能把训练校准结果视为独立泛化验证。

本地 `py -3.10 -m unittest discover -s tests -p test_face_pair_single_hand_pool.py -v`：7 项通过。服务器输出目录 `/home/xiong/Desktop/ISAA/outputs/single_hand_pool_epoch83_cpu_smoke_20261106`。
