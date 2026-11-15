# V4：身体、双手与六个面部 token 的输入条件门控

独立入口：`train_best_input_selection_v4_100.py`，服务器脚本：
`tools/train_input_joint_selection_v4_server.sh`。保留 baseline 和 v1/v2/v3，
不加载已有 best 权重，从随机初始化重新训练。

## 门控含义

每个输入片段预测 40 个开关，片段内人物和帧共享：
14 个身体关节、20 组左右手对应点、6 个面部 token。
面部沿用原池化分组，原始 RTMW 零基索引为 23–33、34–44、45–55、
56–66、67–78、79–90；不擅自将其命名为眼、鼻等解剖区域。
每个面部 token 独立开关，不是对 68 个原始面部点逐点设置开关。

路由只读取片段观测及运动描述符，不读取动作标签；标签用于分类损失
与事后分类别统计。没有固定 K，分类损失与打开门控的期望成本共同学习
保留数量。每个身体门成本 1、手部对成本 2、面部 token 成本 1，最大成本
为 60。日志 `active_optional_cost` / `retained_mean` 是加权门控成本，
不是剩余原始输入点数；`face_tokens` 单独记录 0–6 个 token 的开启数量。
`sampled_*` 是训练随机门控统计，其他统计是确定性阈值统计。

输入仍有 129 个候选原始点，其中 68 个面部观测用于池化及轻量路由。
面部开关施加在原 face 编码器输出的六个 token 上；编码仍执行，关闭的
输出 token 不参与最终读出，但此前图消息仍可能传递原始信息。此版本是
输入条件的特征门控，不是无输入依赖的真实节点删除，不承诺训练加速。
双手距离/方向保留。身体和手部沿用输入与输出门控，检测有效性与门值分离。

## 训练与编译

前 5 轮全开且冻结路由器；6–10 轮逐步引入门控和噪声；第 10 轮起
使用随机二值 straight-through 门控。稀疏系数在第 6–25 轮升至 0.01，
主干 SGD，路由器 Adam、学习率 0.001。验证同时保留连续门控评估，
每 5 轮执行确定性二值验证。实际选择与准确率须训练后检查，不能保证
各类一定分化或一定优于 baseline。

修复旧版关闭节点后硬布尔 mask 截断恢复梯度的问题：池化对已经门控的
特征不再二次施加不可导的关闭 mask；面部只对编码结果门控，防止
全零输入经 ReLU 后令恢复梯度消失。测试覆盖三组全关后的分类梯度。
移除 router 输出 tanh 限幅，面部描述符索引与原池化一致并排除填充位。

默认且强制 `torch.compile`，专用入口、通用解析、实际调用和保存配置
一致；任何顺序的 `--no-compile` 都拒绝。编译覆盖描述符、MLP router、
ST 门、成本、身体/手部编码、面部输出门控、池化、融合及分类。
仅保留原模型 `LocalSTBlock.forward`（面部编码）已有 eager 边界；
没有为新增神经模块引入 disable。编译失败或重编译超限直接报错。
v4 的多进程数据加载使用 spawn，避免验证阶段 fork 已初始化的 CUDA/
Triton 状态；旧实验加载逻辑不改动。

```bash
cd /home/xiong/Desktop/ISAA
bash tools/train_input_joint_selection_v4_server.sh \
  --archive data/ntu60_skeletons_rtmw.zip \
  --epochs 100 --window-size 100 \
  --batch-size 32 --test-batch-size 32 --device cuda --compile
```

输出为 `outputs/input_joint_v4_dynamic_select_100_ntu60_xsub/<run>`。
`console.log`、`batches.jsonl`、`epochs.csv`、`status.json` 记录进度；
`latest_gate_usage.json`、`selection_history.jsonl` 记录 40 门全局/分类别
开启率、分类别准确率与样本数、全开/全关比例和分组计数。无样本类别的
零值只是占位。`best.pt` 按连续验证选取，`best_hard.pt` 按过渡完成后的
二值验证选取，`best_sparse.pt` 还要求爬坡结束并实际关门；保留 `last.pt`。

服务器验证：
```bash
/home/xiong/venvs/isaa-cu126/bin/python -s -m unittest tests.test_input_joint_selection_v4 -v
/home/xiong/venvs/isaa-cu126/bin/python -s tools/check_input_joint_selection_v4_compile.py \
  --output docs/input_joint_v4_validation/cuda_compile.json
```
