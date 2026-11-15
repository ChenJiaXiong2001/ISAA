# V3：根据输入动作片段动态选择关节

入口 `train_best_input_selection_v3_100.py`，服务器入口
`tools/train_input_joint_selection_v3_server.sh`。这是独立新增实验，从随机初始化训练；
原 baseline、v1、v2、双手距离/方向计算和旧实验均保留。不读取 baseline best 权重。

## 路由与预算

轻量路由器从每个片段的坐标、置信度、可见率、均值、标准差及运动幅度生成
34 个开关：14 个身体可选关节，以及 20 组左右手对应点。身体每开一个计 1，
手部每开一组计 2。每个输入片段有不同决策，同片段内帧和人物共享开关。
路由器不接受真实动作标签；类别标签仅用于分类损失和分动作统计。
不规定固定 K；损失为分类交叉熵 + 稀疏权重 × 期望打开的可选原始点数。

保持原池化模型的必要依赖：75 个固定原始点，其中包括全部 68 个面部点。
候选输入共 129 点，可选择关闭其余 54 点。本版本没有改为五点面部。
路由器仍需要先观察全部候选输入；主干使用每片段掩码，张量形状不压缩，
不能据此声称减少关键点采集、真实删图或提升训练速度。

## 联合训练

默认 100 轮、100 帧、batch 32、NTU60 xsub：

```bash
cd /home/xiong/Desktop/ISAA
bash tools/train_input_joint_selection_v3_server.sh \
  --archive data/ntu60_skeletons_rtmw.zip \
  --epochs 100 --window-size 100 \
  --batch-size 32 --test-batch-size 32 --device cuda --compile
```

前 5 轮全部打开，不更新路由器；6–10 轮渐变噪声与全开混合；第 10 轮起
训练前向使用随机二值门控，以 straight-through 梯度联合训练。
评估默认输出确定性连续门控，同时每 5 轮增加确定性硬门控验证。
稀疏权重在第 6–25 轮增至 .01；主干 SGD、路由器独立 Adam (.001)。
初始 logit .5 避免连续门控开始即饱和；最终效果及保留点数须由实验确认。

所有训练入口默认且强制 `torch.compile`，任何顺序的 `--no-compile` 都拒绝。
编译失败或到达重编译限制直接报错，不静默回退整模型 eager。
编译覆盖描述符、新增 MLP 路由、门控、身体/手部编码器、池化、融合与分类；
只保留原模型已有的 face LocalSTBlock eager 边界。

## 日志与结果

输出位于 `outputs/input_joint_v3_dynamic_select_100_ntu60_xsub/<运行目录>`。
`console.log`、`batches.jsonl`、`epochs.csv`、`status.json` 持久记录进度；
`selection_history.jsonl` 和 `latest_gate_usage.json` 记录每类开关频率、
每类样本数/准确率、平均及最少/最多保留点、全开比例和路由梯度。
日志 `retained_mean` 是当前确定性门控在阈值下的点数，包括固定点；
训练随机二值实际点数通过 `sampled_retained_mean` 单独记录。
类别样本数为零时对应频率/准确率占位零，不能解释为有效统计。

`best.pt` 按连续门控验证 top1 选择；`best_hard.pt` 按过渡完成后的硬门控
验证 top1 选择；`best_sparse.pt` 要求惩罚爬坡结束且确有阈值关点。
保留 `last.pt`。动态路由不存在一份通用固定子集 JSON，不能沿用 v2 导出命令。

服务器验证：

```bash
/home/xiong/venvs/isaa-cu126/bin/python -s -m unittest discover -s tests -p 'test_input_joint_selection_v3.py' -v
/home/xiong/venvs/isaa-cu126/bin/python -s tools/check_input_joint_selection_v3_compile.py \
  --output docs/input_joint_v3_validation/cuda_compile.json
```
