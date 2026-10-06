# 历史辅助分支参考

本文件保留旧版 RTMWLocalCTR 辅助分支的结构说明，供复现实验和代码考古使用。当前默认主线已经切换为 BodyLocalFusion，默认训练参数和运行命令以 [主 README](../README.md) 为准。

## 当前主线

当前默认模型为 BodyLocalFusion：

~~~text
RTMW-133
→ 身体 22 点 CTR-GCN
→ 手部 42 点 + 面部 6 token 的 ST-GCN
→ 有效位置池化
→ 192 维身体投影 + 96 维局部投影
→ 288 维融合分类
~~~

已完成的 NTU60 XSub 结果为 Val Top-1 89.60%、Val Top-5 98.82%，参数量 907,690。

## 历史模型

旧 RTMWLocalCTR 使用 32 个真实主节点作为 main-only 主干，并可选地启用 133 点辅助分支。该结构曾用于：

- 检查 RTMW-133 输入和 32 点拓扑；
- 验证 masked BN、有效位置池化和日志链路；
- 探索固定局部图、主节点 CTR 和辅助区域融合；
- 进行早期计算量控制实验。

它不是当前默认模型，也没有与 BodyLocalFusion 完成统一条件下的最终公平对比。

显式运行旧模型：

~~~bash
python main.py \
  --model-variant isaa \
  --main-only \
  --feature-mode isaa \
  --node-count 32 \
  --split xsub120
~~~

启用旧版辅助分支：

~~~bash
python main.py \
  --model-variant isaa \
  --no-main-only \
  --feature-mode isaa \
  --node-count 32 \
  --split xsub120
~~~

## 旧模型接口

旧模型类仍保留在 isaa/models/rtmw_local_ctr.py 中，用于历史 checkpoint 和对照实验：

~~~python
from isaa.models.rtmw_local_ctr import RTMWLocalCTR

model = RTMWLocalCTR(num_classes=120, main_only=True)
~~~

旧模型的 32 点动态 CTR、133 点辅助分支、区域池化和 mask-aware BN 不应被写成当前 BodyLocalFusion 的结构。与旧模型相关的准确率、FLOPs、显存和吞吐结果均需以对应实验目录为准，不能自动外推到当前主线。

## 当前发表口径

正式报告中应按以下顺序说明：

1. BodyLocalFusion 是当前默认和当前最佳可复现实验；
2. 32 点 main-only CTR-GCN 是历史工程对照；
3. 133 点固定局部图加 32 点动态 CTR 是历史未验证方案；
4. torso-cross-attn 是最新代码方向，性能尚待确认。
