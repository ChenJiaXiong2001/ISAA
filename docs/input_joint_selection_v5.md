# V5：修复全关塌缩的输入条件门控实验

独立入口 `train_best_input_selection_v5_100.py`，服务器入口
`tools/train_input_joint_selection_v5_server.sh`。从随机初始化训练，不读取
baseline/v4 权重；保留所有旧模型、日志、入口和双手距离/方向。

## 40 个门及输入范围

14 个身体门、20 组双手对应点门、6 个面部 token 门，逐输入片段预测，
同片段帧和人物共享。路由不接收动作标签，没有固定 K，也不强制每类
使用不同门。68 个原始面部点仍池化/编码成六个 token，门控编码后的
读出；不宣称真实输入删除、完整信息隔离或训练加速。候选输入仍为 129 点。

## 针对 v4 的修改

1. 取消全开下限与学习门值之间的渐变混合，避免罚的是低概率、用的仍
   是保底特征。训练采用 Bernoulli(p) 二值前向，反向使用概率的 ST
   替代梯度；成本精确使用相同 p 的期望加权成本/60。
2. 描述符与全局上下文各加 LayerNorm；路由 logit 前向限制在 [-4,4]，
   使用 straight-through clamp，配合越界平方惩罚 .01 避免原始输出漂移。
   不使用会在限幅外截断导数的普通 clamp，也不使用饱和 tanh。
3. p=.05+.90*sigmoid(logit/1.0)。训练随机开门概率保留探索；评估
   使用 p>0.5 的确定性门，允许全部关闭或全部开启，不设最低 K。
   连续验证只作辅助，硬门验证才对应二值部署行为。
4. 固定以有效观测节点数作为池化分母，不随门值缩小，避免小幅特征被
   小分母再次放大，减弱分类效用与稀疏代价之间的失配。
5. 路由 Adam 学习率降至 .0001，梯度裁剪 .5；主干维持原 SGD 和余弦
   学习率。分类与稀疏梯度在每轮首批分别记录，不为诊断运行第二个模型。

## 默认训练阶段

- 第 1–5 轮：全开、路由冻结；稀疏权重零。
- 第 6–15 轮：真实随机二值门，先仅靠分类效用学习路由；稀疏权重零。
- 第 16–45 轮：归一化成本权重从 0 增至 .1。等效每单位原始加权成本
  的最大系数为 .1/60，不能与 v4 未归一化的 .01 直接比较。
- 第 46–100 轮：维持上述稀疏强度。每 5 轮及最后一轮作硬门验证。

这是明确记录的初始实验设置，不保证消除所有全开/全关结果。全关从
路由启动即报告，不再等到稀疏爬坡结束；best_sparse 排除几乎全关结果。
如果无用门最终都被关闭但硬门准确率不受损，也必须通过对照判断价值，
不能靠强制数量伪造动态选点。

## 日志与检查

`active_optional_cost` 为确定性 p>0.5 的加权计数，最大 60；
`sampled_cost` 为训练实际随机二值加权计数；`p_mean` 是平均开启概率。
身体/手对/面部门成本分别 1/2/1。`sparsity_penalty` 为归一化期望成本，
范围 [0,1]。全开预热时精确为 1，路由被冻结。
记录每类 40 门开启率、每类准确率/样本数、全开/全关比例、raw logit
越界惩罚、有效 logit 范围、分类与稀疏梯度和所有配置。

保留 `best.pt`（软验证）、`best_hard.pt`（二值验证）、`best_sparse.pt`
（稀疏爬坡完成且未塌缩的二值验证）及 `last.pt`。提供 forced_mode
诊断用全开/全关控制，默认 0 不干预学习，不作为第二个训练模型。

强制 torch.compile：预设/通用解析均拒绝 --no-compile，编译失败或重编译
超限直接失败。新增 LayerNorm、MLP、ST 门、成本、越界惩罚及池化都参与
编译，身体/手编码、融合/分类也编译；仅保留原 face LocalSTBlock eager
边界。数据加载采用 spawn，避免 fork 继承 CUDA/Triton 状态。
分项梯度诊断需要保留计算图，因此显式设置 AOT donated_buffer=False；
此设置只关闭反向缓冲区捐赠优化，不关闭 torch.compile，并记录在运行配置中。

```bash
cd /home/xiong/Desktop/ISAA
bash tools/train_input_joint_selection_v5_server.sh \
  --archive data/ntu60_skeletons_rtmw.zip \
  --epochs 100 --window-size 100 \
  --batch-size 32 --test-batch-size 32 --device cuda --compile
```

输出为 `outputs/input_joint_v5_dynamic_select_100_ntu60_xsub/<run>`。
单测 `tests/test_input_joint_selection_v5.py` 包含强制编译、200 步持续稀疏
压力后梯度可恢复、二值采样/期望成本一致、三组全关分类梯度、缺测输入、
模型保存与读取。CUDA 实测工具 `tools/check_input_joint_selection_v5_compile.py`。
