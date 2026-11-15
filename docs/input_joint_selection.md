# 输入关节子集选择：原池化100帧/100轮新增变体

这是基于原 `body-local-hand-ctr-wide-relative` 的独立两阶段实验。原模型、旧入口、双手距离/方向与旧实验保留。方法参考 [Feature Selection using Stochastic Gates, ICML 2020](https://proceedings.mlr.press/v119/yamada20a.html)，按关节/左右对应点分组的适配属于本项目新增，不是论文骨架结果复现。

## 两个阶段

1. `select`：从随机初始化训练全候选图，用全数据共享的可学习 Gaussian gates 学习输入选择。所有视频使用同一组参数，不需要先读当前视频来决定最终选点。训练门为 `clamp(mu + sigma * epsilon, 0, 1)`，评估为 `clamp(mu, 0, 1)`。每次前向采样一组门，人物与时间共享。门用于神经编码前和编码后读出前，完全关闭节点同步进入结构掩码，防止偏置/残差使其复活。检测 mask 单独维护，不修改原始检测置信度。前5轮不加噪声、不加稀疏惩罚，随后20轮线性增加惩罚。训练阶段保持候选维度，不宣称此阶段获得计算加速。
2. `compact`：按已导出的固定名单，物理裁剪输入、身体/手部邻接参数和输入 BN 维度。新模型所有神经参数从随机初始化训练，不读取选点模型或原最佳模型权重。只接受 `B,3,T,K,M` 的名单顺序输入，拒绝全133点回退。

新增专用入口和通用入口都强制默认 `torch.compile`，任何顺序的 `--no-compile` 均报错。编译范围包含门、原始几何、身体/双手编码、池化、融合、分类及稀疏项；仅保留原面部 `LocalSTBlock` 的已有 eager 边界。没有新增 disable，没有编译失败后整模型 eager 降级。还强制 `fail_on_recompile_limit_hit=True`，达到重编译上限时直接报错；不支持此设置的旧PyTorch运行时明确拒绝启动。`--dry-run` 也实际编译前向和反向。

## 必须计入的输入依赖

- 身体候选沿用22点。固定保护原始索引 `0,5,6,11,12,56,91,112`：肩髋参考点、原身体面部标志点及双腕。
- 两手各21点。腕点固定保留；其余20个左右对应点对共用开关，每关闭一对删除2个原始输入。这样每个保留点的双手距离/方向都只依赖明确保留的对应点，不隐藏依赖。
- 面部沿用原68点→6 token，68个原始点全部计入固定成本。本版不学习面部原始点选择。
- 原 baseline 实际使用129个不同原始点（不是133个全部都参与分类），34个可学习门控制54个可选原始点；固定成本75点。因此本版的最终输入数范围是75–129，不能宣称能减少到30点。
- compact数据读取时，在置信度auto转换、有效帧判断、时间压缩和模型几何计算之前取原始输入子集。原始133点ZIP和已按名单顺序裁剪的K点ZIP均支持；读取旧ZIP文件的存储开销仍存在。未选点的坐标、分数和有效性不影响输出。上游RTMW提取器未修改，不承诺其检测耗时下降。

## 训练与导出

```bash
# 1. 选点：100帧全序列置信度压缩，100轮，SGD+逐batch余弦
bash tools/train_input_joint_selection_server.sh --archive data/ntu60_skeletons_rtmw.zip

# 2. 从指定checkpoint导出，阈值应在实验前约定；输出文件存在时拒绝覆盖
bash tools/train_input_joint_selection_server.sh \
  --export-selection outputs/input_joint_select_100_ntu60_xsub/<run>/best.pt \
  --selection-output outputs/input_joint_subset.json

# 3. 只读取名单、不加载权重，从头训练少节点模型
bash tools/train_input_joint_selection_server.sh --stage compact \
  --subset outputs/input_joint_subset.json --archive data/ntu60_skeletons_rtmw.zip

# 同一通用入口
python -m isaa.train --model-variant best-input-joint-selection --stage select --device cuda

# 独立评估，仍启用编译；支持跨主机显式覆盖archive
bash tools/train_input_joint_selection_server.sh --evaluate <compact_run>/best.pt \
  --archive data/ntu60_skeletons_rtmw.zip
```

默认配置为 `configs/input_joint_selection.json`；CLI可覆盖门sigma、mu、稀疏权重和ramp等。稀疏项为预计开启的可选原始点数量除以54，`lambda=0.1`只是初始实验设置，不保证准确率或特定保留数量。门参数不使用权重衰减；骨干沿用0.0004。学习率0.1、5轮warmup、最低0.0001、batch32、seed1、无额外数据增强。

评估门值是连续的，`best.pt` 按连续门验证准确率选取；它不是对应硬子集的最终准确率。必须完成compact阶段才能判断子集表现。`best_selection.json` / `last_selection.json` 与对应checkpoint绑定，`selection_history.jsonl`记录每轮开关和输入数量。导出manifest保存checkpoint SHA256和epoch。阈值默认0.5，作用于确定性门值；和稀疏损失里的 `P(z>0)` 不是同一指标。

compact图使用原图在保留索引上的子矩阵，不做节点合并、不重新归一化；后续CTR动态拓扑仍可学习保留节点之间的关系。可能因删除中间关节而损失先验连接，需由实验评估。全开时有原baseline前向一致性测试。

选点100轮和compact100轮共200轮训练预算，应和原baseline的100轮预算明确区分。需要多seed、同数量随机/人工子集以及逐类准确率对照后，才能判断自动选点是否有效。

## 验证

```bash
python -m unittest tests.test_input_joint_selection -v
python tools/check_input_joint_selection_compile.py --device cuda \
  --window-size 100 --output outputs/input_joint_validation/cuda_compile.json
```

CUDA验收工具分别检查选点和真正少输入的compact模型：100帧/两人、真实Inductor编译、warmup→随机门切换、两步训练反向和优化器更新、不同batch大小验证，以及编译/eager评估数值一致性。工具里的候选子集是人工构造的验收样例，不能称为已学到的结果。

### 本次已完成的服务器CUDA验收

2026-10-20，NVIDIA RTX A6000，PyTorch `2.14.0+cu126` / CUDA 12.6：选点模型129个原始输入（身体22点、每手21点），验收compact模型102个原始输入（身体15点、每手11点），均通过100帧、两步真实编译前向/反向/优化器更新及不同batch验证。编译/eager评估输出一致，门梯度有效、梯度范数有限，`fail_on_recompile_limit_hit=True`。详见 [CUDA原始记录](input_joint_validation/cuda_compile.json)。

服务器回归共19项：18项通过、1项旧池化CUDA集成测试按环境变量跳过；新模型CUDA验收由上面的独立工具实际执行。测试覆盖输入依赖、全开对齐、导出重建、随机初始化、强制编译和新增神经计算的Dynamo捕获。另已在本地检查75输入/每手仅腕点时的前后向数值有限。

真实NTU60 ZIP的集成验收也已完成：每个划分4条样本，每阶段1轮/2次优化器更新，完成select→checkpoint导出→compact从头重训→独立编译评估。导出的102输入样例为身体17点、每手10点，和CUDA工具的人工子集不同。此验收预先指定 `mu=0.6`、导出阈值0.6、lr0.001、无warmup；不是默认100轮设置，不据此评价子集或准确率。配置、随机初始化、checkpoint往返和编译启用信息均已核对，详见 [集成验收记录](input_joint_validation/training_smoke.json)。

服务器安装目录为 `/home/xiong/Desktop/ISAA`，验收结果、日志和小样本checkpoint保存在 `outputs/input_joint_validation/20261020-input-selection/`。尚未启动100轮正式训练。

本地Windows旧action-hand-body完整训练集成测试因缺少MSVC `cl` 编译器失败；旧入口的编译开关与随机初始化检查通过，没有为该环境问题关闭编译或修改旧实验。
