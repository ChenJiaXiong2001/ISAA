# ISAA 近三周研究进展报告

**统计周期：2026年10月19日—11月8日**  
**项目：ISAA 全身二维骨架动作识别**

## 一、阶段概览

本阶段研究从基础模型和输入组合比较，转向围绕 `best-face-pair` 基线开展准确率融合、长序列处理、缺失鲁棒性、动态输入选择和选择性预测研究。核心工作特点是：所有新增训练方案均从随机初始化开始，保留原模型、旧入口和旧实验；训练入口默认并强制使用 `torch.compile`，拒绝 `--no-compile`、整模型 eager fallback 和 checkpoint warm start。

阶段性判断已经比较明确：当前可保留的是 0.5 强度的 geometric fusion、长度阈值为 100 帧的 temporal policy，以及检测到 anchor 缺失时的 max5 插值修复；derived geometry dropout 等新增训练候选未达到基线水平，应停止继续投入。

## 二、按时间线整理

### 10月19日—10月22日：基线整理与输入研究扩展

完成原池化 100 帧模型与单流 CTR-GCN 的对比材料、133 节点全身骨架研究报告及发表汇报版本，进一步固定了当前研究的 baseline、数据说明和模型结构表述。随后连续开展 input joint selection v3、v4、v5：从身体关节、双手对应点和面部 token 的片段级动态路由出发，引入随机二值前向、straight-through 梯度、归一化期望成本、LayerNorm、logit 限制和较低路由学习率，以降低全开/全关塌缩风险。

这一阶段完成了机制、解析、梯度和编译边界验证，但尚未获得独立测试集上的精度或真实采集成本收益，因此只能认定为可继续验证的研究方向。

### 10月26日—11月3日：实时摄像头与 face-pair 融合

补充 camera pooling/实时处理相关实现和日志，建立从骨架检测到时序输入的工程验证材料。随后形成 `best-face-pair` 路线：共享 Body、左右手和 Face 编码器，分别建立 Hand–Face 与 Body–Face 分类头，再用校准集上的类别级规则进行概率融合；同时建立 raw-XY 变体，检验加入原始坐标后的影响。

该路线保留了单流 baseline，并能分别观察身体、手部和面部证据，但校准样本主要来自少数受试者，且验证集被重复用于筛查，当前结果不能等同于盲测泛化。

### 11月4日—11月6日：几何、缺失与遮挡鲁棒性

围绕 face-pair checkpoint 系统比较 raw-XY、center/scale geometry、mask-aware geometry、时间压缩、插值修复、置信度平滑和几何融合权重。配对评估显示，p10 缺失增强模型的 clean 准确率为 94.1065%，低于 baseline 的 94.6461%（-0.5397 个百分点），但在压缩后身体点缺失 5%/10% 时分别提升 2.7045/6.2615 个百分点，说明缺失增强有鲁棒性价值，同时存在 clean 代价。

固定 checkpoint 的 arithmetic probability pool 与 normalized weighted-geometric pool 进行 0.5/0.5 混合后，在 16,306 条、20 个 subject 的 formal validation 上，clean 提升 +0.3557pp；point dropout 5% 提升 +0.675pp；group dropout 5%/10% 分别提升 +0.497pp/+0.442pp。point dropout 10% 的提升仅 +0.092pp，区间跨零，因此不应表述为稳定收益。

同时，时间阈值策略对 raw frame length >100 的样本使用 confidence-weighted temporal compression，在 formal validation 上 clean 提升 +1.1959pp，point 5%/10% 分别提升 +1.4780/+1.4902pp。该结果仍基于重复使用的 validation，需要独立受试者验证。

### 11月6日—11月7日：候选训练、结构审计与决策收敛

对 derived geometry dropout 启动了 100 epoch 的 random-scratch 训练。训练完成且工程审计通过：best epoch 87，best validation 93.8857%，编译、数据分区、routing、strict load 和有限值检查均通过。但 paired evaluator 显示其相对 stable control 在 clean、point dropout 和 group dropout 条件下均为负，clean 下降约 0.83–0.91pp，point 5%/10% 下降约 1.86/2.13pp，因此淘汰该训练候选。

max5 anchor interpolation 只在有效帧、前后端点存在、无 invalid frame 且 gap 不超过 5 时插值，不外推。正式 validation clean 约中性（-0.0123pp），但 synthetic precompression anchor dropout 5% 时提升 +3.8513pp；与 geometric fusion 结合后，5%/10% 缺失分别提升约 +0.435/+0.429pp。该收益只说明对合成缺失协议有效，尚不能代表真实摄像头故障。

低置信度拒识实验显示，拒识 10% 样本时覆盖率 90.52%，接受样本准确率 98.62%，捕获 713 个错误中的约 510 个，适合作为人工复核调度信号。以融合最大概率乘以分支 JSD 的排序规则在相同复核预算下劣于最大概率基线，已淘汰。

## 三、当前结果汇总

| 方向 | 结果 | 当前决策 |
|---|---|---|
| 0.5 geometric fusion | clean +0.3557pp；group 5%/10% +0.497/+0.442pp | 保留为默认融合策略 |
| threshold=100 temporal policy | clean +1.1959pp；缺失条件同步提升 | 保留为长序列策略 |
| max5 anchor interpolation | 合成 anchor 缺失 5% 时 +3.8513pp | 作为检测到缺失时的修复策略 |
| p10 缺失增强训练 | clean -0.5397pp，但压缩后身体缺失明显提升 | 值得独立受试者复核 |
| derived geometry dropout scratch | clean 约 -0.83pp，point 缺失下降更明显 | 淘汰 |
| 最大概率拒识 | 10% 拒识时接受准确率 98.62% | 作为复核调度基线 |
| JSD 分歧排序 | 同预算少捕获错误 | 淘汰 |
| 时间平滑、raw-XY、center/scale、rank fusion 等 | 无稳定正证据 | 暂停 |

## 四、工程与可复现性

新增入口保存 compile 配置、数据和成员清单哈希、逐样本预测、分层汇总和审计结果。当前训练配置明确记录编译范围：新增路由、门控、池化、融合和分类模块纳入编译；原 Face `LocalSTBlock` 的既有 eager 边界保留。CPU smoke、回归测试、checkpoint checker、paired evaluator、xset 复核和延迟 benchmark 已完成；真实 CUDA 编译前向/反向验证仍应作为服务器侧必做检查。

## 五、限制与下一步

本阶段大量结果来自同一 NTU60 XSub validation，且被重复用于模型筛查和策略选择；xset 交集结果不是独立测试。缺失、遮挡和时间压缩均为合成代理，不能直接替代真实采集故障。部分长序列、类别和受试者分层样本较少，局部增益不能直接推广为整体规律。

下一阶段应固定 checkpoint 选择规则，在未参与调参的新受试者上同时复核 clean、point/group dropout 和 temporal policy；对 v5 动态门控只保留硬门、全开对照和独立测试；以最大概率拒识作为基线重新设计校准集与测试集完全分离的人工复核实验；补充 face-pair/raw-XY 的跨受试者结果后再决定是否进入正式模型比较。
