# ISAA

**Interpretable Skeleton-Based Action Analysis / 基于骨架的可解释动作分析**

ISAA 当前默认模型是 **BodyLocalFullGCNFusion**（`body-local-hand-ctr-wide-relative-full`）：以当前最佳 wide-relative 模型为基础，把身体、双手和面部 GCN 升级为独立可复用的完整骨干。

当前最佳已训练模型是 `body-local-hand-ctr-wide-relative`：NTU60 XSub 原训练 Top-1 **93.0823%**，统一复测 **93.0946%**。最新 routed 模型复测为 92.2912%。完整版本尚未全量训练，其准确率待验证。结果依据见 `outputs/evaluations/baseline_vs_routed/report.md`。

完整版本的骨干独立存放于 `isaa/models/backbones/ctrgcn.py` 与 `isaa/models/backbones/stgcn.py`，按需导入、调整参数和微调。分支配置见 `configs/full_gcn.json`，接口及训练说明见 [完整 GCN 文档](docs/full_gcn.md)。

```bash
python train_full_gcn.py --device cuda
```

`train_full_gcn.py` 提供完整 GCN 的 NTU60 XSub 训练预设，默认开启 torch.compile（reduce-overhead），输出保存到 `outputs/full_gcn_ntu60_xsub/`。可传入 `--no-compile` 关闭编译，或通过 `--gcn-config` 调整各分支。

下方 89.60% 为历史 BodyLocalFusion 实验结果，不是当前最佳。旧模型通过显式 `--model-variant` 运行。

当前默认结构：

~~~text
RTMW-133
├── 身体 22 点 → 完整 10 层 CTR-GCN（64/128/256）
├── 双手共享 21 点 → 完整 10 层 CTR-GCN（64/128/256）
├── 面部 6 token → 完整 10 层 ST-GCN（64/128/256）
└── 身体特征与局部特征 → 有效位置池化 → 投影 → 拼接 → 分类
~~~

历史 BodyLocalFusion 的 NTU60 XSub 实验结果：

| 指标 | 结果 |
|---|---:|
| 验证 Top-1 | 89.60% |
| 验证 Top-5 | 98.82% |
| 训练 Top-1 | 99.84% |
| 参数量 | 907,690 |
| 训练轮数 | 65 |
| 训练 batch | 32 |
| 随机种子 | 1 |

训练集和验证集相差 9.24 个百分点，说明历史 BodyLocalFusion 能够收敛，但存在明显过拟合。

## 项目定位

项目包含两个阶段：

~~~text
NTU RGB+D RGB 视频
→ RTMDet-tiny 人物检测
→ RTMW-L 133 点姿态估计
→ 人物槽位匹配、人数先验和时序清洗
→ 质量检查
→ YOLO26-X 修复困难样本
→ RTMW-133 骨架序列
→ 完整 GCN 升级版动作识别
→ 身体、手部、面部和时间阶段分析
~~~

RTMDet-tiny 负责正常样本，原始质量检查失败样本使用 YOLO26-X 重新检测人物，再使用相同的 RTMW-L 提取姿态。YOLO26-X 困难样本参数为 conf=0.15、IoU=0.70、imgsz=960、crop margin=0.10。

## 当前默认训练设定

默认运行入口已经切换到完整 GCN：

| 项目 | 默认值 |
|---|---|
| 模型 | body-local-hand-ctr-wide-relative-full |
| 数据划分 | NTU60 XSub |
| 输入节点 | RTMW-133 |
| 身体分支 | 22 点完整 CTR-GCN，10 层，64/128/256 |
| 手部分支 | 双手共享 21 点完整 CTR-GCN，10 层，64/128/256 |
| 面部分支 | 6 token 完整 ST-GCN，10 层，64/128/256 |
| 输入通道 | ZIP raw x/y/score；在线生成相对坐标、跨手距离/方向 |
| 时间窗口 | 64 帧 |
| 最大人数 | 2 |
| 训练轮数 | 65 |
| 训练 / 验证 batch | 32 / 32 |
| 优化器 | SGD |
| 基础学习率 | 0.1 |
| momentum / Nesterov | 0.9 / 开启 |
| weight decay | 0.0004 |
| 预热 | 前 5 轮 |
| 学习率衰减 | 零基轮次 35、55 |
| 随机种子 | 1 |
| cuDNN | 强制开启 |
| torch.compile | 默认开启，模式 reduce-overhead；可通过 --no-compile 关闭 |
| DataLoader workers | 8 |

BodyLocalFusion 不使用 32 点输入。32 点 main-only CTR-GCN 已降为显式的历史对照模型。

## 运行默认完整 GCN

将 NTU60 的 RTMW-133 骨架 ZIP 命名为 `data/ntu60_skeletons_rtmw.zip` 后运行。默认数据协议是 xsub60；NTU120 数据必须显式设置 `--split xsub120` 和相应类别数：

~~~bash
python main.py
~~~

也可以显式写出完整 GCN 的关键参数：

~~~bash
python main.py \
  --model-variant body-local-hand-ctr-wide-relative-full \
  --feature-mode raw \
  --archive /path/to/ntu60_skeletons_rtmw.zip \
  --split xsub60 \
  --num-classes 60 \
  --batch-size 32 \
  --test-batch-size 32 \
  --epochs 65 \
  --num-workers 8
~~~

训练入口从原始 ZIP 在线读取；`--npy-dir` 已弃用。

CPU smoke test：

~~~bash
python main.py \
  --model-variant body-local-hand-ctr-wide-relative-full \
  --feature-mode raw \
  --split xsub60 \
  --num-classes 60 \
  --epochs 1 \
  --max-samples 4 \
  --batch-size 2 \
  --test-batch-size 2 \
  --window-size 8 \
  --num-workers 0 \
  --device cpu \
  --save-dir outputs/smoke_body_local
~~~

## 历史 BodyLocalFusion 模型结构

下方为 89.60% 历史模型；默认完整 GCN 的结构见 `docs/full_gcn.md`。

~~~text
身体分支：
22 个身体节点
→ CTR-GCN
→ 通道宽度 48,48,48,48,96,96,96,192,192,192
→ 第 5、8 层时间下采样
→ 有效位置平均池化
→ 192 维投影

局部分支：
42 个手部节点 + 6 个面部 token
→ ST-GCN
→ 通道宽度 24,24,48,48
→ 有效位置平均池化
→ 96 维投影

192 维身体表示 + 96 维局部表示
→ 288 维拼接
→ 60 类分类器
~~~

面部 68 个原始节点通过固定区域压缩为 6 个 token。有效节点和有效帧使用 mask 参与池化，减少 NaN、空人和低置信度点的影响。

## 可选模型与历史模型

当前默认模型是 body-local-hand-ctr-wide-relative-full。其他模型需要显式指定：

| 参数 | 用途 | 状态 |
|---|---|---|
| body-local-hand-ctr-wide-relative-full | 最佳模型的完整 GCN 升级 | 默认，待全量训练 |
| body-local-hand-ctr-wide-relative | 当前最佳已训练模型，原训练 93.0823% | 最佳结果对照 |
| body-local | 历史 BodyLocalFusion，89.60% | 历史对照 |
| body-local-dropout | 分支融合 dropout 对照 | 已实现，结果待补 |
| body-local-relative | 局部支路使用躯干相对坐标 | 已实现，结果待补 |
| body-local-relative-split | 手部和面部拆分为独立局部分支 | 已实现，结果待补 |
| body-local-hand-ctr-wide-relative-routed | 阶段与质量门控 | 复测 92.2912%，未超过原最佳 |

| body-local-hand-ctr-wide-relative-class-routed | 身体先分类，按动作需求调用手部 | 已实现，见 docs/class_hand_routing.md |

`body-local-hand-ctr-wide-relative-routed` 保持宽通道手部 CTR-GCN、躯干相对坐标以及跨手距离/方向特征，在手部和面部 ST-GCN 的逐帧融合前加入可微的阶段与质量门控，并可通过 `return_routing=True` 导出手部、面部 gate 和质量统计。原始 ZIP 输入会在线构造这些局部特征，八通道 NPY 也可直接使用。当前实现是 soft routing，局部分支仍会执行完整前向；真实的 hard conditional compute 需要后续按时间段 gather 后再测量 FLOPs 和延迟。

| 参数 | 用途 | 状态 |
|---|---|---|
| body-local-time-aug | 时间增强 | 已实现，结果待补 |
| body-local-coord-aug | 坐标噪声增强 | 已实现，结果待补 |
| body-local-full | 更大局部分支容量 | 已实现，结果待补 |
| torso-cross-attn | 躯干主导的跨分支门控 | 已实现，尚无完整结果 |
| original | 官方风格 133 点 CTR-GCN 对照 | 对照模型 |
| isaa | 旧版 ISAA 32 点 main-only/辅助分支 | 历史对照 |

运行旧的 32 点 main-only 对照：

~~~bash
python main.py \
  --model-variant isaa \
  --main-only \
  --feature-mode isaa \
  --node-count 32 \
  --split xsub120
~~~

该模型不应被称为当前最佳 Baseline。

## 实验记录

每次运行会创建独立目录：

~~~text
outputs/<model-variant>_<setting>/<split>/<timestamp-run-id>/
  config.json
  source.zip
  console.log
  batches.jsonl
  epochs.csv
  status.json
  last.pt
  best.pt
~~~

历史 BodyLocalFusion 实验记录：

- backup_train_baseline/config.json
- backup_train_baseline/status.json
- backup_train_baseline/epochs.csv

## 困难样本质量统计

在同一批 100 个原始质量检查失败样本上：

| 方法 | 通过数 | 通过率 | 平均质量分 |
|---|---:|---:|---:|
| RTMDet-tiny | 5/100 | 5% | 80.521 |
| YOLO26-X | 85/100 | 85% | 96.149 |

平均质量分是先逐样本计算 0 到 100 的启发式质量分，再求 100 个样本的算术平均：

~~~text
平均质量分 = Σ(100 个样本质量分) / 100
提升 = 96.149 - 80.521 = 15.628 分
~~~

这个分数不是动作识别准确率，也不是人工关键点误差。

## 测试

~~~bash
python -m unittest tests.test_experiment -v
python -m unittest discover -s tests -v
~~~

测试覆盖参数解析、学习率边界、训练记录隔离、模型前向/反向、mask、checkpoint 和逐批指标记录。

## 研究口径

当前结果应分开报告：

1. RTMW-133 骨架提取和困难样本修复；
2. wide-relative 当前最佳已训练模型，及尚未训练的完整 GCN 升级；
3. 32 点 main-only CTR-GCN 历史对照；
4. HAPM 和 RTMWLocalCTR 历史方案；
5. torso-cross-attn 最新但未验证方案。

“133 点固定局部图 + 32 点动态 CTR”是历史尝试。

## 相关文件

- 当前发表稿：docs/publication_manuscript.md
- 综合分析：docs/combined_project_analysis.md
- 阶段性研究报告：docs/research_report.md
- BodyLocalFusion 配置：backup_train_baseline/config.json
- BodyLocalFusion 状态：backup_train_baseline/status.json
- BodyLocalFusion 指标：backup_train_baseline/epochs.csv

## 新增手部-身体邻近交互变体

原有模型、默认入口和训练脚本保留。新增 `hand-body-proximity`：全身统一使用躯干相对坐标，绝对坐标仅用于躯干上下文，每只手保留全部 21 点，仅对附近非手节点计算交互特征。配置、诊断接口和使用方式见 [手部-身体邻近交互说明](docs/hand_body_proximity.md)。

```bash
python train_hand_body_proximity.py --device cuda
python train_hand_body_proximity.py --dry-run --device cpu --window-size 9 --num-workers 0 --no-compile
```

## 基于最佳模型只做新增

严格增量入口为 `best-hand-body-proximity`：完整保留当前最佳 `body-local-hand-ctr-wide-relative` 的六通道手部特征（含双手距离与方向）、身体/手部/面部骨干、原坐标输入和融合，仅叠加邻近手部-身体交互与躯干上下文。新实验从头独立训练，不加载原最佳权重，所有分支一起学习；结果写入单独目录 `outputs/best_hand_body_proximity_ntu60_xsub`。此前所有变体继续保留。详见 [最佳模型严格增量方案](docs/best_hand_body_proximity.md)。

```bash
python train_best_hand_body_proximity.py --device cuda
python train_best_hand_body_proximity.py --dry-run --device cpu --window-size 9 --num-workers 0 --no-compile
```
