# ISAA

**Interpretable Skeleton-Based Action Analysis / 基于骨架的可解释动作分析**

ISAA 当前默认主线是 **BodyLocalFusion**，也是当前已有完整训练记录且效果最好的动作识别 Baseline。

当前默认结构：

~~~text
RTMW-133
├── 身体 22 点 → 10 层 CTR-GCN
├── 手部 42 点 + 面部 6 token → 4 层局部 ST-GCN
└── 身体特征与局部特征 → 有效位置池化 → 投影 → 拼接 → 分类
~~~

已完成的 NTU60 XSub 实验结果：

| 指标 | 结果 |
|---|---:|
| 验证 Top-1 | 89.60% |
| 验证 Top-5 | 98.82% |
| 训练 Top-1 | 99.84% |
| 参数量 | 907,690 |
| 训练轮数 | 65 |
| 训练 batch | 32 |
| 随机种子 | 1 |

训练集和验证集相差 9.24 个百分点，说明当前 Baseline 能够收敛，但存在明显过拟合。

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
→ BodyLocalFusion 动作识别
→ 身体、手部、面部和时间阶段分析
~~~

RTMDet-tiny 负责正常样本，原始质量检查失败样本使用 YOLO26-X 重新检测人物，再使用相同的 RTMW-L 提取姿态。YOLO26-X 困难样本参数为 conf=0.15、IoU=0.70、imgsz=960、crop margin=0.10。

## 当前默认训练设定

默认运行入口已经切换到 BodyLocalFusion：

| 项目 | 默认值 |
|---|---|
| 模型 | body-local |
| 数据划分 | NTU60 XSub |
| 输入节点 | RTMW-133 |
| 身体分支 | 22 点 CTR-GCN |
| 局部分支 | 手部 42 点 + 面部 6 token ST-GCN |
| 输入通道 | raw x/y/score |
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
| cuDNN | 默认关闭，匹配已记录实验 |
| torch.compile | 默认关闭，匹配已记录实验 |
| DataLoader workers | 8 |

BodyLocalFusion 不使用 32 点输入。32 点 main-only CTR-GCN 已降为显式的历史对照模型。

## 运行当前 Baseline

将 NTU60 的 RTMW-133 骨架 ZIP 命名为 `data/ntu60_skeletons_rtmw.zip` 后运行。默认数据协议是 xsub60；NTU120 数据必须显式设置 `--split xsub120` 和相应类别数：

~~~bash
python main.py
~~~

也可以显式写出当前 Baseline 的关键参数：

~~~bash
python main.py \
  --model-variant body-local \
  --feature-mode raw \
  --archive /path/to/ntu60_skeletons_rtmw.zip \
  --split xsub60 \
  --num-classes 60 \
  --batch-size 32 \
  --test-batch-size 32 \
  --epochs 65 \
  --num-workers 8 \
  --no-cudnn \
  --no-compile
~~~

如果使用已经预处理的 NumPy 数据：

~~~bash
python main.py \
  --model-variant body-local \
  --feature-mode raw \
  --npy-dir /path/to/ntu60_rtmw_npy/xsub60_raw_full \
  --split xsub60 \
  --num-classes 60 \
  --batch-size 32 \
  --test-batch-size 32 \
  --epochs 65 \
  --num-workers 8 \
  --device cuda \
  --no-cudnn \
  --no-compile
~~~

CPU smoke test：

~~~bash
python main.py \
  --model-variant body-local \
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

## 模型结构

BodyLocalFusion 的结构配置如下：

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

当前默认模型是 body-local。其他模型需要显式指定：

| 参数 | 用途 | 状态 |
|---|---|---|
| body-local | 当前最佳 BodyLocalFusion | 默认主线 |
| body-local-dropout | 分支融合 dropout 对照 | 已实现，结果待补 |
| body-local-relative | 局部支路使用躯干相对坐标 | 已实现，结果待补 |
| body-local-relative-split | 手部和面部拆分为独立局部分支 | 已实现，结果待补 |
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

当前最佳实验记录：

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
2. BodyLocalFusion 当前最佳动作识别 Baseline；
3. 32 点 main-only CTR-GCN 历史对照；
4. HAPM 和 RTMWLocalCTR 历史方案；
5. torso-cross-attn 最新但未验证方案。

“133 点固定局部图 + 32 点动态 CTR”是历史尝试，不是当前默认结构，也不是当前最佳结果。

## 相关文件

- 当前发表稿：docs/publication_manuscript.md
- 综合分析：docs/combined_project_analysis.md
- 阶段性研究报告：docs/research_report.md
- BodyLocalFusion 配置：backup_train_baseline/config.json
- BodyLocalFusion 状态：backup_train_baseline/status.json
- BodyLocalFusion 指标：backup_train_baseline/epochs.csv
