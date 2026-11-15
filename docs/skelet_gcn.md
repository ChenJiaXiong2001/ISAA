# SkeleT-GCN 复现

依据论文 [Expressive Keypoints for Skeleton-based Action Recognition via Skeleton Transformation](https://arxiv.org/abs/2406.18011)
及[作者官方代码](https://github.com/YijieYang23/SkeleT-GCN)，锁定 commit
`fb2b2b23cd9665057c11ff87b3b0289a9821e964`。
这是新增的独立模型与入口，从随机初始化训练；不修改或加载项目原最佳模型。

## 已实现范围

- RTMW/COCO-WholeBody 133 点去掉零基索引 23..90 的面部点，保留身体 17、脚 6、手 42，共 65 点。
- 官方 Grouped Mapping Framework：普通块对角重要性权重，第 5/8 块可学习语义降采样矩阵，65→27→11 点，通道分组 1→2→4。
- 10 个时空块，通道 64/64/64/64/128/128/128/256/256/256，第 5/8 块时间步幅 2。
- SkeleT-DGSTGCN、SkeleT-STGCN++、真正 SkeleT-CTRGCN，以及忠实保留公开 CTR 配置的单独选项。
- 可选 Instance Pooling：点编码、关节位置编码、Concat Pool、Group Pool；NTU 默认关闭，保留官方后期人物平均融合。
- `j`、`b`、`jm`、`bm` 四流独立训练；测试 10 clip 概率平均；4 流按 [3,3,2,2] 加权融合。
- NTU60 XSub/XView、NTU120 XSub/XSet；RTMW ZIP 和官方 PYSKL pickle 两种数据来源。

没有复现论文七个数据集的全套训练结果，也没有重新训练姿态估计器。
现有 RTMW 姿态与作者原始姿态提取流程不同；本项目结果必须作为自己的实验报告。

## 官方代码与论文差异

| 项目 | 本实现选择 |
|---|---|
| 公开代码的 expressive-keypoints 分支无法进入 | 修正 Graph 和独立骨骼预处理的布局条件；保留图边及骨骼配对 |
| 下采样后的邻接初始化 | 保留公开代码 `pinv(M) @ A @ M` 再列归一化；论文式 (4) 写作 `M.T @ A @ M` |
| 27→11 映射系数 | 保留官方文件的 0.1667 舍入值，无重新拟合 |
| 官方 `skelet_ctrgcn` 配置实际选择 `unit_gcn` | `ctrgcn-official-config` 忠实复现该配置；`ctrgcn` 使用官方 `unit_ctrgcn` 实现，明确区分 |
| 训练轮数 | 默认官方配置 24 轮 × RepeatDataset(5)，共 120 次数据集遍历；可用 `--epochs 120 --repeat 1` 对齐论文的轮数描述，两者验证频率和打乱顺序不同 |
| Batch size | 默认 128，对齐论文四卡总 batch；这里是单进程训练，BatchNorm 统计不同于官方四卡每卡 32 的非同步 BN |
| Instance Pooling 随机残差丢弃 | 保留整个 batch 一个 Bernoulli 决策的语义；改为设备端张量运算以参与编译 |

上游 Apache-2.0 license 和修改说明见 `isaa/models/skelet/LICENSE`、`NOTICE.md`。
生产训练只依赖现有 PyTorch、NumPy，不需要 MMCV、PYSKL 或 einops。

## 训练

### 作者公开 CTR 配置的独立对齐实验（2026-10-13）

旧的真正 `unit_ctrgcn` 实验与默认预设保留。需要对齐作者公开的
`configs/skelet_ctrgcn/ntu60_xsub/j.py` 时，使用新入口：

```bash
bash tools/train_skelet_official_server.sh --archive data/ntu60_skeletons_rtmw.zip --device cuda
```

独立预设 `configs/skelet_ctrgcn_official.json` 使用 `unit_gcn`、MVC BN、
100 帧、2 人、4 个逻辑 DDP rank，每 rank batch 32，总 batch 128；
每 rank 2 个非持久数据 worker，按公开 PYSKL 的 rank/worker seed 与 epoch+seed
DistributedSampler 顺序训练。24 轮 × RepeatDataset(5)，SGD/Nesterov 与
逐 iteration 余弦学习率保持官方参数。无 SyncBN，训练不广播 BN buffers；
验证前按 MMCV DistEvalHook 语义广播 rank-zero 的 BN 均值和方差。
模型随机初始化，不读取旧实验权重。训练结束自动加载 best.pt 做 10 clip
概率平均测试，写入 best_test_metrics.json 和 best_test_predictions.npz。
每轮保存 checkpoint 并保留最近 5 个，另外保存 best.pt 和 last.pt。
预设要求 WORLD_SIZE=4，直接单进程误用会报错，避免悄然改变 BN 和总 batch。

服务器只有一张 A6000，因此四个逻辑 rank 共享 cuda:0，并使用经验证支持
CUDA all-reduce 的 Gloo 代替四卡 NCCL。各 rank 使用相同参数、独立局部 BN，
DDP 对梯度求平均；共享物理 GPU 不代表将 BN 改成 batch 128。
完整神经模型仍强制 torch.compile Inductor/fullgraph，包括前向和反向；
DDP 通信、数据处理、损失与 SGD 在神经模型编译图外。

这个实验对齐公开配置的模型与训练协议，但不能称为与论文执行完全相同：
RTMW 数据、设备和软件栈不同，强制 compile 是用户固定要求，公开训练入口
未配置 compile；论文未公开其结果的随机种子。本实验固定 seed=1，并记录
上述差异。论文邻接公式为 M.T @ A @ M，公开源码为 pinv(M) @ A @ M；
这里保留源码和已有的数值对照结果，不擅自改公式。

本次验证记录：`docs/skelet_validation/official_protocol_alignment.json`
逐字段比较锁定的官方配置；`cuda_official_four_rank_compile.json` 记录四 rank、
100 帧、每 rank batch 32 的两步 CUDA 编译前向/反向/SGD。
各 rank 参数 hash 相同，训练 BN hash 不同，验证广播后 BN hash 相同。
真实 ZIP 训练/验证各 128 样本、1 轮的流程检查完成训练、checkpoint 保存/重载、
10 clip 测试和 60 类概率导出，见 `official_real_data_pipeline.json`；
该小样本准确率仅用于流程检查。
全量官方对齐 Joint 实验输出：
`/home/xiong/Desktop/ISAA/outputs/skelet_ctrgcn_official/20261013-154514-542d935b`，
启动日志 `outputs/skelet_launches/official-ctrgcn-20261013-154509.log`。
旧真正 CTRGCN 实验 `20261012-202145-8f28b9f7` 已完成且保留，
其最佳单 clip 验证准确率 95.48%，不能当作 10 clip 官方对齐实验结果。

默认预设：真正的 SkeleT-CTRGCN（`unit_ctrgcn`，MVC 数据 BatchNorm），NTU60 XSub，joint 流，100 帧、2 人、batch 128，
SGD lr=0.1、Nesterov momentum=0.9、weight decay=0.0005，逐迭代余弦衰减，无预热、无空间增强。
数据按原始全长序列先生成运动流，再采用官方三段式 UniformSample，不使用项目旧 crop/pad 或躯干归一化。
PreNormalize2D 默认图像高 1080、宽 1920；自有 ZIP 分辨率不同时必须指定相应参数。
官方 pickle 使用每个样本的 `img_shape`。

```bash
python train_skelet_gcn.py --archive data/ntu60_skeletons_rtmw.zip --device cuda
```

当前服务器推荐使用固定选择已验证 cu126 虚拟环境的启动脚本：

```bash
bash tools/train_skelet_server.sh --archive data/ntu60_skeletons_rtmw.zip --device cuda
```

脚本默认使用 `$HOME/venvs/isaa-cu126/bin/python`，支持通过 `ISAA_SKELET_PYTHON` 指定其他已验证解释器，
并完整传递参数，仍强制 `torch.compile`。这台服务器的系统 `python` 是用户目录中的 PyTorch cu130，
而 `LD_LIBRARY_PATH` 指向 cu126 虚拟环境的 cuDNN；混用会导致 `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`。
其余 `python` 示例在这台服务器执行前先运行 `source /home/xiong/venvs/isaa-cu126/bin/activate`，
或直接将 `python` 替换为 `/home/xiong/venvs/isaa-cu126/bin/python`。

另外三流分别训练，形成四个独立 checkpoint：

```bash
python train_skelet_gcn.py --archive data/ntu60_skeletons_rtmw.zip --stream b --device cuda
python train_skelet_gcn.py --archive data/ntu60_skeletons_rtmw.zip --stream jm --device cuda
python train_skelet_gcn.py --archive data/ntu60_skeletons_rtmw.zip --stream bm --device cuda
```

可选设置：

```bash
# 使用作者格式的姿态标注（split + annotations）。
python train_skelet_gcn.py --ann-file data/ntu60_expressive-keypoints.pkl --device cuda
# 论文 120 轮表述。
python train_skelet_gcn.py --epochs 120 --repeat 1 --device cuda
# NTU120 XSet。
python train_skelet_gcn.py --archive data/ntu120_skeletons_rtmw.zip --split xset120 --device cuda
# 两个其他骨干。
python train_skelet_gcn.py --backbone 'stgcn++' --device cuda
python train_skelet_gcn.py --backbone dgstgcn --device cuda
# IP 是一个独立消融，不是 NTU 原配置。
python train_skelet_gcn.py --instance-pooling --device cuda
```

显存不足可显式降低 `--batch-size`；这会改变训练协议，不标记为原设置的数值复现。
入口不接受训练初始化 checkpoint；`--evaluate` 仅用于测试。
每次新建唯一输出目录，保存参数、实际骨干、编译范围、源码快照、逐批损失、epoch 指标、运行状态、`best.pt` 和 `last.pt`。

## 编译要求

专用入口 `train_skelet_gcn.py` 和通用模块入口 `python -m isaa.skelet_training` 均默认且强制 `torch.compile`，
拒绝任何位置的 `--no-compile`，包括随后再加 `--compile` 的情况。
实际调用 Inductor、`fullgraph=True`、`suppress_errors=False`；关闭 CUDA Graph capture。
完整神经计算，包括可学习映射、图卷积、时间卷积、IP、人物读出和分类器都在编译范围内。
没有 `torch.compiler.disable` 边界，也没有整模型 eager 回退；环境不支持或编译失败直接报错。
数据解码、NumPy 特征构建、损失与 SGD 在模型图外。

`--dry-run` 也执行真实编译前向/反向，而不是仅 eager 形状检查：

```bash
python tools/check_skelet_compile.py --device cuda --window-size 100 --num-workers 0
```

本地 Windows/PyTorch 2.1 CPU 环境不支持该编译训练，使用与 CUDA/cuDNN 库匹配的 Linux PyTorch 环境。
服务器 cu126 venv 应使用 `/home/xiong/venvs/isaa-cu126/bin/python`，避免系统 cu130 Python
误用 cu126 的 `LD_LIBRARY_PATH`。

## 测试与四流融合

模型结构、数据流、协议和分辨率必须与 checkpoint 一致，入口会校验。

```bash
python train_skelet_gcn.py --evaluate /path/to/j/best.pt --stream j --num-clips 10 --predictions outputs/skelet_scores/j.npz --device cuda
python train_skelet_gcn.py --evaluate /path/to/b/best.pt --stream b --num-clips 10 --predictions outputs/skelet_scores/b.npz --device cuda
python train_skelet_gcn.py --evaluate /path/to/jm/best.pt --stream jm --num-clips 10 --predictions outputs/skelet_scores/jm.npz --device cuda
python train_skelet_gcn.py --evaluate /path/to/bm/best.pt --stream bm --num-clips 10 --predictions outputs/skelet_scores/bm.npz --device cuda
python tools/fuse_skelet_scores.py outputs/skelet_scores/j.npz outputs/skelet_scores/b.npz outputs/skelet_scores/jm.npz outputs/skelet_scores/bm.npz --output outputs/skelet_scores/4s.npz
```

融合工具按样本 ID 对齐，检查重复样本、标签、类别数和概率有效性，避免文件行顺序不同导致错配。

## 验证证据

```bash
python -m unittest tests.test_skelet_gcn tests.test_best_action_hand_body -v
# 下列源码数值对照额外需要 einops；训练入口不需要。
python tools/check_skelet_reference.py --reference-root /path/to/pinned/SkeleT-GCN --output docs/skelet_validation/reference_parity.json
```

`docs/skelet_validation/reference_parity.json` 记录 16 组对照：四种命名配置、IP 开关、train/eval。
同一权重下所有前向输出最大差为 0；输入梯度最大差小于 6e-8，并检查所有有梯度的参数。
四种输入流及短/中/长序列训练/测试采样共 8 种场景与官方预处理精确匹配。
IP 对照关闭随机丢弃以使两次计算确定；正式训练保留随机丢弃。

服务器验证在独立 `/tmp/isaa-skelet-validation-20261012` 目录执行，使用 RTX A6000、
PyTorch 2.14.0+cu126，未覆盖服务器实验文件。DGSTGCN 模型 100 帧、batch 2、2 人，
Inductor fullgraph 编译完成两步 SGD 更新，65→27 映射梯度非零且所有已有参数梯度有限。
证明文件为 `docs/skelet_validation/cuda_dgstgcn_compile.json`。
启用 Instance Pooling 后也以相同 100 帧输入完成两步真实 CUDA 编译前向/反向及 SGD 更新，
映射梯度非零、参数梯度有限，记录在 `cuda_dgstgcn_ip_compile.json`；没有排除 IP 神经路径。

切换默认骨干为真正 CTRGCN 后，在正式项目目录以真实 NTU60 RTMW ZIP、100 帧、batch 128、
随机初始化重新训练，完成至少三步 Inductor fullgraph CUDA 前向、反向和 SGD 更新。
配置与批次记录见 `cuda_ctrgcn_training_compile.json`。
该设置显存占用接近 48 GB，日志出现分配重试；更换骨干不代表已经获得吞吐提升。

另用真实 NTU60 RTMW ZIP，训练/验证各限制 4 个样本，batch 2，随机初始化训练 1 轮，
完成训练、验证和 best/last checkpoint 保存；再加载 best checkpoint 做 2 clip 测试，
成功导出 4 个唯一样本 ID、60 类且概率归一化的预测文件。
对应 `real_data_train_*` 和 `real_data_evaluation_*` 记录保存在同一验证目录。
小样本训练和测试 top1 均为 0%，仅用于流程验证。

本地新增测试 11 项通过；与原 best-action-hand-body 回归合跑共 24 项，
21 项通过、3 项因 Windows CPU 编译环境限制跳过。官方标注与 ZIP 的四流多片段预处理、
人物补零以及融合时样本顺序错位/标签不一致均有覆盖。
这验证可训练性，不等于论文准确率复现；完整四流全量训练与测试结果仍需实际实验。
