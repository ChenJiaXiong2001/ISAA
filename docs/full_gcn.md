# 最佳模型的完整 GCN 升级与独立调用

基础模型是 `body-local-hand-ctr-wide-relative`，不是最新的 routed 实验。原训练 NTU60 XSub Top-1 为 93.0823%，统一复测为 93.0946%；详见 `outputs/evaluations/baseline_vs_routed/report.md`。新完整版本尚未进行全量训练，不能沿用原模型的准确率。

新入口为 `body-local-hand-ctr-wide-relative-full`，也是 `main.py` 的默认模型。身体 22 点、双手各 21 点、面部 6 token、输入特征构造、192+96 维投影和原有逐帧平均融合沿用最佳模型。双手共享一套 CTR-GCN。没有新增路由门控。

标准 60 类配置共 6,010,434 个参数，约 601 万。

| 骨干 | 原最佳 | 完整版本 |
|---|---|---|
| 身体 CTR-GCN | 10 层，48/96/192 | 10 层，64/128/256 |
| 双手共享 CTR-GCN | 6 层，32/64/96，无时间下采样 | 10 层，64/128/256，第 5/8 层时间下采样 |
| 面部 ST-GCN | 4 层，24/24/48/48，单分区简化图 | 10 层，64/128/256，第 5/8 层时间下采样 |

CTR-GCN 使用三个空间分区、可学习基础拓扑、逐通道动态细化、两路膨胀卷积＋池化＋点卷积的四路时间模块及残差。ST-GCN 使用三个空间分区、1×1 空间投影、9 帧时间卷积、残差及每层独立的可学习边权。所有骨干包含输入 BatchNorm1d。6 通道手部输入保留相对 x/y、score、跨手距离和方向 x/y。

无效观测先清零，每个块后应用对应 mask；下采样 mask 保留 bin 内任意有效点。普通 BN 会把清零的缺失位置纳入统计，因此不是原模型的 masked BN。完整版本需要独立训练评估。全有效输入时，`forward_masked` 与该骨干的普通 `forward` 一致。

## 独立文件与接口

| 文件 | 用途 |
|---|---|
| `isaa/models/backbones/ctrgcn.py` | 可复用完整 CTR-GCN 骨干 |
| `isaa/models/backbones/stgcn.py` | 可复用完整 ST-GCN 骨干 |
| `isaa/models/backbones/common.py` | mask、下采样与参数校验 |
| `isaa/models/original_ctrgcn.py` | CTR-GCN 基础单元与独立分类模型 |
| `isaa/models/official_stgcn.py` | ST-GCN 基础单元及历史兼容接口 |
| `isaa/models/body_local_full_gcn.py` | 最佳模型的完整骨干组装 |
| `configs/full_gcn.json` | 身体、双手、面部的独立默认配置 |

`OfficialCTRGCNFeatureExtractor` 已移出融合文件，原导入路径保留兼容别名；既有 attention 模型的 state_dict 名称保持兼容。

```python
import torch
from isaa.models.backbones import CTRGCNBackbone, STGCNBackbone

# A_ctr: 3×V×V，target/source 排列；A_st: 3×V×V，source/target 排列。
# 从本项目空间图构造器得到 A_ctr 时，A_st = A_ctr.transpose(-1, -2)。
ctr = CTRGCNBackbone(A_ctr, in_channels=6)
st = STGCNBackbone(3, A_st, dropout=0.1)

# 无分类器：返回逐节点特征，便于其他模型调用。
features, mask_out = ctr.forward_masked(x, mask)  # B×C×T×V、B×1×T×V
features, time_length = ctr(x.unsqueeze(-1))     # B×C×T×V×1

# 修改宽度/时间步长后实例化；默认是标准 10 层。
small_ctr = CTRGCNBackbone(A_ctr, in_channels=6,
                          channels=(32, 64), strides=(1, 2))

# 骨干权重可以单独保存和加载；配置、节点顺序和图定义须匹配。
torch.save(ctr.state_dict(), "hand_ctr.pt")
ctr.load_state_dict(torch.load("hand_ctr.pt", weights_only=True), strict=True)
```

CTR-GCN 每层输出宽度须为 4 的倍数，中间层至少 8 通道；`channels` 与 `strides` 长度一致，步长为 1 或 2。ST-GCN 可以独立调整宽度、深度、步长和 dropout。两个类型默认均为 10 层。

## 训练与微调

专用入口为 `train_full_gcn.py`，显式选择完整 GCN 和 `configs/full_gcn.json`，默认 NTU60 XSub、64 帧、2 人、batch 32、65 轮、SGD 0.1、5 轮预热、35/55 轮衰减、seed 1。默认开启 torch.compile（reduce-overhead）；需要关闭时传入 `--no-compile`。输出保存到 `outputs/full_gcn_ntu60_xsub/<运行编号>/`，包含实际骨干配置、源码快照、日志和 best/last 权重。普通训练参数均可覆盖默认值；指定 `--init-checkpoint` 时默认读取 checkpoint 的骨干配置。

```bash
# GPU 环境完整训练。
python train_full_gcn.py --device cuda

# Windows 当前 CPU 环境只检查前向，无需数据集。
py -3.13 train_full_gcn.py --dry-run --device cpu --window-size 9 --num-workers 0

# 完整模型的同结构权重微调。
python train_full_gcn.py --init-checkpoint <full_run>/best.pt \
  --device cuda --lr 0.01 --epochs 10 --warmup-epochs 2 --lr-steps 7 9
```

```bash
# 标准完整配置；输入 raw ZIP，在线生成局部特征。
python main.py --gcn-config configs/full_gcn.json \
  --archive data/ntu60_skeletons_rtmw.zip --split xsub60 --num-classes 60 \
  --device cuda --batch-size 32 --test-batch-size 32 --epochs 65

# 先验证完整入口。
python main.py --dry-run --device cpu --window-size 9 --num-workers 0 --no-compile

# 完整版本训练出的同结构 checkpoint 可继续微调；重新建立优化器与学习率计划。
python main.py --init-checkpoint <full_run>/best.pt \
  --gcn-config configs/full_gcn.json --lr 0.01 --epochs 10 --warmup-epochs 2 --lr-steps 7 9

python tools/evaluate_best.py --checkpoint <full_run>/best.pt --device cuda
```

默认 batch 与原最佳一致，显存不足时可用 `--batch-size 8 --grad-accum-steps 4`；BN 实际使用的 batch 也随之改变。新模型没有继承旧实现里不参与前向的共享局部网络，所有注册骨干参数均参与反向传播。

编辑配置中的 `body`、`hand`、`face` 即可独立调整对应骨干，不需要修改融合文件。解析后的配置写入实验记录和 checkpoint，评估直接读取 checkpoint 配置，避免配置文件后来修改导致模型不匹配。改变深度、通道后须建立对应结构的模型；原 93.08% 最佳 checkpoint 与新骨干不兼容，入口会明确拒绝。

微调时未指定 `--gcn-config` 会自动采用 checkpoint 内的骨干配置；显式提供配置可调整兼容结构的 dropout 等设置。

单独训练或微调某个分支时，可在 Python 中使用 `model.body_encoder`、`model.hand_encoder`、`model.face_encoder`，分别设置 `requires_grad_` 并为待训练分支创建优化器。冻结骨干同时需要 `eval()` 固定 BN 统计，且在调用整体 `model.train()` 后重新设置冻结骨干的 eval 状态。
