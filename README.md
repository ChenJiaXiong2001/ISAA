# ISAA

**Interpretable Skeleton-Based Action Analysis / 基于骨架的可解释动作分析**

本目录是独立的最新研究工程：133 点固定局部骨架 + 32 个真实主节点 CTR 跨部位协同。
模型、数据处理、骨架布局和日志组件均在本目录内，不需要导入原项目。
内置骨架布局仅保留 RTMW-133，已移除原生 NTU/Kinect 25 点布局及注册。
NTU120 在此仅表示动作数据集，读取的是其 RTMW-133 提取结果。
此前 HAPM 和本轮早期实验继续保留在 `HumanActionParticipationModeling` 原目录中供参考。

## 模型

- 输入：`B x 3 x T x 133 x M`，通道为相对 `x,y,score`，支持单人和多人。
- 预处理：以肩髋中心平移、按躯干尺度归一化，保留缺失点及有效帧 mask。
- 局部结构：133 点固定 self/inward/outward 骨架图，学习特征投影权重。
- 跨部位协同：直接读取原布局的 32 个中心关节，三分支 CTR 学习主节点间的通道相关连接。
- 细节点融合：根据关节所属部位回传主节点上下文，不采用区域池化。
- 时序建模：10 个图卷积、时间卷积、残差模块，最后对有效节点、帧和人求全局均值并分类。

默认训练 **100 轮**：前 **5 轮**仅使用 32 点粗阶段，第 **6 轮**启用 133 点局部结构。
通道表为 `64,64,64,96,128,128,128,192,256,256`；时间维不降采样。
模型是 CTR 思路的 RTMW 改造方案，非官方 CTR-GCN 的逐层复现。
固定骨架沿用原 RTMW 布局，面部连接仍为简化链。

## 运行

Python 3.10 或以上。在本目录安装依赖；A6000 训练环境需要支持 CUDA 的 PyTorch。

```bash
python -m pip install -r requirements.txt
python main.py --dry-run --device cpu --window-size 8
python -m unittest discover -s tests -v
```

真实训练默认读取 `data/ntu120_skeletons_rtmw.zip`，也可通过 `--archive` 指向已有数据：

```bash
python main.py --archive ../HumanActionParticipationModeling/data/ntu120_skeletons_rtmw.zip --batch-size 32 --num-workers 8
```

数据已放到本目录 `data/` 时：

```bash
python main.py --batch-size 32 --num-workers 8 --epochs 100 --fine-start-epoch 5
```

先检查真实数据的精细阶段训练、验证和保存链路：

```bash
python main.py --archive ../HumanActionParticipationModeling/data/ntu120_skeletons_rtmw.zip --epochs 1 --fine-start-epoch 0 --max-samples 4 --batch-size 2 --window-size 8 --num-workers 0 --device cpu --save-dir outputs/smoke
```

`python -m isaa.train` 和 `python isaa/train.py` 也可启动同一训练程序。
数据及输出的相对路径均以 ISAA 根目录为基准；默认输出为 `outputs/xsub120/last.pt`、`best.pt`。
保留原双行进度条、时间戳、Top1/Top5/loss、吞吐和数据等待时间输出。
CUDA 默认启用 TF32、固定尺寸卷积优化、常驻 worker、预取和非阻塞传输。
入口每次从头训练；同一输出目录的 checkpoint 会被更新，目前没有断点续训参数。
`--max-samples` 仅用于链路检查，其结果不代表全量数据准确率。

## 分析接口

```python
from isaa.models.rtmw_local_ctr import RTMWLocalCTR

model = RTMWLocalCTR(num_classes=120)
model.set_fine_enabled(True)
result = model(x, valid_frame_mask, return_node_features=True)
# logits, node_features, node_mask, node_indices
```

可导出逐节点、逐帧表示用于后续动作分析。当前实现完成分类主干及特征接口；
特征值和 CTR 边权本身不等于节点贡献或因果解释，可解释性评估仍属于后续研究。
归一化会移除整体平移与尺度变化，需结合动作类别评估其影响。

## 目录

```text
main.py                      主研究入口
isaa/train.py                训练、验证、checkpoint
isaa/models/rtmw_local_ctr.py 固定局部图与主节点 CTR 模型
isaa/data/                   ZIP 数据读取与相对坐标预处理
isaa/layouts/                原 RTMW 主节点、区域归属和骨架连接
isaa/graph/                  固定图构建
isaa/utils/                  控制台、随机种子和运行组件
tests/                       模型与训练链路测试
outputs/                     训练输出
```

## 当前验证记录

2026-09-07：通过 25 个 Python 文件的语法与独立导入路径检查、实际布局自动注册、
32 个主节点及 133 点归属检查、默认训练参数和入口转发检查；
读取原目录 ZIP 的一个真实样本，确认坐标与置信度形状符合数据接口。
本机缺少 PyTorch，安装尝试未找到可用包，模型测试在导入 torch 时中止。
尚未验证前向、反向、CUDA 吞吐或真实训练效果。
