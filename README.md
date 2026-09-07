# ISAA

**Interpretable Skeleton-Based Action Analysis / 基于骨架的可解释动作分析**

本目录是独立的最新研究工程：32 个真实主节点运行 CTR-GCN 主干，133 点固定骨架提供轻量辅助特征。
模型、数据处理、骨架布局和日志组件均在本目录内，不需要导入原项目。
内置骨架布局仅保留 RTMW-133，已移除原生 NTU/Kinect 25 点布局及注册。
NTU120 在此仅表示动作数据集，读取的是其 RTMW-133 提取结果。
此前 HAPM 和本轮早期实验继续保留在 `HumanActionParticipationModeling` 原目录中供参考。

## 模型

- 输入：`B x 3 x T x 133 x M`，通道为相对 `x,y,score`，支持单人和多人。
- 预处理：以肩髋中心平移、按躯干尺度归一化，保留缺失点及有效帧 mask。
- 主干输入：直接从原始 133 点中抽取 32 个真实主节点，进行按节点/通道的输入归一化。
- 空间主干：10 层三分支 CTR，动态拓扑始终为 32x32；每层保留可学习基础图、共享 alpha、分支求和后归一化和图卷积残差。
- 时间主干：仅处理 32 点。每层四分支先降到 C/4 通道，包含核为 5、dilation 为 1/2 的时间卷积、最大池化和 1x1 投影，最后拼接回 C 通道。
- 辅助分支：133 点运行两层固定 self/inward/outward 图卷积，通道为 3->16->16；按真实边索引聚合并保留原归一化权重，固定图不训练。
- 细节时序：16 通道 depthwise 时间卷积（核 5）+ 1x1 通道混合 + masked BN + 残差/ReLU。保留原时间长度和全部 133 点，不运行 CTR 或高通道多尺度时间模块。
- 区域汇总：使用既有的 32 区域归属，每帧计算有效节点均值和区域内可学习 softmax 加权汇总，拼接为 32 通道区域特征；区域之间不混合，不沿时间平均。均值按有效节点数归一化，空区域输出零。
- 早期融合：区域特征投影到首个主干 block 的输出通道（默认 64），以可学习系数（初始 0.1）加到该 block 输出，只融合一次且早于首次时间下采样。真实主节点特征保留为主路径；中心缺失但区域有有效细节点时，融合后的主干位置仍有效。
- 分类：仅对融合后的 32 点主干做有效位置全局均值和分类。133 点分支通过该分类损失训练，没有分类前全局辅助向量或额外分类损失。

默认训练 **100 轮**，从第 1 轮同时启用两路。
`--aux-start-epoch 5` 可在前 5 轮仅训练主干，第 6 轮启用辅助分支；旧参数 `--fine-start-epoch` 是同义别名。
`--auxiliary-channels` 设置辅助宽度，默认 16。主干通道表为 `64,64,64,64,128,128,128,256,256,256`。
第 5、8 层时间 stride=2，64 帧变为 32、16 帧；节点数始终是 32。奇数长度向上取整，mask 按时间分箱取有效性并集。
主干采用 CTR-GCN 的空间/时间模块组织；RTMW 输入、跨人共享的 masked 输入归一化、有效位置池化以及辅助分支属于项目适配，不能直接加载官方权重。
辅助分支保留逐节点时序表示供导出，区域汇总仍是有损压缩，不保证无损保留所有输入信息。当前仍使用 Adam 和原项目数据增强，不宣称复现官方训练结果。
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
python main.py --batch-size 32 --num-workers 8 --epochs 100
```

先检查真实数据的双分支训练、验证和保存链路：

```bash
python main.py --archive ../HumanActionParticipationModeling/data/ntu120_skeletons_rtmw.zip --epochs 1 --aux-start-epoch 0 --max-samples 4 --batch-size 2 --window-size 8 --num-workers 0 --device cpu --save-dir outputs/smoke_ctr32_aux133
```

`python -m isaa.train` 和 `python isaa/train.py` 也可启动同一训练程序。
数据及输出的相对路径均以 ISAA 根目录为基准；默认输出为 `outputs/rtmw_ctr32_region133_v3/xsub120/last.pt`、`best.pt`。
checkpoint 的 architecture 为 `rtmw_ctr32_region133_v3`；结构与旧版不兼容，需要重新训练，默认目录避免覆盖旧实验。
保留原双行进度条、时间戳、Top1/Top5/loss、吞吐和数据等待时间输出。
CUDA 默认启用 TF32、固定尺寸卷积优化、常驻 worker、预取和非阻塞传输。
入口每次从头训练；同一输出目录的 checkpoint 会被更新，目前没有断点续训参数。
`--max-samples` 仅用于链路检查，其结果不代表全量数据准确率。

## 分析接口

```python
from isaa.models.rtmw_local_ctr import RTMWLocalCTR

model = RTMWLocalCTR(num_classes=120)
model.set_fine_enabled(True)  # Enable the auxiliary branch (default).
result = model(x, valid_frame_mask, return_node_features=True)
# node_features: B x M x 256 x ceil(T/4) x 32
# node_indices: original RTMW indices of the 32 main joints
# time_indices: 0, 4, 8, ... (input-frame anchors, not isolated-frame features)
# auxiliary_node_features: B x M x 16 x T x 133
# auxiliary_node_mask / auxiliary_node_indices: original observation mask / indices
# regional_features: B x M x 32 x T x 32 (mean + attention, default auxiliary width 16)
# regional_mask: B x M x 1 x T x 32
```

`node_mask` 与下采样后的主干特征对应；融合后包含有效区域细节提供的位置，不再仅表示主节点原始观测。
关闭辅助分支时所有 `auxiliary_node_*`、`regional_features` 和 `regional_mask` 字段为 None。
`set_fine_enabled` 只切换辅助分支，主干节点始终为 32。可导出两路不同分辨率的表示用于后续动作分析。
当前实现完成分类主干及特征接口；
特征值和 CTR 边权本身不等于节点贡献或因果解释，可解释性评估仍属于后续研究。
归一化会移除整体平移与尺度变化，需结合动作类别评估其影响。

## 目录

```text
main.py                      主研究入口
isaa/train.py                训练、验证、checkpoint
isaa/models/rtmw_local_ctr.py 32 点 CTR-GCN 主干与 133 点辅助分支
isaa/data/                   ZIP 数据读取与相对坐标预处理
isaa/layouts/                原 RTMW 主节点、区域归属和骨架连接
isaa/graph/                  固定图构建
isaa/utils/                  控制台、随机种子和运行组件
tests/                       模型与训练链路测试
outputs/                     训练输出
```

## 当前验证记录

本次结构更新：`compileall` 语法检查和 `git diff --check` 通过。
测试已覆盖稀疏固定图与稠密计算的数值/梯度等价性、图恢复、辅助时间顺序、区域隔离、
有效节点计数、空区域及 attention 梯度、缺失中心的区域补充、一次早期融合、32 点主干、
奇数帧下采样、缺失点及 padding、辅助开关、特征索引、checkpoint 往返和训练指标。
当前 Python 环境缺少 PyTorch；安装未找到可用包，测试在导入 torch 时中止，不能视为测试通过。
尚未验证前向、反向、CUDA 吞吐或真实训练效果。

此前 1.41 G MAC 对应 v2 稠密固定图和分类前融合方案，不用于本版本。
按边聚合降低理论运算量，但 scatter/gather 在 GPU 上的实际吞吐取决于尺寸和访存，需要实测。
