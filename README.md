# ISAA

**Interpretable Skeleton-Based Action Analysis / 基于骨架的可解释动作分析**

当前实验默认只使用 **32 个真实主节点**，训练超参数对标官方 CTR-GCN NTU120 配置。
原始文件仍为 RTMW-133 ZIP；完成相对坐标归一化后，在 DataLoader worker 中抽取 32 点，
仅将 `B x 3 x T x 32 x M` 送入 GPU。输入通道为相对 x/y 和 score，最多两人。
不创建或运行面部 token 压缩、区域池化、辅助图和融合模块，全程不切换到精细分支。
此前的辅助分支结构说明保存在 [辅助实验参考](docs/auxiliary_reference.md)。

## 默认训练设定

参考本机官方仓库拷贝的
[NTU120 XSub 配置](https://github.com/Uason-Chen/CTR-GCN/blob/67d8710578b842a5d6384cd8293d627f03c6ddc1/config/nturgbd120-cross-subject/default.yaml)
及 [学习率实现](https://github.com/Uason-Chen/CTR-GCN/blob/67d8710578b842a5d6384cd8293d627f03c6ddc1/main.py)。
XSet 使用相同的优化参数。

| 项目 | 默认值 |
|---|---|
| 优化器 | SGD |
| 基础学习率 | 0.1 |
| momentum / Nesterov | 0.9 / 开启 |
| weight decay | 0.0004 |
| 预热 | 前 5 轮：0.02、0.04、0.06、0.08、0.1 |
| 学习率衰减 | 零基轮次 [35,55]，每次乘 0.1 |
| 总轮数 | 65（覆盖此前 100 轮默认值） |
| 训练 / 验证 batch | 64 / 64 |
| 帧数 / 最大人数 | 64 / 2 |
| 主干宽度 | standard：64×4、128×3、256×3 |
| 训练 drop_last | 开启；验证不丢样本 |
| 损失 / 随机种子 | 交叉熵 / 1 |

衰减沿用原代码的零基索引：控制台第 36 轮学习率为 0.01，第 56 轮为 0.001。
修改 batch 不自动缩放学习率。主干第 5、8 层时间 stride=2，64 帧降至 16 帧。

**对齐范围是上述训练超参数和主干宽度。** 本项目使用 RTMW 二维相对坐标、置信度、
32 点拓扑、masked BN 和有效位置平均，不能直接加载官方 25 点模型权重。
预处理沿用固定长度随机裁剪、验证中心裁剪及短序列补齐；尚未替换为官方
`p_interval=[0.5,1]/[0.95]` 裁剪缩放和三维随机旋转。TF32、cuDNN benchmark 仍默认开启。
这些差异会影响结果，当前实验不代表官方准确率复现。

## 运行

Python 3.10+，依赖见 `requirements.txt`；GPU 训练需要安装支持 CUDA 的 PyTorch。
将数据放在 `data/ntu120_skeletons_rtmw.zip`，在 ISAA 根目录运行：

```bash
python main.py --num-workers 8
```

默认即 `--main-only --backbone-width standard --epochs 65 --batch-size 64 --test-batch-size 64`。
原目录中已有 ZIP 时，无需复制：

```bash
python main.py --archive ../HumanActionParticipationModeling/data/ntu120_skeletons_rtmw.zip --num-workers 8
```

模型前向检查和小规模训练检查：

```bash
python main.py --dry-run --device cpu --window-size 8
python main.py --epochs 1 --max-samples 4 --batch-size 2 --test-batch-size 2 --window-size 8 --num-workers 0 --device cpu --save-dir outputs/smoke
```

若样本数小于训练 batch，使用 `--no-drop-last` 或减小 batch。
显存不足时可降低 `--batch-size`，但这将偏离官方 batch 设定。
保留原双行进度与时间戳汇总，实际记录目录会在启动时打印。

## 全程记录

每次运行创建独立目录，默认：

```text
outputs/rtmw_ctr32_only_v5_standard_sgd/xsub120/<时间戳-运行ID>/
  config.json       实际参数、软件环境、骨架索引、数据路径/大小/修改时间和参考来源
  source.zip        本次运行使用的 Python 源码和依赖清单
  console.log       控制台摘要、模型信息、警告与 Python 异常
  batches.jsonl     每个已完成的训练/验证 batch
  epochs.csv        每轮两阶段指标、学习率、最佳准确率/轮次和累计耗时
  status.json       running/completed/interrupted/failed，已完成轮次和最近 batch
  last.pt           最后完成轮次的模型、优化器和调度参数
  best.pt           验证 Top1 最佳的 checkpoint
```

`--save-dir` 指定父目录，其下仍创建独立运行子目录，重复启动不会覆盖旧实验。
批记录包含 epoch、phase、step、该阶段累计 global_step、样本数、loss、Top1、Top5、lr、
数据等待和计算/指标同步耗时。文件中的准确率为 0～1，控制台显示百分比。
每批行缓冲写出，逐轮 CSV 刷新；JSON 状态及 checkpoint 使用临时文件替换。
进度条原地刷新不写入日志；每批明细由 JSONL 保存，避免控制字符污染文件。
指标逐批传回 CPU 会产生同步开销，吞吐统计包含此开销。

正常完成记录为 completed；Ctrl+C 为 interrupted；Python 异常为 failed，并保留已写出的记录。
进程被强制终止或机器断电时无法执行结束处理，status 可能仍为 running，应查看 JSONL 和 checkpoint。
当前入口仍从头训练，没有断点续训参数。不会为每一轮另存一份权重，也不复制原始训练数据。
`--dry-run` 创建配置、源码和日志记录，但没有真实训练指标或 checkpoint。

## 模型接口与辅助对照

```python
from isaa.models.rtmw_local_ctr import RTMWLocalCTR

model = RTMWLocalCTR(num_classes=120, backbone_width="standard", main_only=True)
result = model(x, valid_frame_mask, return_node_features=True)
# x: B x 3 x T x 32 x M，32 点顺序必须与 model.main_joint_indices 一致
# 也接受已归一化的 133 点输入，并在主干入口抽取 32 点
# 返回 logits、node_features、node_mask、node_indices、time_indices
# 所有 auxiliary_* 和 regional_* 字段为 None
```

纯主干模式不能开启辅助分支。模型构造函数保留旧默认值以兼容原辅助实验调用，
当前训练入口明确传入 `main_only=True` 和 `backbone_width="standard"`。
加载 checkpoint 时应使用其中的 `model_config` 构造模型，再加载 `model` 字段。

后续需要辅助对照时，可显式使用 `--no-main-only --aux-start-epoch 0`；
它仍采用本页 SGD 训练设定。只指定 `--aux-start-epoch` 不会取消纯主干模式。
旧 v4 辅助权重不能严格加载到 v5 纯主干，compact 和 standard 也需要各自训练。
逐节点特征和 CTR 边权用于后续分析，不直接等于节点贡献或因果解释。

## 验证

```bash
python -m unittest tests.test_experiment -v
python -m unittest discover -s tests -v
```

测试包含学习率边界、日志持久化和运行隔离，以及 32 点输入、辅助模块缺席、
缺失点、模型前向/反向、checkpoint 和逐批指标汇总。
2026-09-12：两个无 PyTorch 依赖的测试通过，65 轮学习率与本机官方实现逐轮一致，
参数解析、语法及 diff 格式检查通过。
本机当前缺少 PyTorch，模型测试在导入 torch 时中止，需在安装依赖的环境运行；
未进行新实验的完整训练。
