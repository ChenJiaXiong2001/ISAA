# 基于当前最佳模型的分类前无池化方案

新变体为 `body-local-hand-ctr-wide-relative-no-pool`，实现位于 `isaa/models/best_no_pool.py`。基底是当前最佳已训练模型 `body-local-hand-ctr-wide-relative`，统一复测 NTU60 XSub Top-1 为 93.0946%。新变体从头训练，识别率尚待完整实验验证。

## 改动范围

| 部分 | 原最佳模型 | 新变体 |
|---|---|---|
| 身体骨干 | 22 点、10 层 CTR-GCN，48/96/192 | 相同 |
| 手部骨干 | 双手共享 21 点、6 层 CTR-GCN，32/64/96 | 相同 |
| 手部输入 | 相对 x/y、score、跨手距离、方向 x/y | 相同 |
| 面部骨干 | 6 token、4 层 ST-GCN，24/48 | 相同 |
| 节点读出 | 各分支有效节点平均 | 按节点顺序展开，Linear + ReLU |
| 时间读出 | 局部插值至身体长度，再平均时间 | 各分支保留自身时间长度，按帧顺序展开，Linear + ReLU |
| 左右手与面部融合 | 左右手平均，再与面部平均 | 左手、右手、面部按顺序拼接，学习投影 |
| 人物读出 | 人物平均 | 人物槽位按顺序拼接，学习投影 |
| 输出 | 身体 192 维 + 局部 96 维，线性分类 | 相同输出维度及分类器结构 |

本次按约定只去掉分类前的池化。面部 68 点到 6 个区域 token 的有效点均值、躯干坐标计算、CTR-GCN 内部关系统计、骨干时间下采样和 mask 传播沿用原最佳模型，因此不能称作全网络不存在任何平均或池化操作。

## 读出结构

默认输入为 `B × 3 × 64 × 133 × 2`；也接受原最佳的八通道缓存。身体时间下采样后为 16 帧，手部与面部保持 64 帧。

```text
身体：每帧 [22 × 192] 展开 → Linear(4224,192) → ReLU
      每人 [16 × 192] 展开 → Linear(3072,192) → ReLU
      [2 人 × 192] 展开 → Linear(384,192) → ReLU → 原身体投影

每只手：保留原 96→48 通道映射
        每帧 [21 × 48] 展开 → 共享 Linear(1008,48) → ReLU
        每人 [64 × 48] 展开 → 共享 Linear(3072,48) → ReLU

面部：每帧 [6 × 48] 展开 → Linear(288,48) → ReLU
      每人 [64 × 48] 展开 → Linear(3072,48) → ReLU

局部：[2 人 × (左手48 + 右手48 + 面部48)] 展开
      → Linear(288,48) → ReLU → 原局部投影

身体192 + 局部96 → 原线性分类器 → 60 类
```

新增读出投影不使用 bias；无效节点和无效帧在投影前清零，整个缺失分支的读出为零。输入少于配置的人数时，末尾人物槽位补零。原身体/局部投影及分类器仍包含各自的原有 bias。

节点、帧、左右手、人物槽位对应不同权重位置，最终表示可区分这些顺序；每一步投影仍然压缩信息，并不保证信息无损。固定顺序也意味着模型不再对人物槽位交换保持不变，训练和评估必须使用相同的人物跟踪、排序规则。

默认 60 类、64 帧、2 人时，参数量从 **1,115,716** 增至 **2,926,228**，约为原来的 2.62 倍。新模型移除了继承结构中未参与原最佳前向的历史 `local_blocks`，实际身体、手部和面部编码器均保留。学习投影增加了容量，若识别率变化，需要同时考虑读出方式与参数量增加的影响。

## 训练与评估

独立入口复用原有 NTU60 XSub 训练设定：65 轮、batch 32、SGD、学习率 0.1、5 轮预热、衰减节点 35/55、seed 1。所有参数随机初始化并联合训练，强制启用项目现有 torch.compile 配置，专用入口和通用入口均拒绝 `--no-compile`。编译失败报错，不回退整模型 eager。编译范围为身体/手部编码器和新增节点、时间、人物读出；原最佳面部 `LocalSTBlock` 已有的 eager 边界保留。

```bash
python train_best_no_pool.py --device cuda
```

Windows 本机可使用 `py -3.10` 替代 `python`。数据默认读取 `data/ntu60_skeletons_rtmw.zip`，可通过 `--archive` 指定路径。输出保存至 `outputs/best_no_pool_ntu60_xsub/<run-id>/`，含配置、源代码快照、日志、`best.pt` 和 `last.pt`。

仅检查默认形状的前向（`--dry-run` 不执行编译训练）：

```bash
python train_best_no_pool.py --dry-run --device cuda --num-workers 0
```

评估沿用原验证协议：

```bash
python tools/evaluate_best.py --checkpoint outputs/best_no_pool_ntu60_xsub/<run-id>/best.pt --device cuda
```

`window_size`、`max_persons` 和读出顺序记录在 checkpoint 的 `model_config.readout_config` 中；评估自动按该配置重建模型。改变窗口长度或最大人数会改变读出参数形状，需要重新训练。训练入口不接受 `--init-checkpoint`、`--gcn-config` 或交互配置。

## 已验证与待验证

已验证原始与八通道缓存输入的骨干输入/输出与原最佳逐项一致；读出能区分节点和时间顺序；缺失分支、空人物槽位、非有限输入及帧 mask 可正常处理；所有实际编码器和新读出均获得梯度；小型 ZIP 完成训练、保存、重新加载及统一评估。默认 64 帧、2 人、60 类的 CPU 前向通过。

本地 Windows/PyTorch 2.1 不支持编译的测试会显式跳过；在支持编译的环境执行小型 ZIP 编译训练、保存、重载和评估。可设置 `ISAA_TEST_DEVICE=cuda` 执行 CUDA 编译测试。尚未开始全量训练；不能把原最佳的 93.0946% 作为新变体的准确率。

```bash
python -m unittest tests.test_best_no_pool tests.test_wide_relative_routed_model tests.test_train_ablation_stages -v
```
