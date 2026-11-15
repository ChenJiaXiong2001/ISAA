# 无固定数量选点 v2

新增独立变体 best-input-joint-selection-v2，保留 v1、baseline、旧权重与旧入口。训练从随机初始化开始，不加载 baseline best。此次修正门控训练，不更改面部为 5 点，也不实现单次训练中物理裁剪续训。

## 修正内容

- 不设置固定 K、保留率或节点数量预算。仍由动作分类损失和节点成本共同决定数量；允许全开或全关可选点的结果，不能保证稀疏或无损。
- 前 5 轮严格全开；门参数不更新，也不参与 Adam 步骤。第 6 轮门参数重置到 0.5，清空门优化器状态。
- 接下来的 5 轮逐渐从全开混合过渡到随机 STG；高斯扰动也从 0 逐渐增大。不是突然启用全部噪声。
- 每次门更新后投影 mu 到 [-0.5, 1.0]，并清除边界上继续向外推动的 Adam 一阶动量，避免参数漂到 2–3 后稀疏梯度失效。这是本项目的有界 STG 改编，不是对论文的完整复现。
- 识别网络使用原 SGD/余弦学习率；门使用独立 Adam，默认学习率 0.001，无权重衰减，不随分类学习率一起衰减。
- 节点成本改成未归一化的预期可选原始节点数：loss = CE + lambda * sum(cost * Phi(mu/sigma))。左右手对应关节共享门，成本为 2；可选身体点成本为 1。
- lambda 从第 6 轮开始用 20 轮升到 0.01。默认最大节点成本约 0.54。v1 的 0.1 是对除以 54 的归一化量加权，两者不能只比较系数大小；v2 默认最终每节点代价约为 v1 的 5.4 倍。此值是初始实验配置，不是已验证的最优超参数。

保留原候选集：129 原始点、34 个门控制 54 可选点，固定依赖共 75 点。面部 68 原始点仍按原方式池化成 6 token；身体、双手、双手距离/方向及融合保留。

## 可观察结果

console.log / batches.jsonl 显示当前阈值保留点数、mu 范围、预期点数、边界比例，以及分类梯度/稀疏梯度/总梯度范数。分类门梯度通过总梯度减去解析稀疏梯度获得，记录于梯度裁剪前；不进行额外整模型反向。

selection_history.jsonl 保存完整 mu、确定性门值、关节名单、名单稳定轮数。默认每 5 轮以及最后一轮增加硬屏蔽验证 val_hard，先测 0/1 门的影响；它仍使用密集图，不能当作物理精简模型的精度或速度。

- best.pt：软门验证准确率最高，便于与原实验比较。
- best_hard.pt：噪声/成本两个 ramp 结束后，硬屏蔽验证准确率最高。
- best_sparse.pt：两个 ramp 结束后，确实少于 129 点的候选中，硬屏蔽验证准确率最高；可能不存在，不能用假数据补造。
- last.pt：最后一轮，包含独立门优化器状态与诊断信息；当前入口仍不支持断点恢复。
- selection_result.json：标明最终是否减少、是否存在稀疏候选；没有物理裁剪。
- selection_health.json：两个 ramp 后连续 5 轮没有减少时写出警告；全关闭可选节点也打印警告。只预警，不擅自停训练或更改超参数。健康文件可能保留先前警告，最新状态看最后一条 history/result。

默认 0.5 为预先固定的导出阈值，不代表绝对重要性。未收敛的中间过渡期门值会受 open-mix 影响，不应拿它做最终关节名单。

## 服务器命令

完整训练（当前更新不会自动启动）：

```bash
cd /home/xiong/Desktop/ISAA
bash tools/train_input_joint_selection_v2_server.sh \
  --stage select \
  --archive data/ntu60_skeletons_rtmw.zip \
  --epochs 100 --window-size 100 \
  --batch-size 32 --test-batch-size 32 \
  --device cuda --compile
```

建议先做缩短日程的筛查（12 轮，用于确认门能正常分化，不用于与 100 轮 baseline 比准确率）：

```bash
bash tools/train_input_joint_selection_v2_server.sh \
  --stage select --archive data/ntu60_skeletons_rtmw.zip \
  --epochs 12 --warmup-epochs 2 \
  --gate-warmup-epochs 2 --gate-ramp-epochs 5 --gate-noise-ramp-epochs 3 \
  --hard-eval-interval 2 --window-size 100 \
  --batch-size 32 --test-batch-size 32 --device cuda --compile \
  --save-dir outputs/input_joint_v2_screen_12
```

确定 best_sparse.pt 存在且硬屏蔽准确率可接受后，再用新入口的 --export-selection 和 --selection-output 导出；不自动运行额外 compact 100 轮。compact 仍只读名单、从随机初始化训练；soft/硬屏蔽/物理精简三种结果需要分开报告。

## 强制编译

专用入口、通用入口、默认配置和实际调用均强制 torch.compile，所有顺序的 --no-compile 都拒绝。suppress_errors=False、fail_on_recompile_limit_hit=True；编译失败明确报错，不回退整模型 eager。编译覆盖新门控、节点成本、身体/双手编码、几何、池化融合和分类器；仅保留原面部 LocalSTBlock 的既有 eager 边界。

验证须包含新旧单元测试、真实服务器 CUDA 100 帧前向/反向、热身与门更新切换、部分 batch 的软/硬评估，以及小规模真实数据 select→导出→compact→checkpoint 评估。小规模检查只证明运行正确，不等于默认完整训练已能达到指定精度或删点率。
