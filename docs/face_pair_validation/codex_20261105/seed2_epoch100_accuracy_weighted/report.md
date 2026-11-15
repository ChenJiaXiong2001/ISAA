# seed2 原比例加权复评

## 运行与复现

本次运行是 `best-face-pair` 的独立随机初始化训练，seed=2，100 epochs。训练在 NVIDIA RTX A6000、PyTorch `2.14.0+cu126`、CUDA 12.6 上完成，状态为 `completed`，共 119,300 次优化器更新，耗时约 31,802 秒。

训练入口使用 `torch.compile` + Inductor，`compile=true`、`required=true`、`suppress_errors=false`、`fail_on_recompile_limit_hit=true`，没有整模型 eager 回退；新增身体/双手编码、池化、HF/BF 拼接、两个分类器和概率融合均在编译范围内，面部 `LocalSTBlock` 保留原模型已有 eager 边界。配置中的 `init_checkpoint=null`，初始化标记为 `random_scratch_no_checkpoint`。

训练配置、成员清单、逐 epoch 指标和 checkpoint 均保存在本目录。官方验证成员 SHA256 为 `82fd60410fd04e381c65d3bb9f24f18f7993124ee2581902942fbfb4a2ecddc0`，与 seed1 和 canonical baseline 相同。seed2 `best.pt` SHA256 为 `25037c807a45e8ec647f89b58557f63c9a68c89ce28a62949bc5b99ce6e47432`。

## 结果

训练日志记录的 epoch100 原融合（硬路由）为 94.9221%（15,478/16,306）。对同一 `best.pt` 使用冻结的训练校准计数，按用户指定的原准确率比例做 log-domain 概率融合，得到：

| 方法 | 正确数 | Top-1 | Top-5 | macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| HF 单头 | 14,435 | 88.5257% | 98.1970% | 88.5131% |
| BF 单头 | 15,151 | 92.9167% | 99.3131% | 92.9914% |
| γ=1 原准确率比例加权 | **15,575** | **95.5170%** | 99.5033% | **95.5385%** |

相对 canonical baseline 94.6400%（15,432/16,306），seed2 γ=1 增加 143 条正确，提升 **+0.8770 个百分点**。相对 seed1 epoch83 的 95.2778%（15,536/16,306），增加 39 条正确，提升 **+0.2392 个百分点**。这次结果支持该方向在不同随机种子下仍有提升，但还不能替代多 seed 统计结论。

按验证样本名与 canonical baseline 错误集合配对，seed2 将 baseline 的 439 条错误改正，同时把 296 条 baseline 正确变为错误；paired bootstrap（50,000 次）对 Top-1 差异的 95% 区间为 **+0.552～+1.202 个百分点**。seed2 与 seed1 的 +39 条差异区间跨过 0，说明不同随机种子之间的细小差异仍需更多重复实验确认。

同一 checkpoint 的第二次全量样本导出得到 15,574/16,306（95.5109%），与首次复评相差 1 条，符合 TF32/Inductor CUDA 非确定性；保守按重复结果与 canonical baseline 做 20,000 次 paired bootstrap，提升为 +0.8708 个百分点，95% 区间 +0.5518–+1.2020 个百分点。HF logits 最大绝对值为 218.63，但模型、优化器和融合输出均有限，概率归一化误差约 3.6e-7。

训练日志中的硬路由指标与本复评的软融合指标不是同一读出，不能直接互换。复评使用与训练相同的验证成员、batch=32、raw RTMW、confidence-weighted 100 帧、最多 2 人、无增强；官方验证数据没有参与权重估计。最大概率归一化误差为 `3.576e-7`，所有输出有限。

## 证据文件

- `config.json`：完整训练参数、编译策略、环境和数据分割摘要。
- `status.json`：100 epochs 正常完成。
- `epochs.csv`：逐 epoch 训练、校准、验证和路由指标。
- `split_manifest.json`：fit/calibration/validation 成员与 SHA256。
- `summary.json`：γ=1 评估摘要、逐方法指标和数值核验。
- `per_class_accuracy.csv`、三个 `confusion_matrix_*.csv`、`errors.csv`：逐类与逐样本证据。
- `errors_all.csv`、`summary_all.json`：第二次完整样本导出，用于配对复核并记录 1 条 CUDA 非确定性差异。
- `best.pt`、`source.zip`：可复评 checkpoint 和本次训练源码快照。
