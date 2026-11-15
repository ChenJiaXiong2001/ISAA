# 已完成正式训练资料

已按完整逐轮记录核验并去重，收录 **40 组**正式训练：24 轮 2 组、65 轮 22 组、100 轮 15 组、150 轮 1 组。

只收录计划至少 24 轮、未限制训练样本数、完成全部计划轮数的正式训练。
24 轮的两组 SkeleT 实验每轮重复数据 5 次。短程筛选、smoke/dry-run、失败、中断、仍在运行及完成证据不足的记录不在本目录。
原目录和全部实验均保留；本目录归集完成训练的记录，服务器通过原始目录链接访问完整权重和大日志。

- [训练总表 CSV](runs.csv) / [结构化目录 JSON](runs.json)
- [全部原始文件位置及大小](artifacts.csv)
- [训练权重的位置、大小与 SHA256](checkpoints.csv)
- [已归集的数据与日志](records/)

每组运行的详情页同时关联已有研究说明。主表按结束时间排列，保留不同 seed 和历史模型各自的记录。

表中指标来自训练 `epochs.csv`，用于定位历史记录，不能跨协议直接排名。
canonical pooled baseline 的统一评估为 **94.6400%**，其训练日志值为 **94.6461%**；依据见 [canonical 配置](../../configs/canonical_baseline.json)。
SkeleT-CTRGCN 两组记录来自 SkeleT-GCN 代码适配，不能标作原始 CTR-GCN。

| 实验 / 模型 | 轮数 | seed | 日志最佳 Top-1 | 数据与日志 |
|---|---:|---:|---:|---|
| body-local | 65/65 | 1 | 89.5989% | [查看](runs/20260929_body-local_09106999/README.md) |
| torso-cross-attn | 65/65 | 1 | 89.9240% | [查看](runs/20260930_torso-cross-attn_4f2950b2/README.md) |
| body-local-time-aug | 65/65 | 1 | 88.7710% | [查看](runs/20261002_body-local-time-aug_2aef0fed/README.md) |
| body-local-time-aug | 65/65 | 1 | 88.5502% | [查看](runs/20261002_body-local-time-aug_fb3e6c3a/README.md) |
| body-local-time-aug | 65/65 | 1 | 87.7468% | [查看](runs/20261003_body-local-time-aug_d32c3def/README.md) |
| body-local | 65/65 | 1 | 87.8204% | [查看](runs/20261003_body-local_0cfd81f5/README.md) |
| body-local-coord-aug | 65/65 | 1 | 88.4214% | [查看](runs/20261004_body-local-coord-aug_a4b17d42/README.md) |
| body-local | 65/65 | 1 | 89.3352% | [查看](runs/20261004_body-local_ecd87a0f/README.md) |
| body-local-relative | 65/65 | 1 | 89.9546% | [查看](runs/20261005_body-local-relative_54e441c0/README.md) |
| body-local-relative-split | 65/65 | 1 | 89.6357% | [查看](runs/20261005_body-local-relative-split_6dfcda15/README.md) |
| body-local-hand-ctr-relative | 65/65 | 1 | 92.0949% | [查看](runs/20261005_body-local-hand-ctr-relative_ed9879f0/README.md) |
| body-local-hand-ctr-wide-relative | 65/65 | 1 | 92.4936% | [查看](runs/20261007_body-local-hand-ctr-wide-relative_7abb08ce/README.md) |
| body-local-hand-ctr-wide-relative | 65/65 | 1 | 92.6837% | [查看](runs/20261006_body-local-hand-ctr-wide-relative_b069db37/README.md) |
| body-local-hand-ctr-wide-relative | 65/65 | 1 | 92.3709% | [查看](runs/20261008_body-local-hand-ctr-wide-relative_fa41a7d2/README.md) |
| body-local-hand-ctr-wide-relative | 65/65 | 1 | 93.0823% | [查看](runs/20261008_body-local-hand-ctr-wide-relative_15427b44/README.md) |
| body-local-hand-ctr-wide-relative-routed | 65/65 | 1 | 92.2912% | [查看](runs/20261010_body-local-hand-ctr-wide-relative-routed_1a0dd72d/README.md) |
| body-local-hand-ctr-wide-relative-full | 65/65 | 1 | 92.4016% | [查看](runs/20261010_body-local-hand-ctr-wide-relative-full_d7027e01/README.md) |
| best-hand-body-proximity | 65/65 | 1 | 93.0455% | [查看](runs/20261011_best-hand-body-proximity_7a566e43/README.md) |
| best-action-hand-body | 65/65 | 1 | 93.1436% | [查看](runs/20261011_best-action-hand-body_a3e59440/README.md) |
| body-local-hand-ctr-wide-relative-no-pool | 65/65 | 1 | 92.1072% | [查看](runs/20261012_body-local-hand-ctr-wide-relative-no-pool_a20847d6/README.md) |
| body-local-hand-ctr-wide-relative-no-pool | 65/65 | 1 | 89.1451% | [查看](runs/20261012_body-local-hand-ctr-wide-relative-no-pool_3cbf60d7/README.md) |
| body-local-hand-ctr-wide-relative | 65/65 | 1 | 93.6404% | [查看](runs/20261012_body-local-hand-ctr-wide-relative_0c6b227d/README.md) |
| SkeleT-CTRGCN/ctrgcn | 24/24 | 1 | 95.4802% | [查看](runs/20261012_SkeleT-CTRGCN_ctrgcn_190a51ad/README.md) |
| SkeleT-CTRGCN/ctrgcn-official-config | 24/24 | 1 | 95.2533% | [查看](runs/20261013_SkeleT-CTRGCN_ctrgcn-official-config_0d3fa06b/README.md) |
| body-local-hand-ctr-wide-relative | 100/100 | 1 | 94.6461% | [查看](runs/20261014_body-local-hand-ctr-wide-relative_1f8e66b0/README.md) |
| body-hand-weighted-pool | 100/100 | 1 | 94.4744% | [查看](runs/20261015_body-hand-weighted-pool_9017904e/README.md) |
| best-hand-relation-fusion | 100/100 | 1 | 94.6339% | [查看](runs/20261016_best-hand-relation-fusion_6a3db682/README.md) |
| body-local-hand-ctr-wide-relative | 100/100 | 2 | 93.7201% | [查看](runs/20261016_body-local-hand-ctr-wide-relative_1bf4f3a7/README.md) |
| best-hand-relation-fusion | 100/100 | 2 | 94.3640% | [查看](runs/20261016_best-hand-relation-fusion_952a1987/README.md) |
| body-local-shared-global-graph-100 | 100/100 | 1 | 94.1249% | [查看](runs/20261017_body-local-shared-global-graph-100_67d33c42/README.md) |
| body-local-hand-ctr-wide-relative-same-channel-late-aug | 100/100 | 1 | 94.5725% | [查看](runs/20261018_body-local-hand-ctr-wide-relative-same-channel-late-aug_dfa00067/README.md) |
| best-input-joint-selection | 100/100 | 1 | 94.7381% | [查看](runs/20261020_best-input-joint-selection_c1592145/README.md) |
| best-input-joint-selection-v5-dynamic | 100/100 | 1 | 92.6898% | [查看](runs/20261021_best-input-joint-selection-v5-dynamic_f305e3f9/README.md) |
| body-local-hand-ctr-wide-relative | 100/100 | 1 | 94.0942% | [查看](runs/20261021_body-local-hand-ctr-wide-relative_3f2968ae/README.md) |
| body-local-hand-ctr-wide-relative | 150/150 | 1 | 93.5668% | [查看](runs/20261022_body-local-hand-ctr-wide-relative_5c85f3bf/README.md) |
| best-face-pair | 100/100 | 1 | 94.4192% | [查看](runs/20261103_best-face-pair_81808aac/README.md) |
| best-face-pair-raw-xy | 100/100 | 1 | 94.2291% | [查看](runs/20261104_best-face-pair-raw-xy_3bdb0556/README.md) |
| best-face-pair | 100/100 | 2 | 94.9221% | [查看](runs/20261105_best-face-pair_11fb95c0/README.md) |
| best-face-pair-hand-face-geometry | 100/100 | 17 | 94.5848% | [查看](runs/20261105_best-face-pair-hand-face-geometry_548d4bb2/README.md) |
| best-face-pair | 100/100 | 29 | 93.8857% | [查看](runs/20261107_best-face-pair_6e6c27ff/README.md) |
