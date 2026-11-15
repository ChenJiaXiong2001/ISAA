# 预压缩缺失的固定 max5 验证协议

固定候选为：100 帧压缩后，仅对肩髋锚点长度不超过 5 帧、前后有效且不跨无效帧的零分缺口做双侧线性插值。不做低置信阈值化；在训练校准长序列的已有筛查里，`0.5 + max5` 没有相对单独 max5 带来额外准确率收益。

冻结 seed2 的 `best.pt`，SHA256 `25037c807a45e8ec647f89b58557f63c9a68c89ce28a62949bc5b99ce6e47432`。HF/BF 类别权重取该 checkpoint 内训练受试者校准计数，按原准确率比例 gamma=1 混合，不使用验证标签重新估计权重。原始帧先以样本名与固定 seed 确定 5% 独立锚点缺失，然后经过现有 confidence-weighted 压缩；在同一 clean 或受损输入上分别执行无修复和 max5 冻结推理。

每条输出包含样本名、真实标签、clean 和受损条件下两种方法的预测；汇总保留 checkpoint、成员顺序、clean/受损张量、frame mask 的 SHA256，可配对统计修正、新增错误及净变化。`tools/summarize_precompression_subject_bootstrap.py` 按完整受试者重采样，报告样本加权的差值区间、逐人结果和逐人等权平均。两名训练校准受试者的数据不能支持可靠的受试者总体区间；全验证若运行，可报告各人的贡献及其重采样区间，但该区间不消除模型选择或验证集复用偏差。

此前官方 XSub 验证集已用于历史 checkpoint、模型及候选探索，因而下面的全量结果只能作为固定数据集、固定合成缺失下的补充评估，不是独立泛化确认。模拟丢失不能代表真实检测器的连续遮挡、低分噪声或其他领域；双侧插值最多需要 5 个未来压缩帧，不是实测端到端延迟。

## 运行

以下在远端 A6000 长训结束且 GPU 空闲后执行，当前工作目录为 `/home/xiong/Desktop/ISAA/experiments/face_pair_handface_geometry_20261105/code`。保持输出目录不存在，避免覆盖已有结果。以下变量和命令使用 Linux shell：

```bash
RUN=/home/xiong/Desktop/ISAA/outputs/best_face_pair_100_seed2_20261105/20261105-051202-2fcd5590
ARCHIVE=/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip
OUT=/home/xiong/Desktop/ISAA/outputs

# 可选复核已完成的 160 条训练校准长序列；与现有报告重复，不是新增独立证据。
python3 -u tools/evaluate_face_pair_precompression_repair.py \
  --checkpoint "$RUN/best.pt" --manifest "$RUN/split_manifest.json" --archive "$ARCHIVE" \
  --members-file "$OUT/precompression_calibration_long_remaining_20261106/members.csv" \
  --output-dir "$OUT/precompression_max5_remaining160_recheck_20261106" \
  --partition calibration --probability 0.05 --seed 1729 --max5-only \
  --batch-size 32 --num-workers 4 --device cuda

# 预先固定的整个 XSub 验证集。
python3 -u tools/evaluate_face_pair_precompression_repair.py \
  --checkpoint "$RUN/best.pt" --manifest "$RUN/split_manifest.json" --archive "$ARCHIVE" \
  --output-dir "$OUT/precompression_max5_validation_20261106" \
  --partition validation --max-samples 0 --probability 0.05 --seed 1729 --max5-only \
  --batch-size 32 --num-workers 4 --device cuda

python3 tools/summarize_precompression_subject_bootstrap.py \
  --evaluation-dir "$OUT/precompression_max5_validation_20261106" \
  --output "$OUT/precompression_max5_validation_20261106/subject_bootstrap.json" \
  --repeats 20000 --seed 20261105
```

此前 160 条 CPU 单线程评估包含三种方法和两种条件，耗时 242 秒。按前向次数粗略推算，全部 16,306 条采用 CPU 单线程的两方法约需 4.6 小时，实际取决于数据读取和机器负载；CUDA 实际耗时尚未测量，不能据此推算精确完成时间。推理脚本使用冻结 eager 模型，训练仍保持强制 compile 的既定约束。

`--max5-only` 已经用 2 条真实校准样本 CPU 烟测，输出仅包含无修复和 max5；受试者 bootstrap 和成员选择共 4 项测试通过。160 条历史结果经新汇总工具验证，逐人净变化和既有统计一致，但只有两个受试者时 bootstrap 仅用于检查程序。
