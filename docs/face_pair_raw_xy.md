# Face-pair raw-XY 变体

`best-face-pair-raw-xy` 在现有 `best-face-pair` 的随机初始化训练流程上新增一个坐标消融：身体分支保持原实现；手部支路保留 torso-relative `x/y`，并追加 RTMW ZIP 的原始 `x/y`；面部支路保留 torso-relative `x/y`，并追加原始 `x/y`。置信度、双手距离和方向特征也保留。原 `best-face-pair` 的 torso-relative 版本、入口和检查点保持不变。

新架构标识为 `rtmw_best_relplusrawxy_handface_bodyface_classwise_v1`，输出目录默认是 `outputs/best_face_pair_raw_xy_100_ntu60_xsub`。训练仍固定使用 100 帧、confidence-weighted 压缩、随机初始化、batch warmup cosine，并强制 `torch.compile`/Inductor；传入 `--no-compile` 会报错。

服务器启动：

```bash
bash tools/train_face_pair_server.sh \
  --model-variant best-face-pair-raw-xy \
  --epochs 100 \
  --device cuda \
  --save-dir outputs/best_face_pair_raw_xy_100_ntu60_xsub
```

编译验证：

```bash
python -s tools/check_face_pair_compile.py \
  --model-variant best-face-pair-raw-xy \
  --output docs/face_pair_validation/raw_xy_cuda_compile.json
```
