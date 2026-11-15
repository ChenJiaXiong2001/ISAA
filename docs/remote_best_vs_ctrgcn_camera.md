# 摄像头远程对比：最新最佳模型与原论文 CTR-GCN

入口是项目根目录的 `run_remote_full_action.cmd`。Windows 只负责打开摄像头、JPEG 编码、发送帧和显示返回结果；服务器负责两条完整推理链路。

两条链路从同一张原始摄像头图像开始，但人物框和节点提取完全独立：

- 最新最佳模型：自己的 YOLO 人物框 → RTMW-133 全身节点 → 100 个真实骨架帧 → `best-face-pair` epoch-83 checkpoint。默认输出使用最新研究冻结的 50% 算术概率池 + 50% 加权几何池，验证集 Top-1 为 95.6335%；原始算术池保留为诊断结果。
- 原论文 CTR-GCN：自己的 YOLO pose 人物框 → 独立 COCO-17 姿态 → 映射为 NTU-25 → 原始 CTR-GCN Joint checkpoint。它不会读取 RTMW-133 的任何节点或框。

服务器默认路径已经对应当前部署：

```text
face checkpoint:
/home/xiong/Desktop/ISAA/outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490/best.pt

face validation summary:
/home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/verification/best_epoch83_accuracy_weighted/summary.json

CTR-GCN raw checkpoint:
/home/xiong/Desktop/ISAA/runtime_ctrgcn25_raw.pt
```

直接启动：

```powershell
.\run_remote_full_action.cmd
```

也可以显式覆盖服务器地址、权重或姿态模型：

```powershell
py -3.10 tools\remote_full_action_camera.py `
  --ssh-host 100.123.254.107 `
  --remote-face-checkpoint /home/xiong/Desktop/ISAA/outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490/best.pt `
  --remote-ctrgcn-checkpoint /home/xiong/Desktop/ISAA/runtime_ctrgcn25_raw.pt
```

当前 RTMW 分支默认使用 `YOLO conf=0.35`、人物框阈值 `0.35`、身体关键点阈值 `0.25`，适合普通摄像头室内画面；CTR-GCN 分支仍使用自己的 YOLO pose 检测参数。现场光照变化时可单独调整 RTMW 分支：

```powershell
py -3.10 tools\remote_full_action_camera.py `
  --yolo-conf 0.35 `
  --pool-person-threshold 0.35 `
  --pool-body-threshold 0.25
```

窗口中的绿色框属于最新最佳模型，橙色框属于 CTR-GCN；两套框可能不同，这是独立人物检测和节点提取的预期结果。界面左上角的固定面板显示两者 Top-1、各自收集帧数、已提交/已返回/排队/丢帧数量和当前结果延迟，文字不再贴在人物框旁边而互相覆盖。

客户端会连续捕获并把 JPEG 放入有界队列；默认每次把最多 32 帧写入同一个 SSH 批次，最多缓存 600 帧。`batch-size` 只减少网络写入次数，服务器上的两个姿态分支仍会逐帧推理，因此它不能把单帧 GPU 推理自动变成真正的 32 帧并行推理。若希望一次传几十帧，可显式调整：

```powershell
.\run_remote_full_action.cmd --batch-size 32 --max-pending-frames 600
```

默认摄像头采样为 12 FPS，接近当前远端双分支的实测处理能力，用于避免摄像头 30 FPS 持续压满队列。可调整采样率；设为 `0` 才是不限速采集：

```powershell
.\run_remote_full_action.cmd --capture-fps 12
```

当摄像头输入速度长期高于服务器处理速度时，队列最终仍会满；面板中的“丢帧”和“落后”会直接显示实际情况。按 `q` 或 `Esc` 退出。

这个现场结果只能作为端到端工程对比。摄像头 RGB 经过 COCO-17 到 NTU-25 的二维代理映射，不等同于原论文使用的 Kinect 深度 25 点，因此 CTR-GCN 的论文 NTU 测试准确率不能直接当作现场准确率。
