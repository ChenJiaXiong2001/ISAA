# 本地摄像头体验

这个体验入口复用服务器核验过的 canonical baseline（原池化 100 帧/100 轮模型）：本地摄像头画面先经过 RTMDet/YOLO 人物框和 RTMW-L 全身 133 点提取，再将最多两个人的骨架组成滚动 100 帧窗口，最后调用原池化 checkpoint 做 60 类动作识别。模型输入仍是 `x/y/score`，分类网络仍负责双手距离/方向和局部融合；体验入口不会重新训练或加载其他模型。

## 1. 本机参数已经就绪

已从 `xiong@100.123.254.107` 下载到 `runtime/pool_100/`：

- `best.pt`：原池化模型，第98轮最佳权重，训练预算100轮，输入100帧；
- `config.json`：该次训练保存的配置；
- `evaluation_summary.json`：服务器独立评估记录，Top-1 94.6400%。

权重 SHA256 为 `674643b36e5fe77888f315c5b735259e5ba0f4109321c06d0c343d4b0748113f`，与服务器一致。人物检测/姿态权重沿用相邻 `nturgbd_skeleton_extraction/models/`，RTMW-L、RTMDet-tiny、YOLO26-X 三份文件已核对与服务器完全一致。

摄像头入口当前默认使用本机 `nturgbd_skeleton_extraction/models/yolov8n.pt` 做人物检测，以降低CPU开销；骨架模型为RTMW-L。直接启动 `run_pool_camera.cmd` 即使用YOLOv8n。可通过 `--yolo-model <本地权重路径>` 显式指定其他检测器。

本机使用 `py -3.10`，已有 PyTorch 2.1.0 CPU、OpenCV、MMPose、MMEngine、MMDetection 和 Ultralytics。直接双击项目根目录的 `run_pool_camera.cmd`，或运行：

```powershell
py -3.10 tools/realtime_pool_camera.py --device cpu --pose-device cpu
```

已验证摄像头0可以打开和读取画面、真实视频帧可以提取人物及133点骨架、下载的动作模型能识别三个离线样例。还用3个新提取的摄像头姿态帧（补齐到100帧并附带有效帧mask）跑通了实际摄像头到动作模型的链路，完成后关闭摄像头。这是运行检查，不是摄像头识别准确率测试。YOLOv8n搭配RTMW-L的CPU单帧提取实测约0.32～0.34秒，具体取决于画面、人数和设备负载。

## 2. 重新下载服务器参数

先在 `nturgbd_skeleton_extraction` 项目按其 README 安装 Python 3.10、PyTorch、MMPose、MMDetection 和 Ultralytics。然后在本项目运行：

```powershell
py -3.10 tools/download_pool_runtime.py `
  --ssh-host 100.123.254.107 `
  --ssh-user xiong `
  --ssh-key "$env:USERPROFILE\.ssh\isaa_codex" `
  --output-dir runtime/pool_100
```

默认下载服务器上的原池化 100 帧/100 轮 `best.pt` 和 `config.json`。大体积数据 ZIP 不是摄像头体验必需项；需要时再显式追加 `--remote-archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip`。

## 3. 有CUDA环境时启动

```powershell
py -3.10 tools/realtime_pool_camera.py `
  --checkpoint runtime/pool_100/best.pt `
  --pose-project ..\nturgbd_skeleton_extraction `
  --pose-device cuda:0 `
  --device cuda
```

窗口会显示人物框、RTMW-133 骨架、当前100帧窗口长度，以及按分数从高到低排列的Top 5动作编号、中文名称和置信度，每个候选单独一行。按 `q` 或 `Esc` 退出。没有CUDA时可把两个device都改成 `cpu`，但RTMW-L和模型会明显变慢。

动作分类不再设置信度显示门槛：确认有人且有效骨架帧数满足要求后，每次识别都显示Top 5，即使分数低于或等于50%也显示。分数是原60类softmax概率，不在Top 5内重新归一化。无人、推理出错或超时时清空候选列表。

无人时显示“未检测到有效人物，等待入镜”，不调用动作分类；检测到无人后清空上一段动作和窗口。默认人物框置信度至少0.6，身体至少6个关键点达到0.35且包含至少2个躯干点，连续3次有效骨架结果后才确认有人。未通过检查的人物框和骨架不显示、不进入动作模型。确认次数按骨架推理结果计算，CPU上可能等待数秒；离场清除也要等下一次骨架推理结果。若背景仍被检测为人，可提高 `--person-threshold 0.75 --yolo-conf 0.75`。这属于摄像头入口的过滤，不改训练权重或60类模型结构。

## 4. 漏帧空位补齐和记录

每次成功读取摄像头画面都会分配连续帧号，骨架推理结果携带实际处理画面的帧号。确认人物后，例如第10帧和第14帧有骨架，中间第11～13帧没有处理结果，只记录3帧漏帧统计，不把空骨架塞入动作窗口。动作窗口只保存真实完成的骨架结果；只有累计满100个真实骨架帧时才调用动作模型。检测到无人、推理出错或超时仍清空当前动作和窗口，重新入镜不跨段补齐。

窗口下方显示 `Read`（读取FPS）、`Pose`（骨架结果FPS）、`Skipped`（累计未处理比例）、`Gap`（相邻骨架源画面的时间间隔）。每次启动自动生成 `outputs/camera_logs/<时间戳>.csv`，每秒记录一次，退出时再记录最终计数；可用 `--log-dir <目录>` 改位置。CSV包含读取/提交/结果帧数、读取/骨架FPS、漏帧数及平均每秒漏帧、漏帧比例、尚未得到结果的帧数、人物段内补位数和时间间隔。

漏帧只按OpenCV成功读取的画面计算：截至最新骨架结果的源帧号，没有结果的位置才算漏帧；更晚的画面暂记为待结果。摄像头硬件或驱动内部未交付的帧无法通过此计数获知。空位能保留缺失的时间位置，不能恢复丢失的动作细节。

## 5. 运行边界

摄像头是滚动窗口体验：窗口固定保存最近100个真实骨架结果。未满100个真实骨架帧、人物离场或推理中断时都不会输出动作；界面会持续显示收集状态。窗口满100帧后默认每1秒取最近100帧分析一次（可用 `--classify-interval` 调整），不会每个骨架结果都调用分类器。`--classify-after` 固定为100，传入其他值会直接报错，避免误用短窗口或零填充输入。

## 6. 使用SSH服务器算力

本地摄像头仍由Windows打开，画面、漏帧补位和Top 5显示仍在本机完成；YOLOv8n和RTMW-L骨架提取通过一条持续SSH通道在服务器执行。直接运行：

```powershell
.\run_pool_camera_remote.cmd
```

当前 `run_pool_camera.cmd` 也已改为远程默认；若临时不使用服务器，可运行 `run_pool_camera_local.cmd`。

也可以手动指定服务器参数：

```powershell
py -3.10 tools/realtime_pool_camera.py --device cpu --classify-after 100 `
  --ssh-host 100.123.254.107 --ssh-user xiong `
  --ssh-key "$env:USERPROFILE\.ssh\isaa_codex" --remote-python /home/xiong/venvs/isaa-camera/bin/python --remote-device cuda:0
```

远端需要存在 `/home/xiong/Desktop/nturgbd_skeleton_extraction`、YOLOv8n/RTMW权重。当前远端使用 `/home/xiong/venvs/isaa-camera/bin/python`，该隔离环境复用 CUDA/PyTorch 并使用 `mmcv-lite`，避免系统 `mmcv` 扩展与 PyTorch ABI 不兼容。SSH断开、远端依赖缺失或推理超时会在窗口中显示错误，不会静默退回本机。动作分类模型仍使用本机 `runtime/pool_100/best.pt`。

体验入口使用eager推理，训练入口的强制 `torch.compile` 约束不被修改；训练/评估仍以checkpoint中保存的100帧、100轮配置为准。动作标签是完整100帧双人窗口的类别；交互动作可能需要两个人同时入镜。摄像头环境的光照、视角和动作截取与NTU验证集不同，画面置信度是模型分数，不能等同于现场准确率。
