# AutoDL 训练编排

编排入口为 `tools/autodl_train_orchestrator.py`。它从 `%USERPROFILE%/.autodl/config.json` 读取 API 配置，在北京 B 区申请单卡 RTX 5090、系统盘扩容为 0 的实例；通过独立远端 runner 启动训练，5 分钟一次读取少量状态，异常时保存尾日志、停止训练、校验完整日志归档，再调用 API 关机并轮询到 `shutdown`。

本地后台启动命令：

```powershell
$py = "C:\Users\15637\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
Start-Process -FilePath $py -ArgumentList "-u","tools/autodl_train_orchestrator.py","--check-interval","300","--batch-size","32" -WorkingDirectory "C:\Users\15637\PycharmProjects\ISAA" -RedirectStandardOutput ".autodl_run\launcher_stdout.log" -RedirectStandardError ".autodl_run\launcher_stderr.log" -WindowStyle Hidden
```

脚本运行日志、每次运行的状态和实例 UUID 位于 `.autodl_run/<run-id>/`，最新状态副本为 `.autodl_run/instance.json`。下载后的归档位于 `autodl_training_logs/<instance-uuid>/logs.tar.gz`，其中包含 `training_launcher.log`、runner 状态、完整训练输出和每个文本日志的 SHA256 清单。

实时查看当前运行日志：

```powershell
& "C:\Users\15637\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -u tools/autodl_train_orchestrator.py --tail
```

当实例已关机时，用本地归档查看：

```powershell
Get-Content (Get-ChildItem autodl_training_logs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName\training_launcher.log -Wait
```

训练入口保持项目强制 `torch.compile` 约束；编排脚本不注入 `--no-compile`。训练代码、runner 和日志部署使用 Git bundle，远端每次运行使用独立目录，不重置项目已有目录。
