@echo off
cd /d "%~dp0"
py -3.10 tools\realtime_pool_camera.py --device cpu --pose-device cpu --classify-after 5 %*
if errorlevel 1 pause
