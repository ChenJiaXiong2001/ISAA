@echo off
cd /d "%~dp0"
py -3.10 tools\realtime_pool_camera.py --device cpu --classify-after 100 --ssh-host 100.123.254.107 --ssh-user xiong --ssh-key "%USERPROFILE%\.ssh\isaa_codex" --remote-python /home/xiong/venvs/isaa-camera/bin/python --remote-device cuda:0 %*
if errorlevel 1 pause
