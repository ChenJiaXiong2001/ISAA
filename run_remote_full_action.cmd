@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
py -3.10 tools\remote_full_action_camera.py %*
endlocal
