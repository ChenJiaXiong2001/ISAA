@echo off
setlocal
cd /d "%~dp0"
py -3.10 tools\compare_pool_vs_ctrgcn25.py %*
endlocal
