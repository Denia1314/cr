@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%BOT_PY%" set "BOT_PY=python"
"%BOT_PY%" -m crbot sync now
pause
