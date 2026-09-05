@echo off
setlocal
set "PYTHONUTF8=1"
set "BOT_ROOT=%~dp0"
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%BOT_PY%" set "BOT_PY=python"
cd /d "%BOT_ROOT%"
"%BOT_PY%" -m crbot --config "%BOT_ROOT%config.json" calibrate
if errorlevel 1 pause
