@echo off
setlocal EnableExtensions
set "PYTHONUTF8=1"
chcp 65001 >nul
cd /d "%~dp0"
set "BOT_PY=%CD%\.venv\Scripts\python.exe"
if exist "%BOT_PY%" goto python_ready
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%BOT_PY%" goto python_ready
set "BOT_PY=python"
:python_ready
"%BOT_PY%" -m crbot sync now
pause
