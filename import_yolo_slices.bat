@echo off
setlocal
call "%~dp0tools\python_env.bat" console deps
if errorlevel 1 exit /b 1
if exist "training\external\clash-royale-detection-dataset\.git" goto import
git clone --depth 1 https://github.com/wty-yy/Clash-Royale-Detection-Dataset.git training/external/clash-royale-detection-dataset
if errorlevel 1 exit /b 1
:import
"%BOT_PY%" -m crbot.slice_dataset import
pause
