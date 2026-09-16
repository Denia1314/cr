@echo off
setlocal
call "%~dp0tools\python_env.bat"
if errorlevel 1 exit /b 1
"%BOT_PY%" -m pip install -r "%BOT_ROOT%requirements-build.txt"
if errorlevel 1 exit /b 1
"%BOT_PY%" -m pip install -r "%BOT_ROOT%requirements-training.txt"
if errorlevel 1 exit /b 1
"%BOT_PY%" -m pip install onnxruntime-gpu==1.23.2
if errorlevel 1 exit /b 1
"%BOT_PY%" "%BOT_ROOT%tools\build_standalone.py"
if errorlevel 1 exit /b 1
echo Built: %BOT_ROOT%RoyalLab.exe
