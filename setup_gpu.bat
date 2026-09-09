@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
set "BOT_BASE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%BOT_BASE%" set "BOT_BASE=python"
"%BOT_BASE%" -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-training.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "from ultralytics import YOLO"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m crbot.gpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m unittest discover -s tests -p test_gpu.py -v
if errorlevel 1 goto failed
echo GPU setup verified. Restart Royal Lab to use this environment.
pause
exit /b 0
:failed
echo GPU setup failed. See the error above.
pause
exit /b 1
