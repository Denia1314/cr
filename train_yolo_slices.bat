@echo off
setlocal
call "%~dp0tools\python_env.bat" console deps
if errorlevel 1 exit /b 1
"%BOT_PY%" -c "from ultralytics import YOLO" >nul 2>nul
if not errorlevel 1 goto ready
"%BOT_PY%" -m pip install -r requirements-training.txt
if errorlevel 1 exit /b 1
:ready
if not exist "training\external\katacr-slices\manifest.json" goto missing
if exist "training\exports\katacr-bootstrap\manifest.json" goto train
"%BOT_PY%" -m crbot.slice_dataset export --source training/external/katacr-slices --output training/exports/katacr-bootstrap
if errorlevel 1 exit /b 1
:train
rem Automatically resume saved progress, or fine-tune the latest completed model.
"%BOT_PY%" -m crbot.slice_dataset train --source training/exports/katacr-bootstrap --output models/battlefield/katacr --epochs 50 --device auto
pause
exit /b
:missing
echo Run import_yolo_slices.bat first.
pause
exit /b 1
