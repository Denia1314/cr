@echo off
setlocal
call "%~dp0tools\python_env.bat" console deps
if errorlevel 1 exit /b 1
"%BOT_PY%" -c "from ultralytics import YOLO" >nul 2>nul
if not errorlevel 1 goto ready
"%BOT_PY%" -m pip install -r requirements-training.txt
if errorlevel 1 exit /b 1
:ready
if exist "training\exports\katacr-real-units-v3\manifest.json" goto train
"%BOT_PY%" -m crbot.slice_dataset export-real --output training/exports/katacr-real-units-v3 --per-battle 32 --units-only --focus-units cannon-cart giant
if errorlevel 1 exit /b 1
:train
rem Automatically resume saved progress, or fine-tune the latest completed model.
set "BOT_BASE_MODEL=yolo11n.pt"
if exist "models\battlefield\katacr-real-units-v2\candidate\weights\best.pt" set "BOT_BASE_MODEL=models\battlefield\katacr-real-units-v2\candidate\weights\best.pt"
"%BOT_PY%" -m crbot.slice_dataset train --source training/exports/katacr-real-units-v3 --output models/battlefield/katacr-real-units-v3 --base-model "%BOT_BASE_MODEL%" --epochs 100 --batch 16 --device auto
pause
