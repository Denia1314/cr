@echo off
setlocal
cd /d "%~dp0"
call "%~dp0tools\python_env.bat" console deps
if errorlevel 1 exit /b 1
"%BOT_PY%" -c "from ultralytics import YOLO" >nul 2>nul
if not errorlevel 1 goto ready
"%BOT_PY%" -m pip install -r requirements-training.txt
if errorlevel 1 exit /b 1
:ready
if exist "training\exports\katacr-real-units-v3\manifest.json" goto prepared
"%BOT_PY%" -m crbot.slice_dataset export-real --output training/exports/katacr-real-units-v3 --per-battle 32 --units-only --focus-units cannon-cart giant
if errorlevel 1 exit /b 1
:prepared
set "BOT_BASE_MODEL=yolo11n.pt"
if exist "models\battlefield\katacr-real-units-v2\candidate\weights\best.pt" set "BOT_BASE_MODEL=models\battlefield\katacr-real-units-v2\candidate\weights\best.pt"
set "BOT_PLAN="
if /i "%~1"=="--plan" set "BOT_PLAN=--plan"
set /a BOT_ROUND=0
echo Continuous YOLO training: 20 epochs per run, batch 80, low learning rate.
echo Press Ctrl+C to stop. Saved checkpoints are preserved.
:repeat
set /a BOT_ROUND+=1
echo Starting run %BOT_ROUND%...
"%BOT_PY%" -m crbot.slice_dataset train --source training/exports/katacr-real-units-v3 --output models/battlefield/katacr-real-units-v3 --base-model "%BOT_BASE_MODEL%" --safe-finetune --epochs 20 --batch 80 --seed %RANDOM% --device auto %BOT_PLAN%
if errorlevel 1 goto failed
if defined BOT_PLAN exit /b 0
echo Run complete. Next run starts in 10 seconds. Ctrl+C stops the loop.
timeout /t 10 /nobreak >nul
if errorlevel 1 exit /b 1
goto repeat
:failed
echo Training stopped or failed. Existing models and checkpoints are preserved.
pause
exit /b 1
