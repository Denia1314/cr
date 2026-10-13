@echo off
setlocal
cd /d "%~dp0"
call "%~dp0tools\python_env.bat" console deps
if errorlevel 1 exit /b 1
if exist "training\exports\katacr-rare-real-v1\manifest.json" goto train
"%BOT_PY%" -m crbot.rare_unit_dataset --output training/exports/katacr-rare-real-v1
if errorlevel 1 exit /b 1
:train
"%BOT_PY%" -m crbot.slice_dataset train --source training/exports/katacr-rare-real-v1 --output models/battlefield/katacr-rare-real-v1 --base-model models/battlefield/katacr-real-units-v3/candidate-3/weights/best.pt --safe-finetune --epochs 20 --batch 16 --device auto
pause
