@echo off
setlocal
call "%~dp0tools\python_env.bat" train deps
if errorlevel 1 exit /b 1
if "%~1"=="" (
  echo 请把保存的 tracking-annotations.jsonl 拖到这个文件上。
  pause
  exit /b 2
)
set "TRACK_MODEL_PATH=models\tracking\candidate-%RANDOM%-%RANDOM%.json"
"%BOT_PY%" -m crbot.tracking_learning train "%~1" --output "%TRACK_MODEL_PATH%"
pause
