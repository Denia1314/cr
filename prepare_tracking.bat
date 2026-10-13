@echo off
setlocal
call "%~dp0tools\python_env.bat" train deps
if errorlevel 1 exit /b 1
set "TRACK_REVIEW_DIR=training\tracking-review-%RANDOM%-%RANDOM%"
"%BOT_PY%" -m crbot.tracking_learning prepare --output "%TRACK_REVIEW_DIR%"
if errorlevel 1 (
  pause
  exit /b 1
)
start "" "%BOT_ROOT%%TRACK_REVIEW_DIR%\review.html"
