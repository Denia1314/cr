@echo off
setlocal
call "%~dp0tools\python_env.bat" console deps
if errorlevel 1 (
  pause
  exit /b 1
)
"%BOT_PY%" -m crbot --config "%BOT_ROOT%config.json" sync now
set "BOT_EXIT=%errorlevel%"
pause
exit /b %BOT_EXIT%
