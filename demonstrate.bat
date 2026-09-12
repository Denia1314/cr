@echo off
setlocal
call "%~dp0tools\python_env.bat" console
if errorlevel 1 (
  pause
  exit /b 1
)
"%BOT_PY%" -m crbot --config "%BOT_ROOT%config.json" demonstrate
set "BOT_EXIT=%errorlevel%"
if not "%BOT_EXIT%"=="0" pause
exit /b %BOT_EXIT%
