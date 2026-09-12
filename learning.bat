@echo off
setlocal
call "%~dp0tools\python_env.bat" console
if errorlevel 1 (
  pause
  exit /b 1
)
if "%~1"=="" (
  "%BOT_PY%" -m crbot --config "%BOT_ROOT%config.json" learn audit
) else (
  "%BOT_PY%" -m crbot --config "%BOT_ROOT%config.json" learn %*
)
set "BOT_EXIT=%errorlevel%"
pause
exit /b %BOT_EXIT%
