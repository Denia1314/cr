@echo off
setlocal
call "%~dp0tools\python_env.bat" console
if errorlevel 1 (
  pause
  exit /b 1
)
"%BOT_PY%" -m unittest discover -s tests -v
set "BOT_EXIT=%errorlevel%"
pause
exit /b %BOT_EXIT%
