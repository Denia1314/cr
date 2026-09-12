@echo off
setlocal
call "%~dp0tools\python_env.bat" ui deps
if errorlevel 1 (
  pause
  exit /b 1
)
start "" "%BOT_PYW%" "%BOT_ROOT%launch_ui.pyw"
exit /b %errorlevel%
