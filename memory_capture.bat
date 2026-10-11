@echo off
setlocal
call "%~dp0tools\python_env.bat" ui
if errorlevel 1 exit /b 1
start "" "%BOT_PYW%" -m crbot.memory_capture_ui %*
