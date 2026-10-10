@echo off
setlocal
call "%~dp0tools\python_env.bat" console
if errorlevel 1 exit /b 1
"%BOT_PY%" -m crbot.memory_lab diagnose %*
set "BOT_EXIT=%errorlevel%"
echo.
echo Report: reports\memory-lab-diagnose.json
echo Exit 2 means access is blocked, the target is unavailable, or diagnosis failed.
pause
exit /b %BOT_EXIT%
