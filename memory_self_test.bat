@echo off
setlocal
call "%~dp0tools\python_env.bat" console
if errorlevel 1 exit /b 1
"%BOT_PY%" -m crbot.memory_lab self-test %*
set "BOT_EXIT=%errorlevel%"
echo.
echo Report: reports\memory-lab-self-test.json
echo This tests a private child process, not Clash Royale memory.
pause
exit /b %BOT_EXIT%
