@echo off
setlocal
set "PYTHONUTF8=1"
set "BOT_ROOT=%~dp0"
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%BOT_PY%" goto run
where py >nul 2>nul && set "BOT_PY=py" && goto run
where python >nul 2>nul && set "BOT_PY=python" && goto run
echo Python 3.10 or newer was not found.
pause
exit /b 1
:run
cd /d "%BOT_ROOT%"
"%BOT_PY%" -m crbot --config "%BOT_ROOT%config.json" run %*
if errorlevel 1 pause
