@echo off
setlocal
set "BOT_ROOT=%~dp0"
set "BOT_PYW=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"
if exist "%BOT_PYW%" goto run
where pyw >nul 2>nul && set "BOT_PYW=pyw" && goto run
where pythonw >nul 2>nul && set "BOT_PYW=pythonw" && goto run
echo Python 3.10 or newer was not found.
pause
exit /b 1
:run
cd /d "%BOT_ROOT%"
start "" "%BOT_PYW%" "%BOT_ROOT%launch_ui.pyw"
