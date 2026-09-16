@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\cleanup_artifacts.ps1" %*
set "cleanup_result=%errorlevel%"
echo.
if not "%cleanup_result%"=="0" echo Cleanup incomplete. See the message and report above.
pause
exit /b %cleanup_result%
