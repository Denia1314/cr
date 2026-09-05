@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%BOT_PY%" set "BOT_PY=python"
set "SYNC_GH=%LOCALAPPDATA%\CodexTools\github-cli\bin\gh.exe"
if exist "%SYNC_GH%" goto auth
set "SYNC_GH=%ProgramFiles%\GitHub CLI\gh.exe"
if exist "%SYNC_GH%" goto auth
set "SYNC_GH=gh"
where gh >nul 2>nul
if not errorlevel 1 goto auth
echo Install GitHub CLI first: winget install --id GitHub.cli -e
echo Then reopen this script.
pause
exit /b 1
:auth
"%SYNC_GH%" auth status >nul 2>nul
if not errorlevel 1 goto setup
"%SYNC_GH%" auth login --hostname github.com --git-protocol https --web --skip-ssh-key
if errorlevel 1 goto failed
:setup
"%BOT_PY%" -m crbot sync setup %*
if errorlevel 1 goto failed
"%BOT_PY%" -m crbot sync now
if errorlevel 1 goto failed
echo Setup complete. Restart the Royal Lab console to enable background sync.
pause
exit /b 0
:failed
echo Setup or sync failed. Check GitHub login and private data repository access.
pause
exit /b 1
