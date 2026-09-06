@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%BOT_PY%" goto dependencies
where python >nul 2>nul
if errorlevel 1 goto python_missing
set "BOT_PY=python"
:dependencies
"%BOT_PY%" -c "import cv2, numpy, PIL" >nul 2>nul
if not errorlevel 1 goto github_cli
echo Required Python packages are missing. Installing them now...
"%BOT_PY%" -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
if errorlevel 1 goto dependency_failed
"%BOT_PY%" -c "import cv2, numpy, PIL" >nul 2>nul
if errorlevel 1 goto dependency_failed
:github_cli
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
:python_missing
echo Python 3.10 or newer was not found. Install Python and enable Add Python to PATH.
pause
exit /b 1
:dependency_failed
echo Python dependency installation failed. This is not a GitHub login error.
echo Check the network connection, then run setup_sync.bat again.
pause
exit /b 1
