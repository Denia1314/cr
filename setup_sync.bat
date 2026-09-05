@echo off
setlocal EnableExtensions
set "PYTHONUTF8=1"
chcp 65001 >nul
cd /d "%~dp0"
title Royal Lab - GitHub Sync Setup

rem Prefer the project's own environment. The Codex runtime path is only a fallback.
set "BOT_PY=%CD%\.venv\Scripts\python.exe"
if exist "%BOT_PY%" goto python_ready
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%BOT_PY%" goto python_ready
where python >nul 2>nul
if errorlevel 1 goto missing_python
set "BOT_PY=python"
:python_ready
"%BOT_PY%" -c "import crbot" >nul 2>nul
if errorlevel 1 goto broken_python

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
echo Checking GitHub login...
"%SYNC_GH%" auth status --hostname github.com >nul 2>nul
if not errorlevel 1 goto setup
echo.
echo GitHub login is missing or expired.
echo A browser verification page will open and the one-time code will be copied.
echo Complete that page, then return here.
echo Y|"%SYNC_GH%" auth login --hostname github.com --git-protocol https --web --clipboard --skip-ssh-key
if errorlevel 1 goto auth_failed
"%SYNC_GH%" auth status --hostname github.com >nul 2>nul
if errorlevel 1 goto auth_failed
:setup
echo.
echo Connecting the private training-data repository...
"%BOT_PY%" -m crbot sync setup %*
if errorlevel 1 goto failed
"%BOT_PY%" -m crbot sync now
if errorlevel 1 goto failed
echo Setup complete. Restart the Royal Lab console to enable background sync.
pause
exit /b 0
:missing_python
echo Python was not found. Run install.bat first, then reopen this script.
pause
exit /b 1
:broken_python
echo The project Python environment is incomplete. Run install.bat first.
pause
exit /b 1
:auth_failed
echo GitHub login was not completed. Reopen this script and try again.
pause
exit /b 1
:failed
echo.
echo Setup or sync failed.
echo Check that the signed-in GitHub account has Write access to:
echo Denia1314/cr-training-data
pause
exit /b 1
