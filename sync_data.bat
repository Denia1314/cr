@echo off
setlocal EnableExtensions
set "PYTHONUTF8=1"
chcp 65001 >nul
cd /d "%~dp0"
set "BOT_PY=%CD%\.venv\Scripts\python.exe"
if exist "%BOT_PY%" goto python_ready
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%BOT_PY%" goto python_ready
where python >nul 2>nul
if errorlevel 1 goto python_missing
set "BOT_PY=python"
:python_ready
"%BOT_PY%" -c "import cv2, numpy, PIL" >nul 2>nul
if not errorlevel 1 goto sync
echo Required Python packages are missing. Installing them now...
"%BOT_PY%" -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
if errorlevel 1 goto dependency_failed
:sync
"%BOT_PY%" -m crbot sync now
pause
exit /b %errorlevel%
:python_missing
echo Python 3.10 or newer was not found. Install Python and enable Add Python to PATH.
pause
exit /b 1
:dependency_failed
echo Python dependency installation failed. Check the network connection and retry.
pause
exit /b 1
