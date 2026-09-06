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
"%BOT_PY%" -c "import cv2, numpy, PIL" >nul 2>nul
if not errorlevel 1 goto launch
echo Required Python packages are missing. Installing them now...
"%BOT_PY%" -m pip install --disable-pip-version-check -r "%BOT_ROOT%requirements.txt"
if errorlevel 1 goto dependency_error
"%BOT_PY%" -c "import cv2, numpy, PIL" >nul 2>nul
if errorlevel 1 goto dependency_error
:launch
"%BOT_PY%" -m crbot --config "%BOT_ROOT%config.json" run %*
if errorlevel 1 pause
exit /b %errorlevel%
:dependency_error
echo Failed to install the required Python packages.
echo Check the network connection, then run start_bot.bat again.
pause
exit /b 1
