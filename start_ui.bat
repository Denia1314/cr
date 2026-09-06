@echo off
setlocal
set "PYTHONUTF8=1"
set "BOT_ROOT=%~dp0"
set "BOT_PY=%BOT_ROOT%.venv\Scripts\python.exe"
set "BOT_PYW=%BOT_ROOT%.venv\Scripts\pythonw.exe"
if exist "%BOT_PY%" if exist "%BOT_PYW%" goto run
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "BOT_PYW=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"
if exist "%BOT_PY%" if exist "%BOT_PYW%" goto run
where python >nul 2>nul && where pythonw >nul 2>nul && set "BOT_PY=python" && set "BOT_PYW=pythonw" && goto run
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
start "" "%BOT_PYW%" "%BOT_ROOT%launch_ui.pyw"
exit /b 0
:dependency_error
echo Failed to install the required Python packages.
echo Check the network connection, then run start_ui.bat again.
pause
exit /b 1
