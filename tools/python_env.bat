@echo off
rem Called inside each launcher's setlocal; exports BOT_ROOT, BOT_PY and BOT_PYW.
set "PYTHONUTF8=1"
for %%I in ("%~dp0..") do set "BOT_ROOT=%%~fI\"
cd /d "%BOT_ROOT%"
set "BOT_PY=%BOT_ROOT%.venv\Scripts\python.exe"
set "BOT_PYW=%BOT_ROOT%.venv\Scripts\pythonw.exe"
if exist "%BOT_PY%" if /i not "%~1"=="ui" goto ready
if exist "%BOT_PY%" if exist "%BOT_PYW%" goto ready
set "BOT_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "BOT_PYW=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"
if exist "%BOT_PY%" if /i not "%~1"=="ui" goto ready
if exist "%BOT_PY%" if exist "%BOT_PYW%" goto ready
set "BOT_PY=python"
set "BOT_PYW=pythonw"
where python >nul 2>nul
if errorlevel 1 goto try_py
if /i not "%~1"=="ui" goto ready
where pythonw >nul 2>nul
if not errorlevel 1 goto ready
goto missing
:try_py
if /i "%~1"=="ui" goto missing
where py >nul 2>nul
if errorlevel 1 goto missing
set "BOT_PY=py"
:ready
if /i not "%~2"=="deps" exit /b 0
"%BOT_PY%" "%BOT_ROOT%tools\ensure_gpu.py"
if errorlevel 1 exit /b 1
if exist "%BOT_ROOT%.venv\Scripts\python.exe" set "BOT_PY=%BOT_ROOT%.venv\Scripts\python.exe"
if exist "%BOT_ROOT%.venv\Scripts\pythonw.exe" set "BOT_PYW=%BOT_ROOT%.venv\Scripts\pythonw.exe"
"%BOT_PY%" -c "import cv2, numpy, PIL" >nul 2>nul
if not errorlevel 1 exit /b 0
echo Required Python packages are missing. Installing them now...
"%BOT_PY%" -m pip install --disable-pip-version-check -r "%BOT_ROOT%requirements.txt"
if errorlevel 1 goto dependency_failed
"%BOT_PY%" -c "import cv2, numpy, PIL" >nul 2>nul
if errorlevel 1 goto dependency_failed
exit /b 0
:dependency_failed
echo Python dependency installation failed. Check the error above and retry.
exit /b 1
:missing
echo Python 3.10 or newer was not found. Install Python and add it to PATH.
exit /b 1
