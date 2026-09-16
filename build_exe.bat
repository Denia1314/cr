@echo off
setlocal
call "%~dp0tools\python_env.bat"
if errorlevel 1 exit /b 1
"%BOT_PY%" -m pip install -r "%BOT_ROOT%requirements-build.txt"
if errorlevel 1 exit /b 1
"%BOT_PY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name RoyalLab --distpath "%BOT_ROOT%." --workpath "%BOT_ROOT%build\exe" --specpath "%BOT_ROOT%build" "%BOT_ROOT%tools\exe_launcher.py"
if errorlevel 1 exit /b 1
echo Built: %BOT_ROOT%RoyalLab.exe
