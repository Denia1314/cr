@echo off
setlocal
call "%~dp0tools\python_env.bat" console
if errorlevel 1 goto failed
"%BOT_PY%" "%BOT_ROOT%tools\ensure_gpu.py" --force
if errorlevel 1 goto failed
"%BOT_ROOT%.venv\Scripts\python.exe" -m crbot.gpu
if errorlevel 1 goto failed
"%BOT_ROOT%.venv\Scripts\python.exe" -m unittest discover -s tests -p test_gpu.py -v
if errorlevel 1 goto failed
echo GPU setup verified. Restart Royal Lab to use this environment.
pause
exit /b 0
:failed
echo GPU setup failed. See the error above.
pause
exit /b 1
