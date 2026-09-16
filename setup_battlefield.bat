@echo off
setlocal
call "%~dp0tools\python_env.bat" cli deps
if errorlevel 1 goto failed
"%BOT_PY%" "%~dp0tools\ensure_battlefield.py" --verify
if errorlevel 1 goto failed
echo Battlefield setup and read-only screenshot verification completed.
pause
exit /b 0
:failed
echo Battlefield setup failed. Review the error above.
pause
exit /b 1
