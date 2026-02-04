@echo off
setlocal

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

title Launching VisionCaptioner...

echo.
echo ------------------------------------------
echo       STARTING VISION CAPTIONER
echo ------------------------------------------
echo.
echo Activating environment...
call "venv\Scripts\activate.bat"

echo.
echo Launching GUI...
echo (The Splash Screen should appear momentarily)

start "" pythonw.exe main.py

endlocal