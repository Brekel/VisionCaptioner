@echo off
setlocal

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

title Updating VisionCaptioner...

echo.
echo ------------------------------------------
echo       UPDATING VISION CAPTIONER
echo ------------------------------------------
echo.

echo [1/3] Pulling latest changes from git...
git pull
if errorlevel 1 (
    echo.
    echo WARNING: git pull failed. Continuing anyway...
    echo.
)

echo.
echo [2/3] Activating environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found at %SCRIPT_DIR%venv
    echo Please create the virtual environment first.
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"

echo.
echo [3/3] Upgrading pip packages from requirements.txt...
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed. See messages above.
    pause
    exit /b 1
)

echo.
echo ------------------------------------------
echo       UPDATE COMPLETE
echo ------------------------------------------
echo.
pause
endlocal
