@echo off
echo ========================================
echo   Emotion Detection System Launcher
echo ========================================
echo.

cd /d "%~dp0"

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo.
echo Starting application...
cd src
python main.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start!
    echo Make sure all dependencies are installed:
    echo   pip install -r requirements.txt
    pause
)
