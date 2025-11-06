@echo off
echo ========================================
echo   Emotion Detection System - Installer
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python not found!
    echo Please install Python 3.10 or newer from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo WARNING: Some packages may have failed to install
    echo You can continue, but some features might not work
    echo.
)

echo.
echo [3/3] Verifying installation...
python setup.py

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo To run the application:
echo   - Double-click run.bat
echo   - Or run: cd src ^&^& python main.py
echo.
echo Don't forget to configure your Gemini API key in config.json!
echo.
pause
