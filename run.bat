@echo off
REM Albedo to Normal Map Converter - Launcher Script

echo ================================================
echo Albedo to Normal Map Converter
echo Using Depth Anything v3
echo ================================================
echo.

REM Clear Python cache to ensure latest code is loaded
echo Clearing Python cache...
FOR /d /r . %%d IN (__pycache__) DO @IF EXIST "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul

echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

echo Starting Gradio application with venv Python...
echo.
echo The application will open in your browser at:
echo http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the server
echo.

REM Use venv Python directly to ensure correct version
venv\Scripts\python.exe app.py

pause
