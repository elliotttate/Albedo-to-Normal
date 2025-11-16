@echo off
REM Albedo to Normal Map Converter - Launcher Script

echo ================================================
echo Albedo to Normal Map Converter
echo Using Depth Anything v3
echo ================================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at venv\
    echo Running with system Python...
)

echo.
echo Starting Gradio application...
echo.
echo The application will open in your browser at:
echo http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
