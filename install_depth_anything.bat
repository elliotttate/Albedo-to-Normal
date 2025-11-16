@echo off
REM Quick installer for Depth Anything V3 only

echo ================================================
echo Installing Depth Anything V3
echo ================================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found
    echo Using system Python...
)

echo.
echo Checking if Depth-Anything-3 directory exists...
if exist "Depth-Anything-3" (
    echo Directory already exists, using existing clone...
) else (
    echo Cloning Depth Anything V3 repository...
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to clone repository
        echo Make sure Git is installed: https://git-scm.com/downloads
        pause
        exit /b 1
    )
)

echo.
echo Installing Depth Anything V3...
cd Depth-Anything-3
pip install -e ".[app]"

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed
    echo Try installing dependencies first:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

cd ..

echo.
echo ================================================
echo Installation Complete!
echo ================================================
echo.
echo Verifying installation...
python check_install.py

echo.
echo You can now run the application with: run.bat
echo.
pause
