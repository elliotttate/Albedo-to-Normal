@echo off
REM Script to recreate virtual environment with compatible Python version

echo ================================================
echo Python Version Fix Tool
echo ================================================
echo.

echo Current Python version:
python --version
echo.

echo Checking available Python versions...
py --list
echo.

echo ================================================
echo Depth Anything 3 requires Python 3.9 to 3.13.0
echo Your Python 3.13.5 is too new!
echo ================================================
echo.

echo Recommended: Install Python 3.12.10
echo Download from: https://www.python.org/downloads/release/python-31210/
echo.

echo After installing Python 3.12, run this script again
echo or manually recreate the virtual environment:
echo.
echo   rmdir /s /q venv
echo   py -3.12 -m venv venv
echo   venv\Scripts\activate
echo   install_depth_anything.bat
echo.

pause

echo.
echo Do you want to try recreating the virtual environment now?
echo This will delete the current venv folder!
echo.
choice /C YN /M "Continue"

if errorlevel 2 goto :end
if errorlevel 1 goto :recreate

:recreate
echo.
echo Checking for Python 3.12...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 not found!
    echo.
    echo Checking for Python 3.11...
    py -3.11 --version >nul 2>&1
    if errorlevel 1 (
        echo Python 3.11 not found!
        echo.
        echo Please install Python 3.12 or 3.11 first:
        echo https://www.python.org/downloads/
        pause
        exit /b 1
    ) else (
        echo Found Python 3.11!
        set PYTHON_CMD=py -3.11
        py -3.11 --version
    )
) else (
    echo Found Python 3.12!
    set PYTHON_CMD=py -3.12
    py -3.12 --version
)

echo.
echo Removing old virtual environment...
if exist "venv" (
    rmdir /s /q venv
    echo Done!
) else (
    echo No venv folder found, skipping...
)

echo.
echo Creating new virtual environment with compatible Python...
%PYTHON_CMD% -m venv venv

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Verifying Python version...
python --version

echo.
echo Virtual environment created successfully!
echo.
echo ================================================
echo Next Steps:
echo ================================================
echo.
echo 1. Install PyTorch:
echo    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.
echo 2. Install dependencies:
echo    pip install -r requirements.txt
echo.
echo 3. Install Depth Anything 3:
echo    install_depth_anything.bat
echo.
echo OR run the full setup:
echo    setup.bat
echo.

pause
goto :end

:end
echo.
echo Done!
