@echo off
REM Setup script for Albedo to Normal Map Converter

echo ================================================
echo Albedo to Normal Map Converter - Setup
echo ================================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing PyTorch with CUDA support...
echo If you don't have a CUDA GPU, you can skip this and install CPU-only version
echo.
choice /C YN /M "Do you have an NVIDIA GPU with CUDA support"

if errorlevel 2 goto cpu_only
if errorlevel 1 goto cuda_install

:cuda_install
echo.
echo Installing PyTorch with CUDA 12.1...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
goto install_deps

:cpu_only
echo.
echo Installing PyTorch (CPU only)...
pip install torch torchvision

:install_deps
echo.
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo Cloning Depth Anything V3...
if exist "Depth-Anything-3" (
    echo Depth-Anything-3 already exists, skipping clone...
) else (
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
)

echo.
echo Installing Depth Anything V3...
cd Depth-Anything-3
pip install -e ".[app]"
cd ..

echo.
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo To run the application, execute: run.bat
echo Or manually: python app.py
echo.
pause
