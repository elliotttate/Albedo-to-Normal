@echo off
REM Opens Python 3.12 download page in browser

echo ================================================
echo Opening Python 3.12 Download Page
echo ================================================
echo.
echo This will open the Python 3.12.10 download page
echo in your web browser.
echo.
echo After downloading:
echo 1. Run the installer
echo 2. Check "Add Python to PATH"
echo 3. Complete installation
echo 4. Run: fix_python_version.bat
echo.
echo Press any key to open download page...
pause >nul

start https://www.python.org/downloads/release/python-31210/

echo.
echo Download page opened in browser!
echo.
echo Look for: Windows installer (64-bit)
echo File: python-3.12.10-amd64.exe
echo.
pause
