#!/bin/bash
# Quick installer for Depth Anything V3 only

echo "================================================"
echo "Installing Depth Anything V3"
echo "================================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found"
    echo "Using system Python..."
fi

echo ""
echo "Checking if Depth-Anything-3 directory exists..."
if [ -d "Depth-Anything-3" ]; then
    echo "Directory already exists, using existing clone..."
else
    echo "Cloning Depth Anything V3 repository..."
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to clone repository"
        echo "Make sure Git is installed"
        exit 1
    fi
fi

echo ""
echo "Installing Depth Anything V3..."
cd Depth-Anything-3
pip install -e ".[app]"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Installation failed"
    echo "Try installing dependencies first:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

cd ..

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Verifying installation..."
python check_install.py

echo ""
echo "You can now run the application with: ./run.sh"
echo ""
