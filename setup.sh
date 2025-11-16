#!/bin/bash
# Setup script for Albedo to Normal Map Converter

echo "================================================"
echo "Albedo to Normal Map Converter - Setup"
echo "================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing PyTorch..."
read -p "Do you have an NVIDIA GPU with CUDA support? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision
fi

echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "Cloning Depth Anything V3..."
if [ -d "Depth-Anything-3" ]; then
    echo "Depth-Anything-3 already exists, skipping clone..."
else
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
fi

echo ""
echo "Installing Depth Anything V3..."
cd Depth-Anything-3
pip install -e ".[app]"
cd ..

echo ""
echo "Making scripts executable..."
chmod +x run.sh

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To run the application, execute: ./run.sh"
echo "Or manually: python app.py"
echo ""
