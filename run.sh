#!/bin/bash
# Albedo to Normal Map Converter - Launcher Script

echo "================================================"
echo "Albedo to Normal Map Converter"
echo "Using Depth Anything v3"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found at venv/"
    echo "Running with system Python..."
fi

echo ""
echo "Starting Gradio application..."
echo ""
echo "The application will open in your browser at:"
echo "http://127.0.0.1:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
