"""
Installation verification script for Albedo to Normal Map Converter
Run this to check if all dependencies are properly installed
"""

import sys

print("="*60)
print("Checking Installation Status")
print("="*60)

# Check Python version
print("\n1. Checking Python version...")
py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# Depth Anything 3 requires Python >= 3.9 and <= 3.13.0
if sys.version_info < (3, 9):
    print(f"   ✗ Python {py_version} (Too old - need 3.9+)")
elif sys.version_info > (3, 13, 0):
    print(f"   ✗ Python {py_version} (Too new - maximum is 3.13.0)")
    print(f"   ⚠ Depth Anything 3 requires Python <= 3.13.0")
    print(f"   → Install Python 3.12 or 3.11 and recreate venv")
    print(f"   → Run: fix_python_version.bat")
elif sys.version_info >= (3, 9):
    print(f"   ✓ Python {py_version} (OK)")
else:
    print(f"   ? Python {py_version}")

# Check PyTorch
print("\n2. Checking PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__} installed")

    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠ CUDA not available (will use CPU - slower)")
except ImportError:
    print("   ✗ PyTorch not installed")
    print("     Install with: pip install torch torchvision")

# Check other dependencies
print("\n3. Checking other dependencies...")

dependencies = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("gradio", "gradio")
]

all_deps_ok = True
for module, package in dependencies:
    try:
        __import__(module)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} not installed")
        all_deps_ok = False

# Check Depth Anything 3
print("\n4. Checking Depth Anything 3...")
try:
    from depth_anything_3.api import DepthAnything3
    print("   ✓ Depth Anything 3 installed")

    # Try to get version if available
    try:
        import depth_anything_3
        if hasattr(depth_anything_3, '__version__'):
            print(f"   ✓ Version: {depth_anything_3.__version__}")
    except:
        pass

except ImportError as e:
    print("   ✗ Depth Anything 3 NOT installed")
    print("\n   To install:")
    print("   1. git clone https://github.com/DepthAnything/Depth-Anything-V3.git")
    print("   2. cd Depth-Anything-V3")
    print("   3. pip install -e \".[app]\"")
    all_deps_ok = False

# Final status
print("\n" + "="*60)
if all_deps_ok:
    print("✓ ALL CHECKS PASSED - Ready to run!")
    print("\nRun the application with:")
    print("  python app.py")
else:
    print("✗ SOME DEPENDENCIES MISSING")
    print("\nPlease install missing dependencies and run this script again.")
    print("\nQuick fix: Run the setup script:")
    print("  setup.bat (Windows) or ./setup.sh (Linux/Mac)")

print("="*60)
