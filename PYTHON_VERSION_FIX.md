# Python Version Issue Fix

## Problem

You have Python **3.13.5**, but Depth Anything 3 requires Python **<= 3.13** (maximum 3.13.0).

```
ERROR: Package 'depth-anything-3' requires a different Python: 3.13.5 not in '<=3.13,>=3.9'
```

## Solution Options

### Option 1: Install Python 3.12 (Recommended)

Python 3.12 is the most stable version for this project.

#### Step 1: Download Python 3.12

Download from: https://www.python.org/downloads/release/python-31210/

**Windows:**
- Scroll down to "Files"
- Download: `Windows installer (64-bit)`
- Filename: `python-3.12.10-amd64.exe`

#### Step 2: Install Python 3.12

1. Run the installer
2. ✅ **IMPORTANT**: Check "Add Python 3.12 to PATH"
3. Click "Install Now"

#### Step 3: Recreate Virtual Environment

```cmd
cd E:\Github\albedo_to_normal

# Remove old virtual environment
rmdir /s /q venv

# Create new virtual environment with Python 3.12
py -3.12 -m venv venv

# Activate it
venv\Scripts\activate

# Verify Python version (should show 3.12.x)
python --version

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install Depth Anything 3
install_depth_anything.bat
```

---

### Option 2: Use Python 3.11 (Also Compatible)

Python 3.11 is also fully compatible.

Download from: https://www.python.org/downloads/release/python-31111/

Then follow the same steps as Option 1, replacing `py -3.12` with `py -3.11`.

---

### Option 3: Quick Fix - Use System Python Launcher

If you have multiple Python versions installed:

```cmd
cd E:\Github\albedo_to_normal

# List available Python versions
py --list

# Create venv with Python 3.12 (if installed)
py -3.12 -m venv venv

# Or Python 3.11
py -3.11 -m venv venv

# Activate and continue
venv\Scripts\activate
```

---

## After Installing Correct Python Version

1. **Recreate the virtual environment** (see commands above)
2. **Run the installer:**
   ```cmd
   install_depth_anything.bat
   ```
3. **Launch the app:**
   ```cmd
   run.bat
   ```

---

## Checking Your Python Version

```cmd
# Check system Python
python --version

# Check Python in virtual environment
venv\Scripts\activate
python --version
```

You should see: `Python 3.12.x` or `Python 3.11.x`

---

## Why This Happens

Python 3.13.5 is **newer** than the maximum supported version (3.13.0). Some packages have strict version requirements to ensure compatibility.

**Compatible versions for this project:**
- ✅ Python 3.9.x
- ✅ Python 3.10.x
- ✅ Python 3.11.x (Recommended)
- ✅ Python 3.12.x (Recommended)
- ✅ Python 3.13.0
- ❌ Python 3.13.1+

---

## Quick Reference

| Python Version | Compatible? | Recommended? |
|----------------|-------------|--------------|
| 3.9.x | ✅ Yes | ⚠️ Older |
| 3.10.x | ✅ Yes | ⚠️ Older |
| 3.11.x | ✅ Yes | ⭐ **Best** |
| 3.12.x | ✅ Yes | ⭐ **Best** |
| 3.13.0 | ✅ Yes | ⚠️ Edge case |
| 3.13.1+ | ❌ No | ❌ Too new |

---

## Need Help?

If you're stuck:
1. Download Python 3.12.10 from the link above
2. Install it (check "Add to PATH")
3. Delete the `venv` folder
4. Run: `py -3.12 -m venv venv`
5. Run: `venv\Scripts\activate`
6. Run: `install_depth_anything.bat`
