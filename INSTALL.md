# Installation Guide

Step-by-step installation instructions for the Albedo to Normal Map Converter.

## ⚠️ IMPORTANT: Python Version Requirement

**Depth Anything 3 requires Python >= 3.9 and <= 3.13.0**

❌ **Python 3.13.1 or newer is NOT compatible**

✅ **Recommended versions:**
- Python 3.12.x (Best choice)
- Python 3.11.x (Best choice)
- Python 3.10.x
- Python 3.9.x

If you have Python 3.13.1+, see [PYTHON_VERSION_FIX.md](PYTHON_VERSION_FIX.md) for instructions.

---

## Prerequisites

- **Python 3.9 - 3.13.0** ([Download Python 3.12](https://www.python.org/downloads/release/python-31210/))
- **Git** ([Download](https://git-scm.com/downloads))
- **NVIDIA GPU with CUDA** (optional but highly recommended)

## Installation Methods

Choose one of the following methods:

---

## Method 1: Automated Setup (Recommended)

### Windows

1. Open Command Prompt in the project folder:
   ```cmd
   cd E:\Github\albedo_to_normal
   ```

2. Run the setup script:
   ```cmd
   setup.bat
   ```

3. Follow the prompts (select GPU support if you have NVIDIA GPU)

4. Wait for installation to complete (~5-10 minutes)

### Linux/Mac

1. Open Terminal in the project folder:
   ```bash
   cd E:/Github/albedo_to_normal
   ```

2. Make scripts executable:
   ```bash
   chmod +x setup.sh run.sh
   ```

3. Run the setup script:
   ```bash
   ./setup.sh
   ```

4. Follow the prompts

---

## Method 2: Manual Installation

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

**Activate it:**

Windows:
```cmd
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### Step 2: Install PyTorch

**With CUDA 12.1 (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**With CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only (no GPU):**
```bash
pip install torch torchvision
```

> Check [PyTorch website](https://pytorch.org/get-started/locally/) for other CUDA versions

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- numpy
- opencv-python
- Pillow
- gradio

### Step 4: Install Depth Anything V3

This is the most important step!

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
   ```

2. **Navigate to the directory:**
   ```bash
   cd Depth-Anything-3
   ```

3. **Install the package:**
   ```bash
   pip install -e ".[app]"
   ```

   > Note: The quotes around `".[app]"` are important on Windows!

4. **Return to project directory:**
   ```bash
   cd ..
   ```

### Step 5: Verify Installation

Run the verification script:
```bash
python check_install.py
```

You should see all checkmarks (✓). If you see any (✗), install the missing dependencies.

---

## Troubleshooting Installation

### Issue: "git is not recognized"

**Fix:** Install Git from https://git-scm.com/downloads

### Issue: "python is not recognized"

**Fix:**
1. Install Python from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"

### Issue: "No module named 'depth_anything_3'"

**Fix:**
```bash
cd Depth-Anything-3
pip install -e ".[app]"
cd ..
```

If Depth-Anything-3 folder doesn't exist:
```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e ".[app]"
cd ..
```

### Issue: "ERROR: Could not find a version that satisfies the requirement torch"

**Fix:** Update pip first:
```bash
python -m pip install --upgrade pip
```

Then try installing PyTorch again.

### Issue: CUDA installation fails

**Fix:** Install CPU version first to test:
```bash
pip install torch torchvision
```

Then verify your NVIDIA drivers are up to date.

### Issue: "ModuleNotFoundError: No module named 'gradio'"

**Fix:**
```bash
pip install gradio>=4.0.0
```

### Issue: Virtual environment activation fails

**Windows PowerShell:** You may need to allow script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

---

## Verification

After installation, run this command to verify everything is working:

```bash
python check_install.py
```

**Expected output:**
```
============================================================
Checking Installation Status
============================================================

1. Checking Python version...
   ✓ Python 3.10.x (OK)

2. Checking PyTorch...
   ✓ PyTorch 2.x.x installed
   ✓ CUDA available: NVIDIA GeForce RTX 3080
   ✓ CUDA version: 12.1

3. Checking other dependencies...
   ✓ numpy
   ✓ opencv-python
   ✓ Pillow
   ✓ gradio

4. Checking Depth Anything 3...
   ✓ Depth Anything 3 installed

============================================================
✓ ALL CHECKS PASSED - Ready to run!

Run the application with:
  python app.py
============================================================
```

---

## Post-Installation

Once all checks pass:

1. **Run the application:**
   ```bash
   python app.py
   ```

   Or use the launcher:
   - Windows: `run.bat`
   - Linux/Mac: `./run.sh`

2. **Open your browser** at: http://127.0.0.1:7860

3. **First run will download the model** (~400MB-2GB)
   - This is automatic
   - Only happens once
   - Stored in HuggingFace cache

---

## Disk Space Requirements

- **Python packages:** ~3-5 GB
- **Depth Anything models:** 400MB - 2GB (downloaded on first use)
- **Virtual environment:** ~1-2 GB

**Total:** ~5-10 GB

---

## Optional: Using Different CUDA Versions

If you have a different CUDA version installed:

1. Check your CUDA version:
   ```bash
   nvidia-smi
   ```

2. Install matching PyTorch from [PyTorch website](https://pytorch.org/get-started/locally/)

Example for CUDA 11.7:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

---

## Need Help?

1. Run `python check_install.py` to diagnose issues
2. Check [README.md](README.md#troubleshooting) for troubleshooting
3. Ensure you're in the virtual environment (you should see `(venv)` in your prompt)

---

**Next:** Once installation is complete, see [QUICKSTART.md](QUICKSTART.md) to start using the app!
