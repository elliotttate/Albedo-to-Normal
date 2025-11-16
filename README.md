# Albedo to Normal Map Converter

A Gradio-based GUI application that converts albedo textures to normal maps using **Depth Anything v3** for depth estimation.

## 🚨 Common Installation Issues

### Issue 1: Python Version Error

```
ERROR: Package 'depth-anything-3' requires a different Python: 3.13.5 not in '<=3.13,>=3.9'
```

**Solution:** Install Python 3.12 or 3.11

1. Download Python 3.12: https://www.python.org/downloads/release/python-31210/
2. Run: `fix_python_version.bat` (recreates venv with correct Python)
3. Or see [PYTHON_VERSION_FIX.md](PYTHON_VERSION_FIX.md) for details

**Compatible Python versions:** 3.9, 3.10, 3.11, 3.12, 3.13.0 (not 3.13.1+)

---

### Issue 2: Depth Anything 3 Not Installed

**Windows:**
```cmd
install_depth_anything.bat
```

**Linux/Mac:**
```bash
chmod +x install_depth_anything.sh
./install_depth_anything.sh
```

Or see [INSTALL.md](INSTALL.md) for detailed installation instructions.

---

## Features

✨ **Dual Processing Modes**
- Single image preview mode with instant results
- Batch processing mode for entire folders

🎨 **High-Quality Normal Maps**
- Uses state-of-the-art Depth Anything v3 for depth estimation
- Configurable smoothing parameters
- Standard normal map format output

🖥️ **User-Friendly Interface**
- Simple web-based GUI built with Gradio
- Real-time progress tracking
- Support for JPG and PNG formats

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended) or CPU
- 4GB+ GPU memory for large models

### Step 1: Clone or Create Project Directory

```bash
cd E:\Github\albedo_to_normal
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/Mac
```

### Step 3: Install PyTorch with CUDA Support

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and install the appropriate version.

For Windows with CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:
```bash
pip install torch torchvision
```

### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install Depth Anything v3

```bash
# Clone the repository
git clone https://github.com/DepthAnything/Depth-Anything-V3.git
cd Depth-Anything-V3

# Install in editable mode
pip install -e .

# Optional: Install with app support (includes Gradio)
pip install -e ".[app]"

cd ..
```

## Usage

### Running the Application

```bash
python app.py
```

The application will start and open in your default browser at `http://127.0.0.1:7860`

### Single Image Mode

1. Go to the **"Single Image"** tab
2. Upload an albedo texture (JPG or PNG)
3. Adjust smoothing parameters if needed (Advanced Settings)
4. Click **"Generate Maps"**
5. View the depth map and normal map outputs

### Batch Processing Mode

1. Go to the **"Batch Processing"** tab
2. Enter the input folder path containing your albedo textures
   - Example: `E:/Textures/Albedo`
3. Enter the output folder path where results will be saved
   - Example: `E:/Textures/Output`
4. Configure settings:
   - **Smoothing Kernel Size**: 0-15 (higher = smoother, less detail)
   - **Smoothing Sigma**: 0.1-5.0 (controls blur strength)
   - **Save Depth Maps**: Check to also save depth visualizations
5. Click **"Process Folder"**
6. Monitor progress and review results

### Advanced Settings

#### Smoothing Parameters

- **Kernel Size**: Controls the size of the Gaussian blur filter
  - `0` = No smoothing (may be noisy)
  - `5` = Default (balanced)
  - `9-15` = Heavy smoothing (removes detail but reduces noise)

- **Sigma**: Controls the spread of the Gaussian blur
  - Lower values (0.1-1.0): Subtle smoothing
  - Higher values (2.0-5.0): Aggressive smoothing

#### Model Selection

Available models (in Settings tab):

| Model | Size | Quality | License | Use Case |
|-------|------|---------|---------|----------|
| DA3MONO-LARGE | 0.35B | High | Apache 2.0 | **Recommended** - Best balance |
| DA3-BASE | 0.12B | Good | Apache 2.0 | Faster, lighter |
| DA3-LARGE | 0.35B | High | CC BY-NC 4.0 | Non-commercial use |
| DA3NESTED-GIANT-LARGE | 1.40B | Highest | CC BY-NC 4.0 | Best quality, requires 8GB+ VRAM |
| DA3METRIC-LARGE | 0.35B | High | Apache 2.0 | Metric depth estimation |

## Output Format

### Normal Maps

- **Format**: PNG (RGB, 8-bit per channel)
- **Convention**: Tangent-space normal map
  - Red channel (X): Left (-1) to Right (+1)
  - Green channel (Y): Bottom (-1) to Top (+1)
  - Blue channel (Z): Into surface (-1) to Out of surface (+1)
- **Flat surface**: RGB(128, 128, 255) - bluish color
- **Naming**: `<original_filename>_normal.png`

### Depth Maps (Optional)

- **Format**: PNG with colormap (Inferno) for visualization
- **Naming**: `<original_filename>_depth.png`

## Troubleshooting

### Common Issues

#### 1. "CUDA out of memory" Error

**Solution**: Use a smaller model or reduce image resolution
- Switch to `DA3-BASE` in Settings tab
- Or use CPU mode (slower but works)

#### 2. Model Download Fails

**Solution**: Manual download
```bash
# Download from Hugging Face manually
# Place in cache directory or specify local path in Settings
```

#### 3. "No module named 'depth_anything_3'" Error

**Solution**: Ensure Depth Anything V3 is installed
```bash
cd Depth-Anything-V3
pip install -e .
```

#### 4. Slow Processing on CPU

**Solution**: This is expected - GPU is highly recommended
- Consider using `DA3-BASE` model for faster CPU processing
- Or install CUDA-enabled PyTorch

#### 5. Normal Map Looks Too Noisy

**Solution**: Increase smoothing parameters
- Try kernel size: 7 or 9
- Try sigma: 1.5 or 2.0

#### 6. Normal Map Looks Too Flat

**Solution**: The input image may lack depth cues
- Albedo textures without shading are harder to estimate
- Try a different model (e.g., DA3NESTED-GIANT-LARGE)
- Consider adding subtle lighting to the input

## Technical Details

### Pipeline Overview

1. **Input**: Albedo texture (RGB image)
2. **Depth Estimation**: Depth Anything v3 predicts depth map
3. **Smoothing**: Optional Gaussian blur to reduce noise
4. **Normal Calculation**: Compute surface normals from depth gradients
   - Uses central differences: `N = (-dZ/dx, -dZ/dy, 1)`
   - Normalized to unit vectors
5. **Output**: Normal map in standard format (0-255 RGB)

### Limitations

- **Monocular Depth Uncertainty**: Single-image depth has inherent ambiguity
- **Scale Ambiguity**: Depth scale may not be accurate (doesn't affect normals)
- **Texture Misinterpretation**: Flat surfaces with complex textures may show false depth
- **Edge Artifacts**: Discontinuities at object boundaries may cause noise

## Performance

### Expected Processing Times (RTX 3080)

| Model | Single 1K Image | Single 2K Image |
|-------|----------------|----------------|
| DA3-BASE | ~1-2 sec | ~3-4 sec |
| DA3MONO-LARGE | ~2-3 sec | ~5-7 sec |
| DA3NESTED-GIANT-LARGE | ~5-8 sec | ~12-15 sec |

*First run will be slower due to model download*

## Examples

### Input → Output

```
input_folder/
  ├── brick_albedo.png
  ├── wood_albedo.jpg
  └── stone_albedo.png

↓ Process Folder ↓

output_folder/
  ├── brick_albedo_normal.png
  ├── brick_albedo_depth.png
  ├── wood_albedo_normal.png
  ├── wood_albedo_depth.png
  ├── stone_albedo_normal.png
  └── stone_albedo_depth.png
```

## License

This application code is provided as-is. Please note:

- **Depth Anything V3**: Check the official repository for model-specific licenses
  - Apache 2.0: DA3-BASE, DA3MONO-LARGE, DA3METRIC-LARGE
  - CC BY-NC 4.0: DA3-LARGE, DA3NESTED-GIANT-LARGE (non-commercial only)
- **Dependencies**: See individual package licenses

## Credits

- **Depth Anything V3**: ByteDance Research
  - Repository: https://github.com/DepthAnything/Depth-Anything-V3
- **Gradio**: Gradio Team
- **PyTorch**: Meta AI

## Support

For issues related to:
- **This GUI app**: Check the troubleshooting section above
- **Depth Anything V3**: Visit the [official repository](https://github.com/DepthAnything/Depth-Anything-V3)
- **Model accuracy**: Try different models or adjust smoothing parameters

## Citation

If you use this tool in research, please cite Depth Anything V3:

```bibtex
@article{depth_anything_v3,
  title={Depth Anything V3},
  author={ByteDance Research},
  year={2025}
}
```

---

**Version**: 1.0.0
**Last Updated**: 2025-01-16
