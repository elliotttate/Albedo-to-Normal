# Quick Start Guide

Get up and running with the Albedo to Normal Map Converter in minutes!

## 🚀 Installation (5 minutes)

### Windows

1. **Run the setup script:**
   ```cmd
   setup.bat
   ```
   This will:
   - Create a virtual environment
   - Install PyTorch (with CUDA if available)
   - Install all dependencies
   - Clone and install Depth Anything V3

2. **Launch the application:**
   ```cmd
   run.bat
   ```

### Linux/Mac

1. **Make scripts executable:**
   ```bash
   chmod +x setup.sh run.sh
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

3. **Launch the application:**
   ```bash
   ./run.sh
   ```

## 📝 First Run

1. The application will open in your browser at `http://127.0.0.1:7860`

2. On first use, the model will be downloaded (~400MB-2GB depending on model)
   - This happens automatically
   - Only needed once
   - Stored in your HuggingFace cache

## 🎯 Basic Usage

### Quick Test (Single Image)

1. Go to **"Single Image"** tab
2. Upload any image (JPG or PNG)
3. Click **"Generate Maps"**
4. Wait 5-10 seconds
5. View your normal map!

### Batch Processing

1. Go to **"Batch Processing"** tab
2. Enter input folder: `C:\MyTextures\Input`
3. Enter output folder: `C:\MyTextures\Output`
4. Click **"Process Folder"**
5. Check the output folder for results

## ⚙️ Recommended Settings

### For Best Quality
- **Model**: DA3MONO-LARGE (default)
- **Smoothing Kernel**: 5
- **Smoothing Sigma**: 1.0

### For Speed
- **Model**: DA3-BASE
- **Smoothing Kernel**: 3
- **Smoothing Sigma**: 0.8

### For Noisy Results
- **Smoothing Kernel**: 7 or 9
- **Smoothing Sigma**: 1.5 or 2.0

## 🐛 Common First-Run Issues

### "No module named 'depth_anything_3'"

**Fix:**
```bash
cd Depth-Anything-V3
pip install -e ".[app]"
```

### Gradio won't start

**Fix:**
```bash
pip install gradio>=4.0.0
```

### CUDA out of memory

**Fix:** Switch to smaller model in Settings tab (DA3-BASE)

### Slow on CPU

**Expected** - GPU is highly recommended. Install CUDA-enabled PyTorch if you have an NVIDIA GPU.

## 📊 What to Expect

### Processing Times (RTX 3080)
- Single 1024x1024 image: ~2-3 seconds
- Batch of 10 images: ~20-30 seconds

### Output Files
Input: `brick_texture.png`

Output:
- `brick_texture_normal.png` (normal map)
- `brick_texture_depth.png` (depth visualization, optional)

## 🎨 Understanding Results

### Good Normal Map
- Smooth gradients
- Blue-ish color overall (flat areas)
- Red/green tints show surface angles
- Detail matches texture

### Poor Normal Map
- Very noisy (increase smoothing)
- All flat blue (input lacks depth cues)
- Weird artifacts (try different model)

## 💡 Tips

1. **First image takes longer** - Model is loading
2. **Use GPU** - 10-20x faster than CPU
3. **Higher resolution = more detail** - But slower
4. **Test settings on one image** - Before batch processing
5. **Albedo with lighting works better** - Pure flat color is harder

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Try different models in the Settings tab
- Experiment with smoothing parameters
- Check the [Troubleshooting](README.md#troubleshooting) section if needed

## 🆘 Getting Help

1. Check [README.md](README.md#troubleshooting)
2. Verify installation: `python -c "import depth_anything_3; print('OK')"`
3. Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

**Ready?** Run `run.bat` (Windows) or `./run.sh` (Linux/Mac) and start converting! 🚀
