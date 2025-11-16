"""
Albedo to Normal Map Converter using Depth Anything v3
Gradio GUI application for batch processing albedo textures
"""


import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import gradio as gr
from typing import List, Tuple
import traceback

# Import Depth Anything 3
try:
    from depth_anything_3.api import DepthAnything3
except ImportError as e:
    print("\n" + "="*60)
    print("ERROR: Depth Anything 3 is not installed!")
    print("="*60)
    print("\nPlease follow these steps to install it:")
    print("\n1. Clone the repository:")
    print("   git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git")
    print("\n2. Navigate to the directory:")
    print("   cd Depth-Anything-3")
    print("\n3. Install the package:")
    print("   pip install -e \".[app]\"")
    print("\n4. Return to the project directory:")
    print("   cd ..")
    print("\n5. Run the app again:")
    print("   python app.py")
    print("\nOR use the quick installer:")
    print("   install_depth_anything.bat (Windows)")
    print("   ./install_depth_anything.sh (Linux/Mac)")
    print("\n" + "="*60)
    print(f"\nDetailed error: {e}")
    print("="*60 + "\n")
    sys.exit(1)


class AlbedoToNormalConverter:
    """Converts albedo textures to normal maps using Depth Anything v3"""

    def __init__(self, model_name: str = "depth-anything/DA3MONO-LARGE"):
        """
        Initialize the converter with a Depth Anything 3 model

        Args:
            model_name: Name or path of the DA3 model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_name = model_name

    def load_model(self, progress=None):
        """Load the Depth Anything 3 model"""
        if progress:
            progress(0.1, desc="Loading Depth Anything 3 model...")

        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")

        try:
            self.model = DepthAnything3.from_pretrained(self.model_name)
            self.model.to(self.device)
            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def depth_to_normal(
        self,
        depth_map: np.ndarray,
        smooth_kernel_size: int = 5,
        smooth_sigma: float = 1.0,
        depth_scale: float = 1.0,
        albedo_image: np.ndarray = None,
        detail_blend: float = 0.0,
        detail_strength: float = 1.0
    ) -> np.ndarray:
        """
        Convert a depth map to a normal map

        Args:
            depth_map: 2D array of depth values (H, W)
            smooth_kernel_size: Size of Gaussian blur kernel (must be odd)
            smooth_sigma: Sigma value for Gaussian blur
            depth_scale: Multiplier for depth gradients (higher = more pronounced normals)
            albedo_image: Optional RGB albedo image for traditional detail extraction
            detail_blend: Blend factor (0=depth only, 1=traditional only, 0.5=50/50 mix)
            detail_strength: Strength multiplier for traditional detail normals

        Returns:
            Normal map as uint8 RGB image (H, W, 3)
        """
        depth = depth_map.copy()

        # Smooth the depth map to reduce noise
        if smooth_kernel_size > 0:
            # Ensure kernel size is odd
            if smooth_kernel_size % 2 == 0:
                smooth_kernel_size += 1
            depth = cv2.GaussianBlur(
                depth,
                (smooth_kernel_size, smooth_kernel_size),
                smooth_sigma
            )

        H, W = depth.shape

        # Initialize normal components
        Nx = np.zeros((H, W), dtype=np.float32)
        Ny = np.zeros((H, W), dtype=np.float32)
        Nz = np.ones((H, W), dtype=np.float32)

        # Compute central differences for interior pixels
        # Nx = -(dZ/dx), Ny = -(dZ/dy)
        Nx[1:-1, 1:-1] = -(depth[1:-1, 2:] - depth[1:-1, :-2]) * 0.5 * depth_scale
        Ny[1:-1, 1:-1] = -(depth[2:, 1:-1] - depth[:-2, 1:-1]) * 0.5 * depth_scale

        # Blend with traditional normals if albedo image provided
        if albedo_image is not None and detail_blend > 0:
            trad_Nx, trad_Ny, trad_Nz = self.traditional_normals_from_albedo(
                albedo_image, detail_strength
            )
            # Blend the normal components
            Nx = Nx * (1 - detail_blend) + trad_Nx * detail_blend
            Ny = Ny * (1 - detail_blend) + trad_Ny * detail_blend
            Nz = Nz * (1 - detail_blend) + trad_Nz * detail_blend
        
        # Normalize the normal vectors
        norm = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        # Avoid division by zero
        norm = np.maximum(norm, 1e-8)

        Nx /= norm
        Ny /= norm
        Nz /= norm

        # Stack into normal map (H, W, 3) with values in [-1, 1]
        normal_map = np.dstack((Nx, Ny, Nz))

        # Convert to [0, 255] range for visualization
        # (0,0,1) becomes (128,128,255) - a flat surface facing camera
        normal_map_visual = ((normal_map + 1) / 2 * 255).astype(np.uint8)

        return normal_map_visual

    def traditional_normals_from_albedo(
        self,
        albedo_image: np.ndarray,
        detail_strength: float = 1.0
    ) -> np.ndarray:
        """
        Generate normal map from albedo using traditional heightmap technique
        
        Args:
            albedo_image: RGB albedo texture (H, W, 3)
            detail_strength: Strength multiplier for fine details
            
        Returns:
            Normal map components (Nx, Ny, Nz) as float32 arrays
        """
        # Convert to grayscale to use as heightmap
        gray = cv2.cvtColor(albedo_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Apply bilateral filter to preserve edges while smoothing
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # Normalize to 0-1 range
        gray = gray / 255.0
        
        H, W = gray.shape
        
        # Initialize normal components
        Nx = np.zeros((H, W), dtype=np.float32)
        Ny = np.zeros((H, W), dtype=np.float32)
        Nz = np.ones((H, W), dtype=np.float32)
        
        # Compute Sobel gradients for fine detail capture
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Apply detail strength
        Nx = -sobel_x * detail_strength
        Ny = -sobel_y * detail_strength
        
        return Nx, Ny, Nz


    def process_image(
        self,
        image: Image.Image,
        smooth_kernel_size: int = 5,
        smooth_sigma: float = 1.0,
        depth_scale: float = 1.0,
        detail_blend: float = 0.0,
        detail_strength: float = 1.0,
        progress=None
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Process a single albedo image to produce depth and normal maps

        Args:
            image: PIL Image (albedo texture)
            smooth_kernel_size: Smoothing kernel size
            smooth_sigma: Smoothing sigma
            depth_scale: Depth intensity multiplier
            detail_blend: Blend factor for traditional detail normals
            detail_strength: Strength of traditional detail capture
            progress: Gradio progress callback

        Returns:
            Tuple of (depth_map_image, normal_map_image)
        """
        if self.model is None:
            self.load_model(progress)

        # Convert to numpy array
        img_np = np.array(image.convert("RGB"))

        if progress:
            progress(0.3, desc="Running depth estimation...")

        # Run depth inference
        try:
            prediction = self.model.inference([img_np])
            depth = prediction.depth[0]  # Shape: (H, W)
        except Exception as e:
            raise RuntimeError(f"Depth estimation failed: {str(e)}")

        if progress:
            progress(0.7, desc="Computing normal map...")

        # Convert depth to normal map
        normal_map = self.depth_to_normal(
            depth,
            smooth_kernel_size=smooth_kernel_size,
            smooth_sigma=smooth_sigma,
            depth_scale=depth_scale,
            albedo_image=img_np,
            detail_blend=detail_blend,
            detail_strength=detail_strength
        )

        # Normalize depth for visualization
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
        depth_visual = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        depth_visual = cv2.cvtColor(depth_visual, cv2.COLOR_BGR2RGB)

        return Image.fromarray(depth_visual), Image.fromarray(normal_map)

    def process_folder(
        self,
        folder_path: str,
        output_folder: str,
        smooth_kernel_size: int = 5,
        smooth_sigma: float = 1.0,
        depth_scale: float = 1.0,
        detail_blend: float = 0.0,
        detail_strength: float = 1.0,
        save_depth: bool = True,
        progress=gr.Progress()
    ) -> Tuple[str, List[str]]:
        """
        Process all JPG and PNG images in a folder

        Args:
            folder_path: Input folder containing albedo textures
            output_folder: Output folder for generated maps
            smooth_kernel_size: Smoothing kernel size
            smooth_sigma: Smoothing sigma
            depth_scale: Depth intensity multiplier
            detail_blend: Blend factor for traditional detail normals
            detail_strength: Strength of traditional detail capture
            save_depth: Whether to save depth maps
            progress: Gradio progress tracker

        Returns:
            Tuple of (status_message, list_of_processed_files)
        """
        if not os.path.isdir(folder_path):
            return f"Error: '{folder_path}' is not a valid directory", []

        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_files = [
            f for f in Path(folder_path).iterdir()
            if f.suffix in image_extensions and f.is_file()
        ]

        if not image_files:
            return f"No JPG or PNG files found in '{folder_path}'", []

        # Load model once
        if self.model is None:
            progress(0, desc="Loading model...")
            self.load_model()

        processed_files = []
        total = len(image_files)

        for idx, img_path in enumerate(image_files):
            try:
                progress((idx / total), desc=f"Processing {img_path.name} ({idx+1}/{total})...")

                # Load image
                image = Image.open(img_path).convert("RGB")

                # Process
                depth_img, normal_img = self.process_image(
                    image,
                    smooth_kernel_size=smooth_kernel_size,
                    smooth_sigma=smooth_sigma,
                    depth_scale=depth_scale,
                    detail_blend=detail_blend,
                    detail_strength=detail_strength
                )

                # Save outputs
                base_name = img_path.stem

                normal_path = os.path.join(output_folder, f"{base_name}_normal.png")
                normal_img.save(normal_path)

                if save_depth:
                    depth_path = os.path.join(output_folder, f"{base_name}_depth.png")
                    depth_img.save(depth_path)

                processed_files.append(f"✓ {img_path.name} → {base_name}_normal.png")

            except Exception as e:
                error_msg = f"✗ {img_path.name}: {str(e)}"
                processed_files.append(error_msg)
                print(f"Error processing {img_path}: {traceback.format_exc()}")

        progress(1.0, desc="Complete!")

        status = f"Processed {len(image_files)} images. Output saved to: {output_folder}"
        return status, processed_files


def create_gradio_interface():
    """Create and configure the Gradio interface"""

    # Initialize converter
    converter = AlbedoToNormalConverter()

    # Model selection options
    model_choices = [
        "depth-anything/DA3MONO-LARGE",
        "depth-anything/DA3-BASE",
        "depth-anything/DA3-LARGE",
        "depth-anything/DA3NESTED-GIANT-LARGE",
        "depth-anything/DA3METRIC-LARGE"
    ]

    with gr.Blocks(title="Albedo to Normal Map Converter") as app:
        gr.Markdown("""
        # 🎨 Albedo to Normal Map Converter

        Convert albedo textures to normal maps using **Depth Anything v3**.

        ### How to use:
        1. **Single Image Mode**: Upload one image to preview results
        2. **Batch Mode**: Select a folder to process multiple images at once
        """)

        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    single_input = gr.Image(type="pil", label="Input Albedo Texture")

                    with gr.Accordion("Advanced Settings", open=False):
                        single_smooth_kernel = gr.Slider(
                            minimum=0, maximum=15, step=2, value=5,
                            label="Smoothing Kernel Size (0 = no smoothing, must be odd)"
                        )
                        single_smooth_sigma = gr.Slider(
                            minimum=0.1, maximum=5.0, step=0.1, value=1.0,
                            label="Smoothing Sigma"
                        )
                        single_depth_scale = gr.Slider(
                            minimum=0.1, maximum=10.0, step=0.1, value=1.0,
                            label="Depth Intensity (higher = more pronounced normals)",
                            info="Multiplier for depth gradients"
                        )
                        single_detail_blend = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                            label="Detail Blend (0=depth only, 1=texture details only)",
                            info="Mix between depth-based and traditional texture normals"
                        )
                        single_detail_strength = gr.Slider(
                            minimum=0.1, maximum=5.0, step=0.1, value=1.0,
                            label="Detail Strength (fine texture detail intensity)",
                            info="Strength of traditional detail capture"
                        )

                    single_btn = gr.Button("Generate Maps", variant="primary")

                with gr.Column():
                    single_depth_output = gr.Image(label="Depth Map (Visualization)")
                    single_normal_output = gr.Image(label="Normal Map")

            single_btn.click(
                fn=lambda img, k, s, d, db, ds: converter.process_image(img, int(k), s, d, db, ds),
                inputs=[single_input, single_smooth_kernel, single_smooth_sigma, single_depth_scale, single_detail_blend, single_detail_strength],
                outputs=[single_depth_output, single_normal_output]
            )

        with gr.Tab("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    batch_input_folder = gr.Textbox(
                        label="Input Folder Path",
                        placeholder="E:/Textures/Albedo",
                        info="Folder containing JPG or PNG albedo textures"
                    )
                    batch_output_folder = gr.Textbox(
                        label="Output Folder Path",
                        placeholder="E:/Textures/Output",
                        info="Folder where normal maps will be saved"
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        batch_smooth_kernel = gr.Slider(
                            minimum=0, maximum=15, step=2, value=5,
                            label="Smoothing Kernel Size"
                        )
                        batch_smooth_sigma = gr.Slider(
                            minimum=0.1, maximum=5.0, step=0.1, value=1.0,
                            label="Smoothing Sigma"
                        )
                        batch_depth_scale = gr.Slider(
                            minimum=0.1, maximum=10.0, step=0.1, value=1.0,
                            label="Depth Intensity (higher = more pronounced normals)",
                            info="Multiplier for depth gradients"
                        )
                        batch_detail_blend = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                            label="Detail Blend (0=depth only, 1=texture details only)",
                            info="Mix between depth-based and traditional texture normals"
                        )
                        batch_detail_strength = gr.Slider(
                            minimum=0.1, maximum=5.0, step=0.1, value=1.0,
                            label="Detail Strength (fine texture detail intensity)",
                            info="Strength of traditional detail capture"
                        )
                        batch_save_depth = gr.Checkbox(
                            value=True,
                            label="Save Depth Maps"
                        )

                    batch_btn = gr.Button("Process Folder", variant="primary", size="lg")

                with gr.Column():
                    batch_status = gr.Textbox(label="Status", lines=2)
                    batch_results = gr.Textbox(label="Processed Files", lines=15)

            batch_btn.click(
                fn=lambda inp, out, k, s, d, db, ds, save_d: converter.process_folder(
                    inp, out, int(k), s, d, db, ds, save_d
                ),
                inputs=[
                    batch_input_folder,
                    batch_output_folder,
                    batch_smooth_kernel,
                    batch_smooth_sigma,
                    batch_depth_scale,
                    batch_detail_blend,
                    batch_detail_strength,
                    batch_save_depth
                ],
                outputs=[batch_status, batch_results]
            )

        with gr.Tab("Settings"):
            gr.Markdown("### Model Configuration")
            model_select = gr.Dropdown(
                choices=model_choices,
                value="depth-anything/DA3MONO-LARGE",
                label="Depth Anything 3 Model",
                info="Select the model to use (requires reload)"
            )

            def change_model(model_name):
                converter.model_name = model_name
                converter.model = None  # Force reload
                return f"Model changed to: {model_name}. Will reload on next inference."

            model_status = gr.Textbox(label="Status", interactive=False)
            model_select.change(
                fn=change_model,
                inputs=[model_select],
                outputs=[model_status]
            )

            gr.Markdown("""
            ### Model Descriptions:
            - **DA3MONO-LARGE**: High-quality monocular depth (recommended, Apache 2.0 license)
            - **DA3-BASE**: Faster, lighter model (Apache 2.0 license)
            - **DA3-LARGE**: Higher quality (CC BY-NC 4.0 - non-commercial)
            - **DA3NESTED-GIANT-LARGE**: Highest quality, very large (CC BY-NC 4.0)
            - **DA3METRIC-LARGE**: Metric depth estimation (Apache 2.0 license)

            ### GPU Status:
            """)

            device_info = f"**Device**: {converter.device}\n\n"
            if torch.cuda.is_available():
                device_info += f"**GPU**: {torch.cuda.get_device_name(0)}\n\n"
                device_info += f"**CUDA Version**: {torch.version.cuda}"
            else:
                device_info += "⚠️ **No GPU detected** - Processing will be slower on CPU"

            gr.Markdown(device_info)

        gr.Markdown("""
        ---
        ### Tips:
        - **Smoothing**: Higher values reduce noise but may lose detail
        - **Output Format**: Normal maps use standard format (128,128,255 = flat surface)
        - **Performance**: First run will download the model (~1-2GB)
        - **Batch Mode**: Output files will be named `<original_name>_normal.png`
        """)

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("Albedo to Normal Map Converter")
    print("Using Depth Anything v3")
    print("=" * 60)

    app = create_gradio_interface()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
