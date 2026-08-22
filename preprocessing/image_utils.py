import torch
import numpy as np
from PIL import Image

def load_image(image_path: str) -> Image.Image:
    """Loads an image from path and converts to RGB."""
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return image

def preprocess_image(image: Image.Image, width: int = 512, height: int = 512) -> torch.Tensor:
    """
    Preprocesses the PIL image to a PyTorch tensor suitable for Stable Diffusion VAE.
    - Resizes to specified dimensions
    - Converts to numpy array
    - Scales values to [-1, 1]
    - Converts to tensor of shape (1, C, H, W)
    """
    image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = (image_np * 2.0) - 1.0  # Scale to [-1, 1]
    image_tensor = torch.from_numpy(image_np)
    # HWC to CHW
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    return image_tensor

def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """
    Postprocesses a PyTorch tensor from Stable Diffusion VAE to a PIL Image.
    - Scales values from [-1, 1] to [0, 255]
    - Converts back to PIL Image
    """
    image_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np = (image_np + 1.0) / 2.0  # Scale to [0, 1]
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255.0).astype(np.uint8)
    return Image.fromarray(image_np)

def save_image(image: Image.Image, output_path: str):
    """Saves a PIL Image to path."""
    image.save(output_path)
