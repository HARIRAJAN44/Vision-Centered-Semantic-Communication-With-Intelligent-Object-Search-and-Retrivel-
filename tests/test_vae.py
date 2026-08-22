import torch
import numpy as np
from PIL import Image
from semantic.encoder import SemanticEncoder
from semantic.decoder import SemanticDecoder
from preprocessing.image_utils import preprocess_image, postprocess_image

def test_vae_shape():
    encoder = SemanticEncoder()
    decoder = SemanticDecoder()
    
    # Create a dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # Preprocess
    img_tensor = preprocess_image(dummy_img)
    assert img_tensor.shape == (1, 3, 512, 512)
    print(f"Preprocessed image shape: {img_tensor.shape}")
    
    # Encode
    z = encoder.encode(img_tensor)
    assert z.shape == (1, 4, 64, 64)
    print(f"Latent shape: {z.shape}")
    
    # Decode
    reconstructed = decoder.decode(z)
    assert reconstructed.shape == (1, 3, 512, 512)
    print(f"Reconstructed image shape: {reconstructed.shape}")
    
    # Postprocess
    out_img = postprocess_image(reconstructed)
    assert out_img.size == (512, 512)
    print(f"Output image size: {out_img.size}")
    
    print("VAE shape test passed successfully.")

if __name__ == "__main__":
    test_vae_shape()
