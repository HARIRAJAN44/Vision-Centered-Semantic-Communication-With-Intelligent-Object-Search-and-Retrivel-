import torch
from diffusers import AutoencoderKL

class SemanticDecoder:
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Decoder] Loading VAE from {model_name} on {self.device}...")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.vae.to(self.device)
        self.vae.eval() # Inference mode

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent representation back into an image tensor.
        z: Latent tensor of shape (B, 4, H/8, W/8).
        Returns: Image tensor of shape (B, 3, H, W) normalized to [-1, 1].
        """
        z = z.to(self.device)
        
        # Decode the latent tensor
        output = self.vae.decode(z)
        x_hat = output.sample
        
        return x_hat
