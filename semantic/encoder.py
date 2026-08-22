import torch
from diffusers import AutoencoderKL

class SemanticEncoder:
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Encoder] Loading VAE from {model_name} on {self.device}...")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.vae.to(self.device)
        self.vae.eval() # Inference mode

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes a normalized image tensor into a latent representation.
        x: Image tensor of shape (B, C, H, W) normalized to [-1, 1].
        Returns: Latent tensor z of shape (B, 4, H/8, W/8).
        """
        x = x.to(self.device)
        # Obtain posterior distribution
        posterior = self.vae.encode(x).latent_dist
        
        # Sample from posterior using reparameterization trick
        # z = mu(x) + sigma(x) * eps
        z = posterior.sample()
        
        return z
