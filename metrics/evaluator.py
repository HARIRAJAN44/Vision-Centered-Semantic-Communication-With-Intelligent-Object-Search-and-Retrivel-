import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

class Evaluator:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load LPIPS model
        print(f"[Evaluator] Loading LPIPS on {self.device}...")
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_vgg.eval()
        
        # Load CLIP model
        print(f"[Evaluator] Loading CLIP on {self.device}...")
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

    def calc_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculates Structural Similarity Index."""
        img1_np = np.array(img1.convert("L")) # SSIM usually on luminance/grayscale for structure
        img2_np = np.array(img2.convert("L"))
        
        # skimage SSIM
        score, _ = ssim(img1_np, img2_np, full=True, data_range=255)
        return float(score)
        
    def calc_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculates LPIPS."""
        # Convert PIL to tensor [-1, 1]
        img1_t = (torch.from_numpy(np.array(img1)).float() / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img2_t = (torch.from_numpy(np.array(img2)).float() / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.lpips_vgg(img1_t, img2_t)
            
        return score.item()
        
    def calc_clip(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculates CLIP cosine similarity."""
        inputs1 = self.clip_processor(images=[img1], return_tensors="pt").to(self.device)
        inputs2 = self.clip_processor(images=[img2], return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat1 = self.clip_model(pixel_values=inputs1["pixel_values"]).image_embeds
            feat2 = self.clip_model(pixel_values=inputs2["pixel_values"]).image_embeds
            # normalize
            feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
            # cosine similarity between img1 and img2
            sim = torch.nn.functional.cosine_similarity(feat1, feat2)
        return sim.item()
        
    def calc_psnr(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
        img1_np = np.array(img1.convert("RGB"), dtype=np.float32)
        img2_np = np.array(img2.convert("RGB"), dtype=np.float32)
        mse = np.mean((img1_np - img2_np) ** 2)
        if mse <= 1e-10:
            return 100.0
        return float(20 * np.log10(255.0 / np.sqrt(mse)))
        
    def evaluate(self, img1: Image.Image, img2: Image.Image) -> dict:
        """Evaluates all metrics for a single image pair."""
        clip_score = self.calc_clip(img1, img2)
        return {
            'ssim': self.calc_ssim(img1, img2),
            'psnr': self.calc_psnr(img1, img2),
            'lpips': self.calc_lpips(img1, img2),
            'clip': clip_score,
            'clip_similarity': clip_score,
        }
