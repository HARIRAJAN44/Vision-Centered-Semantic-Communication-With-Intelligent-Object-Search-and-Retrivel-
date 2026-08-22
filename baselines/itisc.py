from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
from communication.channel_pipeline import ChannelPipeline

class ItISCBaseline:
    """
    Image-Text-Image Semantic Communication (ItISC) baseline.
    Image -> BLIP -> Text -> UTF-8 bytes -> Channel -> Bytes -> Text -> SD Text2Image -> Image
    """
    def __init__(self, use_turbo=True, channel_type='awgn', device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.channel = ChannelPipeline(use_turbo=use_turbo, channel_type=channel_type)
        
        print(f"[ItISC] Loading BLIP on {self.device}...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        
        print(f"[ItISC] Loading SD Text2Image on {self.device}...")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.sd_pipe = self.sd_pipe.to(self.device)
        # Disable safety checker to speed up and avoid issues
        self.sd_pipe.safety_checker = None
        
    def process(self, image: Image.Image, snr_db: float):
        original_bits = image.width * image.height * 3 * 8
        
        # 1. Image to Text (BLIP)
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        print(f"[ItISC] Generated caption: {caption}")
        
        # 2. Text to Bytes
        text_bytes = caption.encode('utf-8')
        compressed_bits = len(text_bytes) * 8
        
        if len(text_bytes) == 0:
            text_bytes = b" " # Fallback if empty
            
        q = torch.tensor(list(text_bytes), dtype=torch.uint8)
        
        # 3. Transmit
        metadata = {'shape': q.shape}
        q_tilde = self.channel.transmit(q, snr_db, metadata)
        
        # 4. Bytes to Text
        try:
            received_bytes = bytes(q_tilde.cpu().tolist())
            received_caption = received_bytes.decode('utf-8', errors='ignore')
            print(f"[ItISC] Received caption: {received_caption}")
        except:
            received_caption = ""
            
        # 5. Text to Image (SD)
        if received_caption.strip() == "":
            recon_img = Image.new('RGB', image.size)
        else:
            # Generate image (we do a short generation for speed, e.g. 20 steps)
            recon_img = self.sd_pipe(received_caption, num_inference_steps=20, height=image.height, width=image.width).images[0]
            
        return recon_img, compressed_bits, original_bits
