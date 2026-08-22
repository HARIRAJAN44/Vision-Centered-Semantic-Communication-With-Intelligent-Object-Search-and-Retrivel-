import io
import numpy as np
from PIL import Image
from communication.channel_pipeline import ChannelPipeline
import torch

class JPEGBaseline:
    def __init__(self, quality=80, use_turbo=True, channel_type='awgn'):
        self.quality = quality
        self.channel = ChannelPipeline(use_turbo=use_turbo, channel_type=channel_type)
        
    def process(self, image: Image.Image, snr_db: float):
        # 1. Compress to JPEG bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=self.quality)
        jpeg_bytes = img_byte_arr.getvalue()
        
        # 2. Serialize bytes to uint8 tensor to reuse our channel
        q = torch.tensor(list(jpeg_bytes), dtype=torch.uint8)
        
        original_bits = image.width * image.height * 3 * 8
        compressed_bits = len(jpeg_bytes) * 8
        
        # 3. Transmit
        metadata = {'shape': q.shape}
        q_tilde = self.channel.transmit(q, snr_db, metadata)
        
        # 4. Deserialize and decode JPEG
        try:
            received_bytes = bytes(q_tilde.cpu().tolist())
            recon_img = Image.open(io.BytesIO(received_bytes))
            recon_img = recon_img.convert("RGB")
        except Exception as e:
            # If JPEG header is corrupted, we might not be able to open it.
            # In a real simulation, corrupted JPEG usually means complete failure or black image.
            # We return a black image of the same size.
            recon_img = Image.new('RGB', image.size)
            
        return recon_img, compressed_bits, original_bits
