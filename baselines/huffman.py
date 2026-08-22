import numpy as np
from PIL import Image
import torch
import zlib # Using zlib as a fast proxy for Huffman/lossless compression
from communication.channel_pipeline import ChannelPipeline

class HuffmanBaseline:
    def __init__(self, use_turbo=True, channel_type='awgn'):
        self.channel = ChannelPipeline(use_turbo=use_turbo, channel_type=channel_type)
        
    def process(self, image: Image.Image, snr_db: float):
        # We use zlib compress as a proxy for standard Huffman/Deflate
        img_bytes = image.tobytes()
        compressed_bytes = zlib.compress(img_bytes, level=9)
        
        q = torch.tensor(list(compressed_bytes), dtype=torch.uint8)
        
        original_bits = image.width * image.height * 3 * 8
        compressed_bits = len(compressed_bytes) * 8
        
        metadata = {'shape': q.shape}
        q_tilde = self.channel.transmit(q, snr_db, metadata)
        
        try:
            received_bytes = bytes(q_tilde.cpu().tolist())
            decompressed_bytes = zlib.decompress(received_bytes)
            recon_img = Image.frombytes(image.mode, image.size, decompressed_bytes)
        except Exception as e:
            # If lossless stream is corrupted, it fails to decompress
            recon_img = Image.new('RGB', image.size)
            
        return recon_img, compressed_bits, original_bits
