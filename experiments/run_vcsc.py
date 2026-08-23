import os
import torch
import numpy as np
import time
from preprocessing.image_utils import load_image, preprocess_image, postprocess_image, save_image
from semantic.encoder import SemanticEncoder
from semantic.decoder import SemanticDecoder
from quantization.uniform_quantizer import quantize_uniform_8bit, dequantize_uniform_8bit
from quantization.attention_quantizer import quantize_attention_4bit, dequantize_attention_4bit
from communication.channel_pipeline import ChannelPipeline
import argparse

class VCSCPipeline:
    def __init__(self, use_turbo=True, channel_type='awgn', device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SemanticEncoder(device=self.device)
        self.decoder = SemanticDecoder(device=self.device)
        self.channel = ChannelPipeline(use_turbo=use_turbo, channel_type=channel_type)
        
    def process_image(self, image_input, quant_mode: str, snr_db: float, threshold: float = 0.2, save_dir: str = None):
        if isinstance(image_input, str):
            print(f"--- Processing {image_input} ---")
            img = load_image(image_input)
        else:
            print("--- Processing Uploaded Image ---")
            img = image_input
            
        print(f"Quantization Mode: {quant_mode}, Channel: {self.channel.channel_type}, SNR: {snr_db}dB, Turbo: {self.channel.use_turbo}")
        
        # 1. Preprocess
        img_tensor = preprocess_image(img).to(self.device)
        
        # 2. Encode
        start_enc = time.time()
        z = self.encoder.encode(img_tensor)
        enc_time = time.time() - start_enc
        print(f"Latent shape: {z.shape}, Encoding time: {enc_time:.4f}s")
        if save_dir:
            torch.save(z.cpu(), os.path.join(save_dir, 'latent.pt'))
        
        # 3. Quantize
        if quant_mode == '8bit':
            q, meta = quantize_uniform_8bit(z)
        elif quant_mode == '4bit-attn':
            q, meta = quantize_attention_4bit(z, threshold=threshold)
        else:
            raise ValueError(f"Unknown quantization mode: {quant_mode}")
            
        if save_dir:
            q.cpu().numpy().tofile(os.path.join(save_dir, 'quantized_stream.bin'))
            
        original_bits = 512 * 512 * 3 * 8
        compressed_bits = q.numel() * 8 # Approximation of payload
        compression_rate = (compressed_bits / original_bits) * 100
        print(f"Original bits: {original_bits}, Compressed bits: {compressed_bits}, Rate: {compression_rate:.2f}%")
        
        # 4. Transmit
        start_tx = time.time()
        q_tilde = self.channel.transmit(q, snr_db, meta)
        tx_time = time.time() - start_tx
        print(f"Transmission & Channel time: {tx_time:.4f}s")
        
        # 5. Dequantize
        if quant_mode == '8bit':
            z_tilde = dequantize_uniform_8bit(q_tilde, meta)
        elif quant_mode == '4bit-attn':
            z_tilde = dequantize_attention_4bit(q_tilde, meta, device=q_tilde.device)
            
        z_tilde = z_tilde.to(self.device)
        
        # 6. Decode
        start_dec = time.time()
        img_recon_tensor = self.decoder.decode(z_tilde)
        dec_time = time.time() - start_dec
        print(f"Decoding time: {dec_time:.4f}s")
        
        # 7. Postprocess
        img_recon = postprocess_image(img_recon_tensor)
        
        return img_recon, {
            'enc_time': enc_time,
            'tx_time': tx_time,
            'dec_time': dec_time,
            'compression_rate': compression_rate,
            'compressed_bits': compressed_bits
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input image path")
    parser.add_argument('--output', type=str, default="results/reconstructed.png", help="Output image path")
    parser.add_argument('--quantization', type=str, choices=['8bit', '4bit-attn'], default='8bit')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--channel', type=str, choices=['awgn', 'rayleigh'], default='awgn')
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--no-turbo', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    pipeline = VCSCPipeline(use_turbo=not args.no_turbo, channel_type=args.channel)
    recon_img, stats = pipeline.process_image(args.input, args.quantization, args.snr, args.threshold)
    
    save_image(recon_img, args.output)
    print(f"Saved reconstructed image to {args.output}")
