import sys
import os
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baselines.jpeg import JPEGBaseline
from baselines.huffman import HuffmanBaseline
from metrics.evaluator import Evaluator
import traceback

def test():
    try:
        jpeg_pipeline = JPEGBaseline(quality=80, use_turbo=True, channel_type='awgn')
        huffman_pipeline = HuffmanBaseline(use_turbo=True, channel_type='awgn')
        evaluator = Evaluator(device='cpu')

        # Create dummy image 512x512
        img = Image.new('RGB', (512, 512), color='red')
        
        print("Testing JPEG...")
        recon, cb, ob = jpeg_pipeline.process(img, 10.0)
        print("JPEG recon size:", recon.size)
        m = evaluator.evaluate(img, recon)
        print("JPEG metrics:", m)
        
        print("Testing Huffman...")
        recon2, cb2, ob2 = huffman_pipeline.process(img, 10.0)
        print("Huffman recon size:", recon2.size)
        m2 = evaluator.evaluate(img, recon2)
        print("Huffman metrics:", m2)
        
        print("Done!")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test()
