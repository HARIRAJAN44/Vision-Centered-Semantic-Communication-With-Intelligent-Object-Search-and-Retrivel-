import torch
import numpy as np
from communication.channel_pipeline import ChannelPipeline

def test_awgn_pipeline():
    print("Testing AWGN Pipeline...")
    pipeline = ChannelPipeline(use_turbo=False, channel_type='awgn')
    
    # 64x64 random uint8 tensor
    q = torch.randint(0, 256, (1, 4, 64, 64), dtype=torch.uint8)
    metadata = {'shape': q.shape}
    
    # High SNR -> low error
    q_tilde = pipeline.transmit(q, snr_db=30, metadata=metadata)
    assert q.shape == q_tilde.shape
    
    error_rate = (q != q_tilde).float().mean().item()
    print(f"AWGN 30dB bit error rate: {error_rate:.4f}")
    assert error_rate < 0.01

def test_rayleigh_pipeline():
    print("Testing Rayleigh Pipeline...")
    pipeline = ChannelPipeline(use_turbo=False, channel_type='rayleigh')
    
    q = torch.randint(0, 256, (1, 4, 64, 64), dtype=torch.uint8)
    metadata = {'shape': q.shape}
    
    # High SNR 
    q_tilde = pipeline.transmit(q, snr_db=40, metadata=metadata)
    assert q.shape == q_tilde.shape
    
    error_rate = (q != q_tilde).float().mean().item()
    print(f"Rayleigh 40dB bit error rate: {error_rate:.4f}")

if __name__ == "__main__":
    test_awgn_pipeline()
    test_rayleigh_pipeline()
    print("Channel tests passed.")
