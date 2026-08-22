import torch
from quantization.uniform_quantizer import quantize_uniform_8bit, dequantize_uniform_8bit
from quantization.attention_quantizer import quantize_attention_4bit, dequantize_attention_4bit

def test_uniform_8bit():
    print("Testing Uniform 8-bit quantization...")
    z = torch.randn(1, 4, 64, 64)
    q, meta = quantize_uniform_8bit(z)
    assert q.dtype == torch.uint8
    assert q.shape == z.shape
    assert meta['mode'] == '8bit'
    
    z_tilde = dequantize_uniform_8bit(q, meta)
    assert z_tilde.shape == z.shape
    assert z_tilde.dtype == torch.float32
    
    # Calculate MSE
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"Uniform 8-bit MSE: {mse.item():.4f}")
    assert mse < 0.1, "Quantization error too high"

def test_attention_4bit():
    print("Testing Attention 4-bit quantization...")
    z = torch.randn(1, 4, 64, 64)
    q, meta = quantize_attention_4bit(z, threshold=0.2)
    assert q.dtype == torch.uint8
    assert q.shape == z.shape
    assert meta['mode'] == '4bit-attn'
    
    z_tilde = dequantize_attention_4bit(q, meta, device='cpu')
    assert z_tilde.shape == z.shape
    assert z_tilde.dtype == torch.float32
    
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"Attention 4-bit (T=0.2) MSE: {mse.item():.4f}")
    
    # Test T=0.4
    q2, meta2 = quantize_attention_4bit(z, threshold=0.4)
    z_tilde2 = dequantize_attention_4bit(q2, meta2, device='cpu')
    mse2 = torch.nn.functional.mse_loss(z, z_tilde2)
    print(f"Attention 4-bit (T=0.4) MSE: {mse2.item():.4f}")

if __name__ == "__main__":
    test_uniform_8bit()
    test_attention_4bit()
    print("All quantization tests passed.")
