import torch

def quantize_uniform_8bit(z: torch.Tensor):
    """
    Uniform 8-bit quantization as described in Section IV.B.2.a.
    q = clip(round((z - z_min) * s), 0, 255)
    s = 255 / (z_max - z_min)
    
    Returns:
    - q: Quantized tensor (uint8)
    - metadata: dict containing 'z_min', 'z_max', 's'
    """
    z_min = z.min().item()
    z_max = z.max().item()
    
    # Handle the case where z_max == z_min to avoid division by zero
    if z_max == z_min:
        s = 0.0
        q = torch.zeros_like(z, dtype=torch.uint8)
    else:
        s = 255.0 / (z_max - z_min)
        q = torch.round((z - z_min) * s)
        q = torch.clamp(q, 0, 255).to(torch.uint8)
        
    metadata = {
        'z_min': z_min,
        'z_max': z_max,
        's': s,
        'shape': list(z.shape),
        'mode': '8bit'
    }
    
    return q, metadata

def dequantize_uniform_8bit(q: torch.Tensor, metadata: dict) -> torch.Tensor:
    """
    Uniform 8-bit dequantization as described in Section IV.C.1.a.
    z_tilde = q_tilde / s + z_min
    """
    z_min = metadata['z_min']
    s = metadata['s']
    
    if s == 0.0:
        return torch.full(metadata['shape'], z_min, dtype=torch.float32, device=q.device)
        
    q_float = q.to(torch.float32)
    z_tilde = (q_float / s) + z_min
    
    return z_tilde
