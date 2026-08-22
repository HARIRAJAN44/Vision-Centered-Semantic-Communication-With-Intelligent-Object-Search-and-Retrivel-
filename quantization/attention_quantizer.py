import torch

def quantize_attention_4bit(z: torch.Tensor, threshold: float = 0.2):
    """
    Attention-based 4-bit quantization as described in Section IV.B.2.b.
    """
    B, C, H, W = z.shape
    
    # 1. Compute attention map A (Eq. 6)
    # A(i,j) = (1/C) * sum(|z(c,i,j)|)
    A = torch.mean(torch.abs(z), dim=1, keepdim=True) # Shape: (B, 1, H, W)
    
    # 2. Normalize attention map Anorm (Eq. 7)
    # Anorm(i,j) = (A(i,j) - min(A)) / (max(A) - min(A))
    A_min = A.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
    A_max = A.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    
    # Avoid division by zero
    diff = A_max - A_min
    diff[diff == 0] = 1.0
    A_norm = (A - A_min) / diff
    
    # 3. Create binary mask M (Eq. 8)
    M = (A_norm >= threshold).to(torch.float32)
    
    # Extract values for high and low importance regions
    # Since we operate on the whole tensor and apply mask later, we compute mins/maxs globally
    # but based ONLY on the masked regions.
    
    # To get min/max of masked regions, we can use large/small values for the opposite mask
    z_high_masked = torch.where(M == 1, z, torch.tensor(float('inf'), device=z.device))
    z_high_min = z_high_masked.view(B, C, -1).min(dim=2)[0].view(B, C, 1, 1)
    
    z_high_masked_max = torch.where(M == 1, z, torch.tensor(float('-inf'), device=z.device))
    z_high_max = z_high_masked_max.view(B, C, -1).max(dim=2)[0].view(B, C, 1, 1)

    z_low_masked = torch.where(M == 0, z, torch.tensor(float('inf'), device=z.device))
    z_low_min = z_low_masked.view(B, C, -1).min(dim=2)[0].view(B, C, 1, 1)
    
    z_low_masked_max = torch.where(M == 0, z, torch.tensor(float('-inf'), device=z.device))
    z_low_max = z_low_masked_max.view(B, C, -1).max(dim=2)[0].view(B, C, 1, 1)
    
    # Handle edge cases where mask is all 1s or all 0s
    z_high_min[torch.isinf(z_high_min)] = 0.0
    z_high_max[torch.isinf(z_high_max)] = 1.0
    z_low_min[torch.isinf(z_low_min)] = 0.0
    z_low_max[torch.isinf(z_low_max)] = 1.0

    # 4. High-precision (8-bit) quantization for important regions (Eq. 9, 10)
    s_high = 255.0 / torch.clamp(z_high_max - z_high_min, min=1e-8)
    q_high = torch.round((z - z_high_min) * s_high)
    q_high = torch.clamp(q_high, 0, 255)
    
    # 5. Low-precision (4-bit) quantization for less important regions (Eq. 11, 12)
    s_low = 15.0 / torch.clamp(z_low_max - z_low_min, min=1e-8)
    q_low = torch.round((z - z_low_min) * s_low)
    q_low = torch.clamp(q_low, 0, 15)
    
    # 6. Merge according to mask (Eq. 13)
    q_com = M * q_high + (1 - M) * q_low
    q_com = q_com.to(torch.uint8) # Can fit in uint8 since max is 255
    
    metadata = {
        'z_high_min': z_high_min.cpu().tolist(),
        's_high': s_high.cpu().tolist(),
        'z_low_min': z_low_min.cpu().tolist(),
        's_low': s_low.cpu().tolist(),
        'mask': M.cpu().to(torch.uint8).tolist(), # Store mask as list for serialization
        'shape': list(z.shape),
        'mode': '4bit-attn',
        'threshold': threshold
    }
    
    return q_com, metadata

def dequantize_attention_4bit(q_com: torch.Tensor, metadata: dict, device: str = 'cpu') -> torch.Tensor:
    """
    Attention-based 4-bit dequantization as described in Section IV.C.1.b.
    """
    M = torch.tensor(metadata['mask'], dtype=torch.float32, device=device)
    z_high_min = torch.tensor(metadata['z_high_min'], dtype=torch.float32, device=device)
    s_high = torch.tensor(metadata['s_high'], dtype=torch.float32, device=device)
    z_low_min = torch.tensor(metadata['z_low_min'], dtype=torch.float32, device=device)
    s_low = torch.tensor(metadata['s_low'], dtype=torch.float32, device=device)
    
    q_com_float = q_com.to(torch.float32)
    
    # Important regions (M=1)
    z_high_tilde = (q_com_float / s_high) + z_high_min
    
    # Low importance regions (M=0)
    z_low_tilde = (q_com_float / s_low) + z_low_min
    
    # Merge
    z_tilde = M * z_high_tilde + (1 - M) * z_low_tilde
    
    return z_tilde
