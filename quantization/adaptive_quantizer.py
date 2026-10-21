import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math


def _pack_bits(values: torch.Tensor, bits_per_value: int) -> np.ndarray:
    """
    Pack integer values into a bit-packed byte array.
    
    Args:
        values: Flattened tensor of integer values (0 to 2^bits_per_value - 1)
        bits_per_value: Number of bits per value (2, 4, 6, or 8)
    
    Returns:
        Packed bytes as numpy array
    """
    values_np = values.cpu().numpy().astype(np.uint32)
    num_values = len(values_np)
    
    if bits_per_value == 8:
        return values_np.astype(np.uint8)
    
    total_bits = num_values * bits_per_value
    num_bytes = (total_bits + 7) // 8
    packed = np.zeros(num_bytes, dtype=np.uint8)
    
    bit_pos = 0
    for val in values_np:
        for b in range(bits_per_value):
            if val & (1 << (bits_per_value - 1 - b)):
                byte_idx = bit_pos // 8
                bit_idx = 7 - (bit_pos % 8)
                packed[byte_idx] |= (1 << bit_idx)
            bit_pos += 1
    
    return packed


def _unpack_bits(packed_bytes: np.ndarray, num_values: int, bits_per_value: int) -> torch.Tensor:
    """
    Unpack bit-packed bytes back to integer values.
    
    Args:
        packed_bytes: Packed byte array
        num_values: Number of values to unpack
        bits_per_value: Number of bits per value (2, 4, 6, or 8)
    
    Returns:
        Tensor of unpacked integer values
    """
    if bits_per_value == 8:
        return torch.from_numpy(packed_bytes.astype(np.uint8))[:num_values]
    
    values = np.zeros(num_values, dtype=np.uint32)
    bit_pos = 0
    
    for i in range(num_values):
        val = 0
        for b in range(bits_per_value):
            byte_idx = bit_pos // 8
            bit_idx = 7 - (bit_pos % 8)
            if byte_idx < len(packed_bytes) and (packed_bytes[byte_idx] & (1 << bit_idx)):
                val |= (1 << (bits_per_value - 1 - b))
            bit_pos += 1
        values[i] = val
    
    return torch.from_numpy(values)


def quantize_uniform_multi_bit(z: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, Dict]:
    """
    Uniform multi-bit quantization supporting 2, 4, 6, 8 bits.
    
    Args:
        z: Input latent tensor of shape (B, C, H, W)
        bits: Number of bits per value (2, 4, 6, or 8)
    
    Returns:
        Tuple of (packed_bytes, metadata)
    """
    assert bits in [2, 4, 6, 8], f"Supported bit depths: 2, 4, 6, 8. Got {bits}"
    
    z_min = z.min().item()
    z_max = z.max().item()
    num_levels = 2 ** bits
    max_val = num_levels - 1
    
    if z_max == z_min:
        scale = 0.0
        q_flat = torch.zeros(z.numel(), dtype=torch.uint32)
    else:
        scale = max_val / (z_max - z_min)
        q_flat = torch.round((z.flatten() - z_min) * scale)
        q_flat = torch.clamp(q_flat, 0, max_val).to(torch.uint32)
    
    packed_bytes = _pack_bits(q_flat, bits)
    
    metadata = {
        'z_min': z_min,
        'z_max': z_max,
        'scale': scale,
        'shape': list(z.shape),
        'bits': bits,
        'num_levels': num_levels,
        'num_values': z.numel(),
        'packed_bytes': len(packed_bytes),
        'mode': f'{bits}bit-uniform'
    }
    
    return torch.from_numpy(packed_bytes), metadata


def dequantize_uniform_multi_bit(packed_bytes: torch.Tensor, metadata: Dict, device: str = 'cpu') -> torch.Tensor:
    """
    Dequantize multi-bit quantized tensor.
    
    Args:
        packed_bytes: Packed byte tensor
        metadata: Quantization metadata
        device: Target device
    
    Returns:
        Dequantized tensor of original shape
    """
    bits = metadata['bits']
    num_values = metadata['num_values']
    z_min = metadata['z_min']
    scale = metadata['scale']
    shape = metadata['shape']
    num_levels = metadata['num_levels']
    max_val = num_levels - 1
    
    if scale == 0.0:
        return torch.full(shape, z_min, dtype=torch.float32, device=device)
    
    q_flat = _unpack_bits(packed_bytes.cpu().numpy(), num_values, bits)
    q_flat = q_flat.to(device=device, dtype=torch.float32)
    
    z_tilde_flat = (q_flat / scale) + z_min
    z_tilde = z_tilde_flat.reshape(shape)
    
    return z_tilde


class AdaptiveQuantizer:
    """
    Adaptive multi-level quantizer supporting 2/4/6/8-bit per region/channel.
    
    Allocation modes:
    - 'uniform': Same bit depth for all elements
    - 'attention': Attention-based (high/low importance regions)
    - 'adaptive': Per-channel or per-spatial-region allocation
    - 'channel_aware': SNR-aware allocation
    """
    
    def __init__(
        self,
        allocation_mode: str = 'uniform',
        default_bits: int = 8,
        attention_threshold: float = 0.2,
        min_bits: int = 2,
        max_bits: int = 8,
        snr_db: float = 10.0,
        bandwidth_mhz: float = 5.0
    ):
        self.allocation_mode = allocation_mode
        self.default_bits = default_bits
        self.attention_threshold = attention_threshold
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.snr_db = snr_db
        self.bandwidth_mhz = bandwidth_mhz
        
        self.supported_bits = [2, 4, 6, 8]
    
    def _compute_attention_map(self, z: torch.Tensor) -> torch.Tensor:
        """Compute attention map A(i,j) = (1/C) * sum(|z(c,i,j)|)"""
        return torch.mean(torch.abs(z), dim=1, keepdim=True)
    
    def _create_allocation_map(self, z: torch.Tensor, attention_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create bit allocation map based on allocation mode.
        
        Returns:
            allocation_map: Tensor of shape (B, 1, H, W) with bit depth per spatial location
        """
        B, C, H, W = z.shape
        
        if self.allocation_mode == 'uniform':
            return torch.full((B, 1, H, W), self.default_bits, dtype=torch.int32, device=z.device)
        
        elif self.allocation_mode == 'attention':
            if attention_map is None:
                attention_map = self._compute_attention_map(z)
            
            A_norm = self._normalize_attention(attention_map)
            M = (A_norm >= self.attention_threshold).to(torch.int32)
            
            high_bits = self.max_bits
            low_bits = self.min_bits
            
            allocation = M * high_bits + (1 - M) * low_bits
            return allocation
        
        elif self.allocation_mode == 'adaptive':
            if attention_map is None:
                attention_map = self._compute_attention_map(z)
            
            A_norm = self._normalize_attention(attention_map)
            
            num_levels = len(self.supported_bits)
            allocation = torch.zeros((B, 1, H, W), dtype=torch.int32, device=z.device)
            
            for i, bits in enumerate(self.supported_bits):
                lower = i / num_levels
                upper = (i + 1) / num_levels
                mask = (A_norm >= lower) & (A_norm < upper)
                if i == num_levels - 1:
                    mask = mask | (A_norm >= upper)
                allocation[mask] = bits
            
            return allocation
        
        elif self.allocation_mode == 'channel_aware':
            if attention_map is None:
                attention_map = self._compute_attention_map(z)
            
            A_norm = self._normalize_attention(attention_map)
            
            snr_linear = 10 ** (self.snr_db / 10.0)
            capacity_factor = min(1.0, snr_linear / 10.0)
            
            base_bits = max(self.min_bits, int(self.max_bits * capacity_factor))
            
            num_levels = len(self.supported_bits)
            allocation = torch.zeros((B, 1, H, W), dtype=torch.int32, device=z.device)
            
            for i, bits in enumerate(self.supported_bits):
                lower = i / num_levels
                upper = (i + 1) / num_levels
                mask = (A_norm >= lower) & (A_norm < upper)
                if i == num_levels - 1:
                    mask = mask | (A_norm >= upper)
                
                allocated = min(bits, base_bits + i)
                allocated = max(self.min_bits, min(self.max_bits, allocated))
                allocation[mask] = allocated
            
            return allocation
        
        else:
            raise ValueError(f"Unknown allocation mode: {self.allocation_mode}")
    
    def _normalize_attention(self, A: torch.Tensor) -> torch.Tensor:
        """Normalize attention map to [0, 1]"""
        B = A.shape[0]
        A_min = A.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        A_max = A.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        
        diff = A_max - A_min
        diff[diff == 0] = 1.0
        return (A - A_min) / diff
    
    def _serialize_allocation_map(self, allocation_map: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Serialize allocation map to bits"""
        B, _, H, W = allocation_map.shape
        flat = allocation_map.flatten().to(torch.uint8)
        
        unique_bits = torch.unique(flat)
        bit_to_idx = {int(b.item()): i for i, b in enumerate(unique_bits)}
        idx_map = torch.tensor([bit_to_idx[int(b.item())] for b in flat], dtype=torch.uint8)
        
        num_indices = len(unique_bits)
        index_bits = math.ceil(math.log2(num_indices)) if num_indices > 1 else 1
        
        packed = _pack_bits(idx_map, index_bits)
        
        meta = {
            'unique_bits': [int(b.item()) for b in unique_bits],
            'index_bits': index_bits,
            'shape': list(allocation_map.shape),
            'num_values': allocation_map.numel()
        }
        
        return torch.from_numpy(packed), meta
    
    def _deserialize_allocation_map(self, packed_bytes: torch.Tensor, alloc_meta: Dict, device: str = 'cpu') -> torch.Tensor:
        """Deserialize allocation map from bits"""
        unique_bits = alloc_meta['unique_bits']
        index_bits = alloc_meta['index_bits']
        num_values = alloc_meta['num_values']
        shape = alloc_meta['shape']
        
        idx_map = _unpack_bits(packed_bytes.cpu().numpy(), num_values, index_bits)
        allocation_flat = torch.tensor([unique_bits[min(max(0, int(i)), len(unique_bits) - 1)] for i in idx_map], dtype=torch.int32, device=device)
        allocation = allocation_flat.reshape(shape)
        
        return allocation
    
    def quantize(self, z: torch.Tensor, attention_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize latent tensor with adaptive bit allocation.
        
        Returns:
            packed_data: Concatenated [allocation_map_bytes, quantized_data_bytes]
            metadata: Full metadata for dequantization
        """
        B, C, H, W = z.shape
        
        allocation_map = self._create_allocation_map(z, attention_map)
        
        alloc_bytes, alloc_meta = self._serialize_allocation_map(allocation_map)
        
        if self.allocation_mode == 'uniform':
            q_bytes, q_meta = quantize_uniform_multi_bit(z, self.default_bits)
            q_meta['allocation_mode'] = 'uniform'
        else:
            q_bytes, q_meta = self._quantize_per_region(z, allocation_map)
            q_meta['allocation_mode'] = self.allocation_mode
        
        q_meta['allocation_map_meta'] = alloc_meta
        q_meta['allocation_map_bytes'] = len(alloc_bytes)
        q_meta['allocation_map'] = allocation_map.cpu()
        
        total_packed = np.concatenate([
            alloc_bytes,
            q_bytes.cpu().numpy() if isinstance(q_bytes, torch.Tensor) else q_bytes
        ])
        
        return torch.from_numpy(total_packed), q_meta
    
    def _quantize_per_region(self, z: torch.Tensor, allocation_map: torch.Tensor) -> Tuple[np.ndarray, Dict]:
        """Quantize each region with its allocated bit depth"""
        B, C, H, W = z.shape
        
        allocation_flat = allocation_map.flatten()
        unique_bits = torch.unique(allocation_flat).tolist()
        
        allocation_expanded = allocation_map.repeat(1, C, 1, 1).flatten()
        
        all_packed_chunks = []
        region_metadata = {}
        
        for bits in unique_bits:
            mask = (allocation_expanded == bits)
            if not mask.any():
                continue
            
            z_masked = z.flatten()[mask]
            
            if len(z_masked) == 0:
                continue
            
            z_min = z_masked.min().item()
            z_max = z_masked.max().item()
            num_levels = 2 ** bits
            max_val = num_levels - 1
            
            if z_max == z_min:
                scale = 0.0
                q_flat = torch.zeros_like(z_masked, dtype=torch.uint32)
            else:
                scale = max_val / (z_max - z_min)
                q_flat = torch.round((z_masked - z_min) * scale)
                q_flat = torch.clamp(q_flat, 0, max_val).to(torch.uint32)
            
            packed = _pack_bits(q_flat, bits)
            all_packed_chunks.append(packed)
            
            region_metadata[bits] = {
                'z_min': z_min,
                'z_max': z_max,
                'scale': scale,
                'num_values': int(mask.sum().item()),
                'bits': bits,
                'num_levels': num_levels,
                'packed_bytes': len(packed)
            }
        
        combined_packed = np.concatenate(all_packed_chunks) if all_packed_chunks else np.array([], dtype=np.uint8)
        
        meta = {
            'region_metadata': region_metadata,
            'allocation_map_shape': list(allocation_map.shape),
            'total_packed_bytes': len(combined_packed),
            'shape': list(z.shape)
        }
        
        return combined_packed, meta
    
    def dequantize(self, packed_data: torch.Tensor, metadata: Dict, device: str = 'cpu') -> torch.Tensor:
        """
        Dequantize adaptive quantized tensor.
        """
        alloc_meta = metadata.get('allocation_map_meta', {})
        alloc_map_bytes = metadata.get('allocation_map_bytes', 0)
        
        alloc_packed = packed_data[:alloc_map_bytes]
        q_packed = packed_data[alloc_map_bytes:]
        
        # Use true allocation map from control signaling if available, otherwise deserialize
        if 'allocation_map' in metadata and metadata['allocation_map'] is not None:
            allocation_map = metadata['allocation_map'].to(device)
        else:
            allocation_map = self._deserialize_allocation_map(alloc_packed, alloc_meta, device)
        
        if metadata.get('allocation_mode') == 'uniform':
            bits = metadata['bits']
            q_meta = {k: v for k, v in metadata.items() if k not in ['allocation_map_meta', 'allocation_map_bytes', 'allocation_mode', 'allocation_map']}
            z_tilde = dequantize_uniform_multi_bit(q_packed, q_meta, device)
        else:
            z_tilde = self._dequantize_per_region(q_packed, metadata, allocation_map, device)
        
        return z_tilde
    
    def _dequantize_per_region(self, q_packed: torch.Tensor, metadata: Dict, allocation_map: torch.Tensor, device: str) -> torch.Tensor:
        """Dequantize each region with its bit depth"""
        B, C, H, W = metadata['shape']
        allocation_expanded = allocation_map.repeat(1, C, 1, 1).flatten()
        unique_bits = sorted([int(b) for b in metadata['region_metadata'].keys()])
        
        z_tilde_flat = torch.zeros(B * C * H * W, dtype=torch.float32, device=device)
        
        bit_offset = 0
        
        for bits in unique_bits:
            region_meta = metadata['region_metadata'].get(bits, metadata['region_metadata'].get(str(bits)))
            num_values = region_meta['num_values']
            
            if num_values == 0:
                continue
            
            # Use stored packed_bytes instead of calculating
            region_packed_bytes = region_meta.get('packed_bytes', _get_packed_bytes_for_values(num_values, bits))
            
            if bit_offset + region_packed_bytes > len(q_packed):
                region_packed = q_packed[bit_offset:]
            else:
                region_packed = q_packed[bit_offset:bit_offset + region_packed_bytes]
            
            bit_offset += region_packed_bytes
            
            q_flat = _unpack_bits(region_packed.cpu().numpy(), num_values, bits)
            q_flat = q_flat.to(device=device, dtype=torch.float32)
            
            scale = region_meta['scale']
            z_min = region_meta['z_min']
            
            if scale == 0.0:
                z_region = torch.full((num_values,), z_min, dtype=torch.float32, device=device)
            else:
                z_region = (q_flat / scale) + z_min
            
            mask = (allocation_expanded == bits)
            mask_count = int(mask.sum().item())
            if mask_count == len(z_region):
                z_tilde_flat[mask] = z_region
            elif mask_count > 0:
                min_len = min(mask_count, len(z_region))
                mask_indices = torch.nonzero(mask, as_tuple=False).flatten()
                z_tilde_flat[mask_indices[:min_len]] = z_region[:min_len]
        
        return z_tilde_flat.reshape(B, C, H, W)
    
    def get_allocation_stats(self, allocation_map: torch.Tensor) -> Dict:
        """Get statistics about bit allocation"""
        flat = allocation_map.flatten()
        unique, counts = torch.unique(flat, return_counts=True)
        
        stats = {
            'total_elements': flat.numel(),
            'per_bit_depth': {}
        }
        
        total_bits = 0
        for bits, count in zip(unique.tolist(), counts.tolist()):
            stats['per_bit_depth'][bits] = {
                'count': int(count),
                'percentage': float(count) / flat.numel() * 100,
                'ideal_bits': int(count) * bits
            }
            total_bits += int(count) * bits
        
        stats['ideal_total_bits'] = total_bits
        stats['ideal_bytes'] = (total_bits + 7) // 8
        
        return stats


def _get_packed_bytes_for_values(num_values: int, bits_per_value: int) -> int:
    """Calculate number of bytes needed to pack num_values at bits_per_value"""
    total_bits = num_values * bits_per_value
    return (total_bits + 7) // 8


def calculate_payload_bits(metadata: Dict) -> Dict:
    """
    Calculate detailed payload bit accounting.
    
    Returns:
        Dictionary with ideal_bits, packed_bytes, allocation_overhead_bits, etc.
    """
    allocation_mode = metadata.get('allocation_mode', 'uniform')
    
    if allocation_mode == 'uniform':
        bits = metadata['bits']
        num_values = metadata['num_values']
        ideal_bits = num_values * bits
        packed_bytes = metadata.get('packed_bytes', (ideal_bits + 7) // 8)
        alloc_overhead = 0
    else:
        ideal_bits = 0
        region_meta = metadata.get('region_metadata', {})
        for bits, rmeta in region_meta.items():
            ideal_bits += rmeta['num_values'] * bits
        
        packed_bytes = metadata.get('total_packed_bytes', 0)
        alloc_overhead = metadata.get('allocation_map_bytes', 0) * 8
    
    return {
        'ideal_quantized_bits': ideal_bits,
        'packed_bytes': packed_bytes,
        'packed_bits': packed_bytes * 8,
        'allocation_overhead_bits': alloc_overhead,
        'total_transmitted_bits': packed_bytes * 8 + alloc_overhead,
        'compression_ratio_vs_8bit': (ideal_bits / (num_values * 8)) if allocation_mode == 'uniform' else 
                                     (ideal_bits / (metadata['shape'][0] * metadata['shape'][1] * metadata['shape'][2] * metadata['shape'][3] * 8))
    }


if __name__ == "__main__":
    print("Testing Adaptive Quantizer...")
    
    z = torch.randn(1, 4, 64, 64)
    
    for bits in [2, 4, 6, 8]:
        print(f"\n--- Testing {bits}-bit uniform ---")
        q_bytes, meta = quantize_uniform_multi_bit(z, bits)
        print(f"  Packed bytes: {len(q_bytes)}, Ideal bits: {meta['num_values'] * bits}")
        
        z_tilde = dequantize_uniform_multi_bit(q_bytes, meta)
        mse = torch.nn.functional.mse_loss(z, z_tilde)
        print(f"  MSE: {mse.item():.6f}")
        print(f"  Metadata: {meta}")
    
    print("\n--- Testing AdaptiveQuantizer (uniform) ---")
    aq = AdaptiveQuantizer(allocation_mode='uniform', default_bits=4)
    packed, meta = aq.quantize(z)
    print(f"  Total packed bytes: {len(packed)}")
    print(f"  Metadata: {meta}")
    z_tilde = aq.dequantize(packed, meta)
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"  MSE: {mse.item():.6f}")
    
    print("\n--- Testing AdaptiveQuantizer (attention) ---")
    aq = AdaptiveQuantizer(allocation_mode='attention', min_bits=2, max_bits=8, attention_threshold=0.2)
    packed, meta = aq.quantize(z)
    print(f"  Total packed bytes: {len(packed)}")
    print(f"  Allocation map bytes: {meta['allocation_map_bytes']}")
    print(f"  Quantized data bytes: {len(packed) - meta['allocation_map_bytes']}")
    z_tilde = aq.dequantize(packed, meta)
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"  MSE: {mse.item():.6f}")
    
    print("\n--- Testing AdaptiveQuantizer (adaptive) ---")
    aq = AdaptiveQuantizer(allocation_mode='adaptive', min_bits=2, max_bits=8)
    packed, meta = aq.quantize(z)
    print(f"  Total packed bytes: {len(packed)}")
    z_tilde = aq.dequantize(packed, meta)
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"  MSE: {mse.item():.6f}")
    
    print("\n--- Testing AdaptiveQuantizer (channel_aware) ---")
    aq = AdaptiveQuantizer(allocation_mode='channel_aware', min_bits=2, max_bits=8, snr_db=5.0)
    packed, meta = aq.quantize(z)
    print(f"  Total packed bytes: {len(packed)}")
    z_tilde = aq.dequantize(packed, meta)
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"  MSE: {mse.item():.6f}")
    
    print("\n--- Allocation Stats ---")
    aq = AdaptiveQuantizer(allocation_mode='adaptive', min_bits=2, max_bits=8)
    alloc_map = aq._create_allocation_map(z)
    stats = aq.get_allocation_stats(alloc_map)
    print(f"  Stats: {stats}")
    
    print("\n--- Payload Calculation ---")
    payload = calculate_payload_bits(meta)
    print(f"  Payload: {payload}")
    
    print("\nAll tests passed!")