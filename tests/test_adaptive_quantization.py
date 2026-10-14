import torch
from quantization.adaptive_quantizer import (
    quantize_uniform_multi_bit,
    dequantize_uniform_multi_bit,
    AdaptiveQuantizer,
    calculate_payload_bits
)

def test_uniform_multi_bit():
    print("Testing Uniform Multi-bit Quantization...")
    z = torch.randn(1, 4, 64, 64)
    
    for bits in [2, 4, 6, 8]:
        q_bytes, meta = quantize_uniform_multi_bit(z, bits)
        assert meta['bits'] == bits
        assert meta['num_values'] == z.numel()
        assert len(q_bytes) == meta['packed_bytes']
        
        z_tilde = dequantize_uniform_multi_bit(q_bytes, meta)
        assert z_tilde.shape == z.shape
        
        mse = torch.nn.functional.mse_loss(z, z_tilde)
        print(f"  {bits}-bit: MSE={mse.item():.6f}, packed_bytes={len(q_bytes)}")
    
    print("Uniform multi-bit tests passed!")

def test_adaptive_quantizer_uniform():
    print("Testing AdaptiveQuantizer (uniform)...")
    z = torch.randn(1, 4, 64, 64)
    
    aq = AdaptiveQuantizer(allocation_mode='uniform', default_bits=4)
    packed, meta = aq.quantize(z)
    assert meta['allocation_mode'] == 'uniform'
    assert meta['bits'] == 4
    
    z_tilde = aq.dequantize(packed, meta)
    assert z_tilde.shape == z.shape
    
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"  MSE: {mse.item():.6f}")
    print("AdaptiveQuantizer uniform test passed!")

def test_adaptive_quantizer_attention():
    print("Testing AdaptiveQuantizer (attention)...")
    z = torch.randn(1, 4, 64, 64)
    
    aq = AdaptiveQuantizer(allocation_mode='attention', min_bits=2, max_bits=8, attention_threshold=0.2)
    packed, meta = aq.quantize(z)
    assert meta['allocation_mode'] == 'attention'
    
    z_tilde = aq.dequantize(packed, meta)
    assert z_tilde.shape == z.shape
    
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"  MSE: {mse.item():.6f}")
    print("AdaptiveQuantizer attention test passed!")

def test_adaptive_quantizer_adaptive():
    print("Testing AdaptiveQuantizer (adaptive)...")
    z = torch.randn(1, 4, 64, 64)
    
    aq = AdaptiveQuantizer(allocation_mode='adaptive', min_bits=2, max_bits=8)
    packed, meta = aq.quantize(z)
    assert meta['allocation_mode'] == 'adaptive'
    
    z_tilde = aq.dequantize(packed, meta)
    assert z_tilde.shape == z.shape
    
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"  MSE: {mse.item():.6f}")
    print("AdaptiveQuantizer adaptive test passed!")

def test_adaptive_quantizer_channel_aware():
    print("Testing AdaptiveQuantizer (channel_aware)...")
    z = torch.randn(1, 4, 64, 64)
    
    aq = AdaptiveQuantizer(allocation_mode='channel_aware', min_bits=2, max_bits=8, snr_db=5.0)
    packed, meta = aq.quantize(z)
    assert meta['allocation_mode'] == 'channel_aware'
    
    z_tilde = aq.dequantize(packed, meta)
    assert z_tilde.shape == z.shape
    
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"  MSE: {mse.item():.6f}")
    print("AdaptiveQuantizer channel_aware test passed!")

def test_allocation_stats():
    print("Testing allocation stats...")
    z = torch.randn(1, 4, 64, 64)
    
    aq = AdaptiveQuantizer(allocation_mode='adaptive', min_bits=2, max_bits=8)
    alloc_map = aq._create_allocation_map(z)
    stats = aq.get_allocation_stats(alloc_map)
    
    assert stats['total_elements'] == alloc_map.numel()
    assert 'per_bit_depth' in stats
    assert 'ideal_total_bits' in stats
    assert stats['ideal_total_bits'] > 0
    print(f"  Stats: {stats}")
    print("Allocation stats test passed!")

def test_payload_calculation():
    print("Testing payload calculation...")
    z = torch.randn(1, 4, 64, 64)
    
    aq = AdaptiveQuantizer(allocation_mode='uniform', default_bits=4)
    packed, meta = aq.quantize(z)
    payload = calculate_payload_bits(meta)
    
    assert 'ideal_quantized_bits' in payload
    assert 'packed_bytes' in payload
    assert 'total_transmitted_bits' in payload
    print(f"  Payload: {payload}")
    print("Payload calculation test passed!")

def test_round_trip_consistency():
    print("Testing round-trip consistency...")
    z = torch.randn(1, 4, 64, 64)
    
    for mode in ['uniform', 'attention', 'adaptive', 'channel_aware']:
        if mode == 'uniform':
            aq = AdaptiveQuantizer(allocation_mode=mode, default_bits=4)
        elif mode == 'attention':
            aq = AdaptiveQuantizer(allocation_mode=mode, min_bits=2, max_bits=8)
        elif mode == 'adaptive':
            aq = AdaptiveQuantizer(allocation_mode=mode, min_bits=2, max_bits=8)
        else:
            aq = AdaptiveQuantizer(allocation_mode=mode, min_bits=2, max_bits=8, snr_db=10.0)
        
        packed, meta = aq.quantize(z)
        z_tilde = aq.dequantize(packed, meta)
        
        mse = torch.nn.functional.mse_loss(z, z_tilde)
        print(f"  {mode}: MSE={mse.item():.6f}")
        assert z_tilde.shape == z.shape
    
    print("Round-trip consistency test passed!")

if __name__ == "__main__":
    test_uniform_multi_bit()
    test_adaptive_quantizer_uniform()
    test_adaptive_quantizer_attention()
    test_adaptive_quantizer_adaptive()
    test_adaptive_quantizer_channel_aware()
    test_allocation_stats()
    test_payload_calculation()
    test_round_trip_consistency()
    print("\nAll quantization tests passed!")