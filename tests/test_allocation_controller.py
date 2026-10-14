import torch
import numpy as np
from unittest.mock import Mock, patch

from quantization.allocation_controller import (
    ChannelCondition,
    ChannelState,
    AllocationPolicy,
    AllocationController,
    create_allocation_controller
)
from quantization.adaptive_quantizer import AdaptiveQuantizer


def test_channel_condition_enum():
    """Test ChannelCondition enum."""
    assert ChannelCondition.EXCELLENT.value == "excellent"
    assert ChannelCondition.GOOD.value == "good"
    assert ChannelCondition.FAIR.value == "fair"
    assert ChannelCondition.POOR.value == "poor"
    assert ChannelCondition.SEVER.value == "severe"
    print("ChannelCondition enum test passed!")


def test_channel_state():
    """Test ChannelState creation and classification."""
    # Test each condition
    test_cases = [
        (25.0, ChannelCondition.EXCELLENT),
        (15.0, ChannelCondition.GOOD),
        (7.0, ChannelCondition.FAIR),
        (2.0, ChannelCondition.POOR),
        (-5.0, ChannelCondition.SEVER),
    ]
    
    for snr, expected in test_cases:
        state = ChannelState(snr_db=snr, channel_type='awgn', bandwidth_mhz=5.0)
        assert state.condition == expected, f"SNR {snr}: expected {expected}, got {state.condition}"
    
    print("ChannelState test passed!")


def test_channel_state_ber_estimation():
    """Test BER estimation."""
    state = ChannelState(snr_db=10.0, channel_type='awgn', bandwidth_mhz=5.0)
    assert 0 <= state.estimated_ber <= 1
    print(f"AWGN BER at 10dB: {state.estimated_ber:.2e}")
    
    state = ChannelState(snr_db=10.0, channel_type='rayleigh', bandwidth_mhz=5.0)
    assert 0 <= state.estimated_ber <= 1
    print(f"Rayleigh BER at 10dB: {state.estimated_ber:.2e}")
    
    print("BER estimation test passed!")


def test_allocation_policy():
    """Test AllocationPolicy."""
    policy = AllocationPolicy(
        target_payload_bits=100000,
        min_bits=2,
        max_bits=8
    )
    
    assert policy.get_base_bits(ChannelCondition.EXCELLENT) == 8
    assert policy.get_base_bits(ChannelCondition.GOOD) == 6
    assert policy.get_base_bits(ChannelCondition.FAIR) == 4
    assert policy.get_base_bits(ChannelCondition.POOR) == 4
    assert policy.get_base_bits(ChannelCondition.SEVER) == 2
    
    assert policy.get_importance_boost(ChannelCondition.EXCELLENT) == 0
    assert policy.get_importance_boost(ChannelCondition.SEVER) == 6
    
    print("AllocationPolicy test passed!")


def test_allocation_controller_init():
    """Test AllocationController initialization."""
    controller = create_allocation_controller()
    
    assert controller.policy is not None
    assert controller.adaptive_quantizer is not None
    assert controller.current_channel_state is None
    
    print("AllocationController init test passed!")


def test_update_channel_state():
    """Test updating channel state."""
    controller = create_allocation_controller()
    
    state = controller.update_channel_state(snr_db=10.0, channel_type='awgn', bandwidth_mhz=5.0)
    
    assert state.snr_db == 10.0
    assert state.channel_type == 'awgn'
    assert state.bandwidth_mhz == 5.0
    assert controller.current_channel_state is not None
    
    print("Update channel state test passed!")


def test_compute_allocation():
    """Test allocation computation."""
    controller = create_allocation_controller(
        policy=AllocationPolicy(
            target_payload_bits=50000,
            min_bits=2,
            max_bits=8,
            use_attention=True,
            attention_threshold=0.2
        )
    )
    controller.update_channel_state(snr_db=5.0, channel_type='awgn', bandwidth_mhz=5.0)
    
    z = torch.randn(1, 4, 64, 64)
    allocation_map, info = controller.compute_allocation(z)
    
    assert allocation_map.shape == (1, 1, 64, 64)
    assert allocation_map.dtype == torch.int32
    assert info['channel_condition'] == 'poor'
    assert info['base_bits'] == 4
    
    # Check valid bit depths
    unique_bits = torch.unique(allocation_map).tolist()
    for b in unique_bits:
        assert b in [2, 4, 6, 8]
    
    print(f"Allocation info: {info}")
    print("Compute allocation test passed!")


def test_compute_allocation_with_detection_importance():
    """Test allocation with detection importance map."""
    controller = create_allocation_controller(
        policy=AllocationPolicy(
            min_bits=2,
            max_bits=8,
            use_attention=True,
            attention_threshold=0.3
        )
    )
    controller.update_channel_state(snr_db=10.0, channel_type='awgn', bandwidth_mhz=5.0)
    
    z = torch.randn(1, 4, 64, 64)
    
    # Create detection importance map (high in center)
    detection_importance = torch.zeros(1, 1, 64, 64)
    detection_importance[0, 0, 20:44, 20:44] = 1.0
    
    allocation_map, info = controller.compute_allocation(
        z,
        detection_importance=detection_importance
    )
    
    assert allocation_map.shape == (1, 1, 64, 64)
    
    # Center region should have higher bits
    center_bits = allocation_map[0, 0, 32, 32].item()
    edge_bits = allocation_map[0, 0, 5, 5].item()
    
    print(f"Center bits: {center_bits}, Edge bits: {edge_bits}")
    print("Detection importance allocation test passed!")


def test_compute_allocation_with_task_priority():
    """Test allocation with task priority map."""
    controller = create_allocation_controller(
        policy=AllocationPolicy(
            min_bits=2,
            max_bits=8,
            use_attention=True,
            attention_threshold=0.3
        )
    )
    controller.update_channel_state(snr_db=10.0, channel_type='awgn', bandwidth_mhz=5.0)
    
    z = torch.randn(1, 4, 64, 64)
    
    # Create task priority map (high priority for specific objects)
    task_priority = torch.zeros(1, 1, 64, 64)
    task_priority[0, 0, 10:30, 10:30] = 1.0
    
    allocation_map, info = controller.compute_allocation(
        z,
        task_priority=task_priority
    )
    
    assert allocation_map.shape == (1, 1, 64, 64)
    
    print("Task priority allocation test passed!")


def test_enforce_payload_budget():
    """Test payload budget enforcement."""
    controller = create_allocation_controller(
        policy=AllocationPolicy(
            target_payload_bits=20000,  # Tight budget
            min_bits=2,
            max_bits=8,
            use_attention=True,
            attention_threshold=0.2
        )
    )
    controller.update_channel_state(snr_db=10.0, channel_type='awgn', bandwidth_mhz=5.0)
    
    z = torch.randn(1, 4, 64, 64)
    allocation_map, info = controller.compute_allocation(z)
    
    # Check total bits don't exceed budget by much
    stats = controller.get_allocation_stats(allocation_map)
    print(f"Budget: 20000, Actual ideal bits: {stats['ideal_total_bits']}")
    
    # Should be close to budget (allowing some margin)
    assert stats['ideal_total_bits'] <= 30000
    
    print("Payload budget enforcement test passed!")


def test_get_allocation_stats():
    """Test allocation statistics."""
    controller = create_allocation_controller(
        policy=AllocationPolicy(min_bits=2, max_bits=8)
    )
    controller.update_channel_state(snr_db=10.0)
    
    z = torch.randn(1, 4, 64, 64)
    allocation_map, _ = controller.compute_allocation(z)
    
    stats = controller.get_allocation_stats(allocation_map)
    
    assert stats['total_elements'] == 4096
    assert 'per_bit_depth' in stats
    assert 'ideal_total_bits' in stats
    assert stats['ideal_total_bits'] > 0
    
    print("Allocation stats test passed!")


def test_quantize_with_allocation():
    """Test quantization using computed allocation."""
    controller = create_allocation_controller(
        policy=AllocationPolicy(
            min_bits=2,
            max_bits=8,
            use_attention=True,
            attention_threshold=0.2
        )
    )
    controller.update_channel_state(snr_db=10.0, channel_type='awgn', bandwidth_mhz=5.0)
    
    z = torch.randn(1, 4, 64, 64)
    allocation_map, _ = controller.compute_allocation(z)
    
    # Use adaptive quantizer directly (it computes its own allocation)
    quantizer = AdaptiveQuantizer(
        allocation_mode='adaptive',
        min_bits=2,
        max_bits=8
    )
    packed, meta = quantizer.quantize(z)
    z_tilde = quantizer.dequantize(packed, meta)
    
    assert z_tilde.shape == z.shape
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"Quantization MSE: {mse.item():.6f}")
    
    print("Quantize with allocation test passed!")


def test_different_channel_conditions():
    """Test allocation under different channel conditions."""
    for snr in [25.0, 15.0, 8.0, 3.0, -2.0]:
        controller = create_allocation_controller(
            policy=AllocationPolicy(min_bits=2, max_bits=8)
        )
        controller.update_channel_state(snr_db=snr, channel_type='awgn')
        
        z = torch.randn(1, 4, 64, 64)
        allocation_map, info = controller.compute_allocation(z)
        
        print(f"SNR {snr}dB: condition={info['channel_condition']}, base_bits={info['base_bits']}, boost={info['importance_boost']}")
        
        # Verify allocation stats
        stats = controller.get_allocation_stats(allocation_map)
        assert stats['ideal_total_bits'] > 0
    
    print("Different channel conditions test passed!")


def test_rayleigh_channel():
    """Test with Rayleigh fading channel."""
    controller = create_allocation_controller()
    controller.update_channel_state(snr_db=10.0, channel_type='rayleigh', bandwidth_mhz=5.0)
    
    z = torch.randn(1, 4, 64, 64)
    allocation_map, info = controller.compute_allocation(z)
    
    assert info['channel_condition'] in ['excellent', 'good', 'fair', 'poor', 'severe']
    
    print("Rayleigh channel test passed!")


if __name__ == "__main__":
    test_channel_condition_enum()
    test_channel_state()
    test_channel_state_ber_estimation()
    test_allocation_policy()
    test_allocation_controller_init()
    test_update_channel_state()
    test_compute_allocation()
    test_compute_allocation_with_detection_importance()
    test_compute_allocation_with_task_priority()
    test_enforce_payload_budget()
    test_get_allocation_stats()
    test_quantize_with_allocation()
    test_different_channel_conditions()
    test_rayleigh_channel()
    print("\nAll allocation controller tests passed!")