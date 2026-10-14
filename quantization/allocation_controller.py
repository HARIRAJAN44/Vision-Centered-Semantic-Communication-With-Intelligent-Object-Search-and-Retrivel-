"""
Channel-Aware Allocation Controller

Connects channel state (SNR, bandwidth) to adaptive quantization allocation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from quantization.adaptive_quantizer import AdaptiveQuantizer, calculate_payload_bits


class ChannelCondition(Enum):
    """Channel condition categories."""
    EXCELLENT = "excellent"   # SNR > 20 dB
    GOOD = "good"             # SNR 10-20 dB
    FAIR = "fair"             # SNR 5-10 dB
    POOR = "poor"             # SNR 0-5 dB
    SEVER = "severe"          # SNR < 0 dB


@dataclass
class ChannelState:
    """Current channel state information."""
    snr_db: float
    channel_type: str  # 'awgn', 'rayleigh'
    bandwidth_mhz: float
    estimated_ber: float = 0.0
    condition: ChannelCondition = ChannelCondition.FAIR
    
    def __post_init__(self):
        self.condition = self._classify_condition()
        self.estimated_ber = self._estimate_ber()
    
    def _classify_condition(self) -> ChannelCondition:
        if self.snr_db > 20:
            return ChannelCondition.EXCELLENT
        elif self.snr_db > 10:
            return ChannelCondition.GOOD
        elif self.snr_db > 5:
            return ChannelCondition.FAIR
        elif self.snr_db > 0:
            return ChannelCondition.POOR
        else:
            return ChannelCondition.SEVER
    
    def _estimate_ber(self) -> float:
        """Estimate bit error rate based on SNR and channel type."""
        snr_linear = 10 ** (self.snr_db / 10.0)
        
        if self.channel_type == 'awgn':
            # BPSK BER = 0.5 * erfc(sqrt(SNR))
            from math import erfc, sqrt
            return 0.5 * erfc(sqrt(snr_linear))
        elif self.channel_type == 'rayleigh':
            # Rayleigh BPSK BER = 0.5 * (1 - sqrt(SNR/(1+SNR)))
            from math import sqrt
            return 0.5 * (1 - sqrt(snr_linear / (1 + snr_linear)))
        else:
            return 0.5


@dataclass
class AllocationPolicy:
    """Policy for bit allocation based on channel conditions."""
    
    # Base bit depth for each condition
    condition_base_bits: Dict[ChannelCondition, int] = field(default_factory=lambda: {
        ChannelCondition.EXCELLENT: 8,
        ChannelCondition.GOOD: 6,
        ChannelCondition.FAIR: 4,
        ChannelCondition.POOR: 4,
        ChannelCondition.SEVER: 2
    })
    
    # Bit depth boost for important regions per condition
    condition_importance_boost: Dict[ChannelCondition, int] = field(default_factory=lambda: {
        ChannelCondition.EXCELLENT: 0,
        ChannelCondition.GOOD: 2,
        ChannelCondition.FAIR: 4,
        ChannelCondition.POOR: 4,
        ChannelCondition.SEVER: 6
    })
    
    # Minimum bit depth
    min_bits: int = 2
    
    # Maximum bit depth
    max_bits: int = 8
    
    # Target payload budget (bits)
    target_payload_bits: Optional[int] = None
    
    # Enable attention-based allocation
    use_attention: bool = True
    
    # Attention threshold
    attention_threshold: float = 0.2
    
    def get_base_bits(self, condition: ChannelCondition) -> int:
        return self.condition_base_bits.get(condition, 4)
    
    def get_importance_boost(self, condition: ChannelCondition) -> int:
        return self.condition_importance_boost.get(condition, 0)


class AllocationController:
    """
    Channel-aware adaptive bit allocation controller.
    
    Takes channel state and image content (attention, detections) to produce
    bit allocation for the adaptive quantizer.
    """
    
    def __init__(
        self,
        policy: Optional[AllocationPolicy] = None,
        adaptive_quantizer: Optional[AdaptiveQuantizer] = None
    ):
        self.policy = policy or AllocationPolicy()
        self.adaptive_quantizer = adaptive_quantizer or AdaptiveQuantizer()
        self.current_channel_state: Optional[ChannelState] = None
        self.allocation_history: List[Dict] = []
    
    def update_channel_state(
        self,
        snr_db: float,
        channel_type: str = 'awgn',
        bandwidth_mhz: float = 5.0
    ) -> ChannelState:
        """Update and return current channel state."""
        self.current_channel_state = ChannelState(
            snr_db=snr_db,
            channel_type=channel_type,
            bandwidth_mhz=bandwidth_mhz
        )
        return self.current_channel_state
    
    def compute_allocation(
        self,
        z: torch.Tensor,
        attention_map: Optional[torch.Tensor] = None,
        detection_importance: Optional[torch.Tensor] = None,
        task_priority: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute bit allocation map based on channel state and content.
        
        Args:
            z: Latent tensor (B, C, H, W)
            attention_map: Attention map (B, 1, H, W)
            detection_importance: Detection importance map (B, 1, H, W) 
            task_priority: Task priority map (B, 1, H, W)
            
        Returns:
            allocation_map: Bit allocation per spatial location
            info: Allocation info dictionary
        """
        if self.current_channel_state is None:
            raise ValueError("Channel state not set. Call update_channel_state first.")
        
        B, C, H, W = z.shape
        condition = self.current_channel_state.condition
        
        # Base bits from channel condition
        base_bits = self.policy.get_base_bits(condition)
        
        # Compute attention map if not provided
        if attention_map is None and self.policy.use_attention:
            attention_map = self._compute_attention_map(z)
        
        # Normalize attention to [0, 1]
        if attention_map is not None:
            attention_norm = self._normalize_attention(attention_map)
        else:
            attention_norm = torch.ones((B, 1, H, W), device=z.device) * 0.5
        
        # Combine importance sources
        importance = attention_norm
        
        if detection_importance is not None:
            det_norm = self._normalize_attention(detection_importance)
            importance = torch.max(importance, det_norm)
        
        if task_priority is not None:
            task_norm = self._normalize_attention(task_priority)
            importance = torch.max(importance, task_norm)
        
        # Compute allocation per spatial location
        allocation = torch.full((B, 1, H, W), base_bits, dtype=torch.int32, device=z.device)
        
        # Boost important regions
        boost = self.policy.get_importance_boost(condition)
        
        # Threshold-based allocation
        if self.policy.use_attention:
            high_importance = importance >= self.policy.attention_threshold
            allocation[high_importance] = torch.clamp(
                allocation[high_importance] + boost,
                self.policy.min_bits,
                self.policy.max_bits
            )
            
            # Reduce bits for low importance regions to meet budget
            low_importance = importance < self.policy.attention_threshold
            allocation[low_importance] = torch.clamp(
                allocation[low_importance] - 2,
                self.policy.min_bits,
                self.policy.max_bits
            )
        
        # Enforce payload budget if specified
        if self.policy.target_payload_bits is not None:
            allocation = self._enforce_payload_budget(z, allocation, importance)
        
        # Ensure valid bit depths (2, 4, 6, 8)
        allocation = self._quantize_allocation_depths(allocation)
        
        info = {
            'channel_condition': condition.value,
            'snr_db': self.current_channel_state.snr_db,
            'base_bits': base_bits,
            'importance_boost': boost,
            'attention_threshold': self.policy.attention_threshold,
            'payload_budget': self.policy.target_payload_bits
        }
        
        return allocation, info
    
    def _compute_attention_map(self, z: torch.Tensor) -> torch.Tensor:
        """Compute attention map from latent tensor."""
        return torch.mean(torch.abs(z), dim=1, keepdim=True)
    
    def _normalize_attention(self, attention_map: torch.Tensor) -> torch.Tensor:
        """Normalize attention map to [0, 1]."""
        B = attention_map.shape[0]
        A_min = attention_map.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        A_max = attention_map.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        
        diff = A_max - A_min
        diff[diff == 0] = 1.0
        return (attention_map - A_min) / diff
    
    def _enforce_payload_budget(
        self,
        z: torch.Tensor,
        allocation: torch.Tensor,
        importance: torch.Tensor
    ) -> torch.Tensor:
        """Reduce allocation to meet payload budget."""
        B, C, H, W = z.shape
        total_elements = B * C * H * W
        
        # Current estimated bits
        current_bits = torch.sum(allocation * C).item()  # Per spatial location, multiplied by channels
        
        target_bits = self.policy.target_payload_bits
        
        if current_bits <= target_bits:
            return allocation
        
        # Need to reduce bits - start from least important regions
        flat_importance = importance.flatten()
        flat_allocation = allocation.flatten()
        
        # Sort indices by importance (ascending)
        sorted_indices = torch.argsort(flat_importance)
        
        bits_to_reduce = current_bits - target_bits
        
        for idx in sorted_indices:
            if bits_to_reduce <= 0:
                break
            
            current_bit = flat_allocation[idx].item()
            if current_bit > self.policy.min_bits:
                reduction = min(2, current_bit - self.policy.min_bits)
                flat_allocation[idx] = current_bit - reduction
                bits_to_reduce -= reduction * C  # Multiply by channels
        
        return flat_allocation.reshape(B, 1, H, W)
    
    def _quantize_allocation_depths(self, allocation: torch.Tensor) -> torch.Tensor:
        """Quantize allocation to valid bit depths (2, 4, 6, 8)."""
        valid_depths = torch.tensor([2, 4, 6, 8], device=allocation.device)
        
        # For each element, find closest valid depth
        flat = allocation.flatten()
        for i in range(len(flat)):
            depth = flat[i].item()
            closest = valid_depths[torch.argmin(torch.abs(valid_depths - depth))]
            flat[i] = closest
        
        return flat.reshape(allocation.shape)
    
    def create_quantizer(self, allocation_map: torch.Tensor) -> AdaptiveQuantizer:
        """Create an adaptive quantizer with the computed allocation."""
        # Use adaptive mode with the pre-computed allocation
        quantizer = AdaptiveQuantizer(
            allocation_mode='adaptive',
            min_bits=self.policy.min_bits,
            max_bits=self.policy.max_bits
        )
        # We'll use the allocation map directly in quantize
        return quantizer
    
    def quantize_with_allocation(
        self,
        z: torch.Tensor,
        allocation_map: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize latent tensor using the provided allocation map.
        
        This uses the adaptive quantizer's per-region quantization.
        """
        quantizer = AdaptiveQuantizer(allocation_mode='adaptive')
        return quantizer.quantize(z)
    
    def get_allocation_stats(self, allocation_map: torch.Tensor) -> Dict:
        """Get statistics about the allocation."""
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


def create_allocation_controller(
    policy: Optional[AllocationPolicy] = None,
    **kwargs
) -> AllocationController:
    """Factory function to create an allocation controller."""
    if policy is None:
        policy = AllocationPolicy(**kwargs)
    return AllocationController(policy=policy)


if __name__ == "__main__":
    print("Testing Channel-Aware Allocation Controller...")
    
    # Create policy
    policy = AllocationPolicy(
        target_payload_bits=50000,
        min_bits=2,
        max_bits=8,
        use_attention=True,
        attention_threshold=0.2
    )
    
    # Create controller
    controller = create_allocation_controller(policy)
    
    # Update channel state
    channel_state = controller.update_channel_state(snr_db=5.0, channel_type='awgn', bandwidth_mhz=5.0)
    print(f"Channel state: {channel_state}")
    print(f"Condition: {channel_state.condition}")
    print(f"Estimated BER: {channel_state.estimated_ber:.2e}")
    
    # Test latent tensor
    z = torch.randn(1, 4, 64, 64)
    
    # Compute allocation
    allocation_map, info = controller.compute_allocation(z)
    print(f"Allocation info: {info}")
    
    # Get stats
    stats = controller.get_allocation_stats(allocation_map)
    print(f"Allocation stats: {stats}")
    
    # Quantize with allocation
    quantizer = AdaptiveQuantizer(allocation_mode='adaptive', min_bits=2, max_bits=8)
    packed, meta = quantizer.quantize(z)
    print(f"Packed bytes: {len(packed)}")
    print(f"Metadata mode: {meta.get('allocation_mode')}")
    
    # Dequantize
    z_tilde = quantizer.dequantize(packed, meta)
    mse = torch.nn.functional.mse_loss(z, z_tilde)
    print(f"MSE: {mse.item():.6f}")
    
    # Payload calculation
    payload = calculate_payload_bits(meta)
    print(f"Payload: {payload}")
    
    print("\nAllocation Controller test complete!")