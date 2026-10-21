"""
Task-Aware Semantic Communication

Provides task profiles that influence resource allocation priorities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class TaskType(Enum):
    """Supported task types."""
    CUSTOM = "custom"


@dataclass
class TaskProfile:
    """
    Configuration for a task profile.
    
    Defines how different factors contribute to region priority.
    """
    task_type: TaskType
    name: str
    
    # Weight for each priority factor (should sum to ~1.0 for normalization)
    attention_weight: float = 0.3
    detection_confidence_weight: float = 0.3
    class_priority_weight: float = 0.2
    task_relevance_weight: float = 0.2
    
    # Class priorities (higher = more important for this task)
    class_priorities: Dict[str, float] = field(default_factory=dict)
    
    # Task-specific relevance map generator
    relevance_fn: Optional[Callable] = None
    
    # Minimum priority threshold
    min_priority: float = 0.0
    
    # Maximum priority
    max_priority: float = 1.0
    
    def get_class_priority(self, class_name: str) -> float:
        """Get priority for a class, defaulting to 1.0."""
        return self.class_priorities.get(class_name.lower(), 1.0)
    
    def compute_priority(
        self,
        attention: torch.Tensor,
        detection_confidence: Optional[torch.Tensor] = None,
        class_names: Optional[List[List[str]]] = None,
        task_relevance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined priority map.
        
        Args:
            attention: Attention map (B, 1, H, W)
            detection_confidence: Detection confidence map (B, 1, H, W)
            class_names: List of class names per spatial location
            task_relevance: Task-specific relevance map (B, 1, H, W)
            
        Returns:
            Combined priority map (B, 1, H, W) normalized to [0, 1]
        """
    # Handle None attention - use a uniform map if attention is None
        if attention is None:
            attention = torch.ones(1, 1, 64, 64, device='cpu')
        B, _, H, W = attention.shape
        device = attention.device
        
        # Normalize attention to [0, 1]
        attention_norm = self._normalize(attention)
        
        # Detection confidence contribution
        if detection_confidence is not None:
            det_norm = self._normalize(detection_confidence)
        else:
            det_norm = torch.ones_like(attention_norm) * 0.5
        
        # Class priority contribution
        if class_names is not None:
            class_priority_map = self._compute_class_priority_map(class_names, (H, W), device)
            class_norm = self._normalize(class_priority_map)
        else:
            class_norm = torch.ones_like(attention_norm) * 0.5
        
        # Task relevance contribution
        if task_relevance is not None:
            task_norm = self._normalize(task_relevance)
        else:
            task_norm = torch.ones_like(attention_norm) * 0.5
        
        # Weighted combination
        priority = (
            self.attention_weight * attention_norm +
            self.detection_confidence_weight * det_norm +
            self.class_priority_weight * class_norm +
            self.task_relevance_weight * task_norm
        )
        
        # Clamp to range
        priority = torch.clamp(priority, self.min_priority, self.max_priority)
        
        return priority
    
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] per batch."""
        B = tensor.shape[0]
        flat = tensor.view(B, -1)
        t_min = flat.min(dim=1)[0].view(B, 1, 1, 1)
        t_max = flat.max(dim=1)[0].view(B, 1, 1, 1)
        diff = t_max - t_min
        diff[diff == 0] = 1.0
        return (tensor - t_min) / diff
    
    def _compute_class_priority_map(
        self,
        class_names: List[List[str]],
        spatial_shape: Tuple[int, int],
        device: torch.device
    ) -> torch.Tensor:
        """Create class priority map from class names."""
        H, W = spatial_shape
        priority_map = torch.ones(1, 1, H, W, device=device)
        
        # This is a simplified version - in practice, you'd map class names
        # to spatial locations based on detection results
        # For now, return uniform priority
        return priority_map


def get_task_profile(task_type: TaskType) -> TaskProfile:
    """Get default task profile (custom)."""
    return TaskProfile(
        task_type=TaskType.CUSTOM,
        name="Custom",
        attention_weight=0.25,
        detection_confidence_weight=0.25,
        class_priority_weight=0.25,
        task_relevance_weight=0.25
    )


def create_custom_task_profile(
    name: str,
    attention_weight: float = 0.25,
    detection_confidence_weight: float = 0.25,
    class_priority_weight: float = 0.25,
    task_relevance_weight: float = 0.25,
    class_priorities: Optional[Dict[str, float]] = None
) -> TaskProfile:
    """Create a custom task profile."""
    return TaskProfile(
        task_type=TaskType.CUSTOM,
        name=name,
        attention_weight=attention_weight,
        detection_confidence_weight=detection_confidence_weight,
        class_priority_weight=class_priority_weight,
        task_relevance_weight=task_relevance_weight,
        class_priorities=class_priorities or {}
    )


class TaskAwareAllocator:
    """
    Task-aware bit allocator that uses task profiles to guide allocation.
    """
    
    def __init__(
        self,
        task_profile: TaskProfile,
        min_bits: int = 2,
        max_bits: int = 8,
        base_bits: int = 4
    ):
        self.task_profile = task_profile
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.base_bits = base_bits
    
    def compute_allocation(
        self,
        z: torch.Tensor,
        attention_map: Optional[torch.Tensor] = None,
        detection_confidence: Optional[torch.Tensor] = None,
        class_names: Optional[List[List[str]]] = None,
        task_relevance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute bit allocation based on task profile.
        
        Returns:
            allocation_map: Bit depth per spatial location
            info: Allocation info
        """
        B, C, H, W = z.shape
        
        # Compute attention if not provided
        if attention_map is None:
            attention_map = torch.mean(torch.abs(z), dim=1, keepdim=True)
        
        # Compute priority map
        priority = self.task_profile.compute_priority(
            attention=attention_map,
            detection_confidence=detection_confidence,
            class_names=class_names,
            task_relevance=task_relevance
        )
        
        # Map priority to bit depth
        # Priority in [0, 1] -> bits in [min_bits, max_bits]
        allocation = torch.zeros((B, 1, H, W), dtype=torch.int32, device=z.device)
        
        # Use 4 levels: min, low, medium, high, max
        levels = [self.min_bits]
        if self.max_bits > self.min_bits + 2:
            levels.append(self.min_bits + 2)
        if self.max_bits > self.min_bits + 4:
            levels.append(self.min_bits + 4)
        levels.append(self.max_bits)
        
        num_levels = len(levels)
        for i, bits in enumerate(levels):
            lower = i / num_levels
            upper = (i + 1) / num_levels
            mask = (priority >= lower) & (priority < upper)
            if i == num_levels - 1:
                mask = mask | (priority >= upper)
            allocation[mask] = bits
        
        info = {
            'task_type': self.task_profile.task_type.value,
            'task_name': self.task_profile.name,
            'min_bits': self.min_bits,
            'max_bits': self.max_bits,
            'base_bits': self.base_bits,
            'weights': {
                'attention': self.task_profile.attention_weight,
                'detection_confidence': self.task_profile.detection_confidence_weight,
                'class_priority': self.task_profile.class_priority_weight,
                'task_relevance': self.task_profile.task_relevance_weight
            }
        }
        
        return allocation, info


@dataclass
class TaskEvaluationResult:
    """Results of task-aware evaluation."""
    task_type: str
    task_name: str
    
    # Reconstruction metrics
    ssim: float
    psnr: float
    lpips: float
    clip_similarity: float
    
    # Task-specific metrics
    detection_precision: Optional[float] = None
    detection_recall: Optional[float] = None
    detection_map: Optional[float] = None
    retrieval_precision_at_k: Optional[Dict[int, float]] = None
    retrieval_recall_at_k: Optional[Dict[int, float]] = None
    
    # Payload info
    payload_bits: int = 0
    compression_ratio: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'task_type': self.task_type,
            'task_name': self.task_name,
            'ssim': self.ssim,
            'psnr': self.psnr,
            'lpips': self.lpips,
            'clip_similarity': self.clip_similarity,
            'detection_precision': self.detection_precision,
            'detection_recall': self.detection_recall,
            'detection_map': self.detection_map,
            'retrieval_precision_at_k': self.retrieval_precision_at_k,
            'retrieval_recall_at_k': self.retrieval_recall_at_k,
            'payload_bits': self.payload_bits,
            'compression_ratio': self.compression_ratio
        }


class TaskEvaluator:
    """
    Evaluates task performance after transmission.
    """
    
    def __init__(
        self,
        evaluator,  # Existing metrics.Evaluator
        object_detector=None,
        search_engine=None
    ):
        self.evaluator = evaluator
        self.object_detector = object_detector
        self.search_engine = search_engine
    
    def evaluate(
        self,
        original_image,
        reconstructed_image,
        task_profile: TaskProfile,
        original_detections: Optional[List] = None,
        reconstructed_detections: Optional[List] = None,
        original_embedding: Optional[np.ndarray] = None,
        reconstructed_embedding: Optional[np.ndarray] = None,
        payload_bits: int = 0,
        original_bits: int = 512 * 512 * 3 * 8
    ) -> TaskEvaluationResult:
        """
        Evaluate task performance.
        
        Args:
            original_image: Original PIL Image
            reconstructed_image: Reconstructed PIL Image
            task_profile: Task profile used
            original_detections: Detections on original image
            reconstructed_detections: Detections on reconstructed image
            original_embedding: CLIP embedding of original
            reconstructed_embedding: CLIP embedding of reconstructed
            payload_bits: Transmitted payload bits
            original_bits: Original image bits
            
        Returns:
            TaskEvaluationResult
        """
        # Reconstruction metrics
        metrics = self.evaluator.evaluate(original_image, reconstructed_image)
        
        result = TaskEvaluationResult(
            task_type=task_profile.task_type.value,
            task_name=task_profile.name,
            ssim=metrics['ssim'],
            psnr=metrics.get('psnr', 0.0),
            lpips=metrics['lpips'],
            clip_similarity=metrics['clip'],
            payload_bits=payload_bits,
            compression_ratio=payload_bits / original_bits if original_bits > 0 else 0.0
        )
        
        # Task-specific evaluation
        if task_profile.task_type == TaskType.OBJECT_IDENTIFICATION:
            if original_detections and reconstructed_detections and self.object_detector:
                det_metrics = self._evaluate_detection(
                    original_detections, reconstructed_detections
                )
                result.detection_precision = det_metrics.get('precision')
                result.detection_recall = det_metrics.get('recall')
                result.detection_map = det_metrics.get('map')
        
        elif task_profile.task_type == TaskType.OBJECT_RETRIEVAL:
            if self.search_engine and original_embedding is not None and reconstructed_embedding is not None:
                ret_metrics = self._evaluate_retrieval(
                    original_embedding, reconstructed_embedding
                )
                result.retrieval_precision_at_k = ret_metrics.get('precision_at_k')
                result.retrieval_recall_at_k = ret_metrics.get('recall_at_k')
        
        return result
    
    def _evaluate_detection(
        self,
        original_detections: List,
        reconstructed_detections: List,
        iou_threshold: float = 0.5
    ) -> Dict:
        """Evaluate object detection performance."""
        # Simplified IoU-based matching
        matched = 0
        total_original = len(original_detections)
        total_reconstructed = len(reconstructed_detections)
        
        for orig_det in original_detections:
            best_iou = 0
            for recon_det in reconstructed_detections:
                if orig_det.class_name == recon_det.class_name:
                    iou = self._compute_iou(orig_det.bbox, recon_det.bbox)
                    best_iou = max(best_iou, iou)
            if best_iou >= iou_threshold:
                matched += 1
        
        precision = matched / total_reconstructed if total_reconstructed > 0 else 0
        recall = matched / total_original if total_original > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'map': (precision + recall) / 2  # Simplified mAP
        }
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _evaluate_retrieval(
        self,
        original_embedding: np.ndarray,
        reconstructed_embedding: np.ndarray,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict:
        """Evaluate retrieval performance."""
        # Compute similarity
        sim = np.dot(
            original_embedding / np.linalg.norm(original_embedding),
            reconstructed_embedding / np.linalg.norm(reconstructed_embedding)
        )
        
        # For this simplified evaluation, we use similarity as proxy
        precision_at_k = {k: float(sim) for k in k_values}
        recall_at_k = {k: float(sim) for k in k_values}
        
        return {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k
        }


if __name__ == "__main__":
    print("Testing Task-Aware Module...")
    
    # Test predefined profiles
    for task_type in TaskType:
        profile = get_task_profile(task_type)
        print(f"Profile: {profile.name} ({profile.task_type.value})")
        print(f"  Weights: attn={profile.attention_weight}, det={profile.detection_confidence_weight}, "
              f"class={profile.class_priority_weight}, task={profile.task_relevance_weight}")
        print(f"  Class priorities: {profile.class_priorities}")
    
    # Test custom profile
    custom = create_custom_task_profile(
        name="Surveillance",
        attention_weight=0.2,
        detection_confidence_weight=0.4,
        class_priority_weight=0.3,
        task_relevance_weight=0.1,
        class_priorities={'person': 3.0, 'car': 2.0, 'truck': 2.0}
    )
    print(f"Custom: {custom.name}")
    print(f"  Person priority: {custom.get_class_priority('person')}")
    print(f"  Unknown priority: {custom.get_class_priority('unknown')}")
    
    # Test priority computation
    z = torch.randn(1, 4, 64, 64)
    attention = torch.mean(torch.abs(z), dim=1, keepdim=True)
    
    profile = get_task_profile(TaskType.OBJECT_IDENTIFICATION)
    priority = profile.compute_priority(attention)
    print(f"Priority shape: {priority.shape}, range: [{priority.min():.3f}, {priority.max():.3f}]")
    
    # Test allocator
    allocator = TaskAwareAllocator(profile, min_bits=2, max_bits=8)
    allocation, info = allocator.compute_allocation(z, attention_map=attention)
    print(f"Allocation shape: {allocation.shape}")
    print(f"Allocation info: {info}")
    
    stats = {}
    flat = allocation.flatten()
    unique, counts = torch.unique(flat, return_counts=True)
    for bits, count in zip(unique.tolist(), counts.tolist()):
        stats[bits] = int(count)
    print(f"Bit distribution: {stats}")
    
    print("\nTask-Aware module test complete!")