"""
Object Detection Module for VCSC

Provides YOLO-based object detection and ROI extraction for semantic communication.
"""

from .detector import (
    YOLODetector,
    Detection,
    LatentROI,
    create_detector,
    get_available_models
)

__all__ = [
    'YOLODetector',
    'Detection',
    'LatentROI',
    'create_detector',
    'get_available_models'
]