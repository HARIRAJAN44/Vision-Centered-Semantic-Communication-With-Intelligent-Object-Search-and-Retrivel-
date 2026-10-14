import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from object_detection.detector import (
    YOLODetector,
    Detection,
    LatentROI,
    create_detector,
    get_available_models
)


class MockYOLO:
    """Mock YOLO model for testing without downloading weights."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def to(self, device):
        return self
    
    def fuse(self):
        return self
    
    def predict(self, source, imgsz, conf, iou, max_det, device, verbose):
        result = Mock()
        
        # Create boxes list that supports len() and indexing
        # Real YOLO returns tensors of shape (1, 4) for xyxy, (1,) for conf/cls
        box1 = Mock()
        box1.xyxy = [torch.tensor([[100.0, 100.0, 200.0, 200.0]])]  # shape (1, 4)
        box1.conf = [torch.tensor([0.9])]  # shape (1,)
        box1.cls = [torch.tensor([0])]  # shape (1,)
        
        box2 = Mock()
        box2.xyxy = [torch.tensor([[300.0, 150.0, 400.0, 250.0]])]
        box2.conf = [torch.tensor([0.8])]
        box2.cls = [torch.tensor([2])]
        
        boxes_list = [box1, box2]
        result.boxes = boxes_list
        
        return [result]


def _mock_yolo_context():
    """Context manager to mock ultralytics.YOLO"""
    return patch('ultralytics.YOLO', MockYOLO)


def test_detection_dataclass():
    """Test Detection dataclass."""
    det = Detection(
        class_id=0,
        class_name='person',
        confidence=0.95,
        bbox=(100.0, 100.0, 200.0, 200.0),
        image_width=512,
        image_height=512
    )
    
    assert det.class_id == 0
    assert det.class_name == 'person'
    assert det.confidence == 0.95
    assert det.bbox == (100.0, 100.0, 200.0, 200.0)
    
    bbox_xywh = det.bbox_xywh
    assert bbox_xywh == (100.0, 100.0, 100.0, 100.0)
    
    center = det.center
    assert center == (150.0, 150.0)
    
    area = det.area
    assert area == 10000.0
    
    d = det.to_dict()
    assert d['class_id'] == 0
    assert d['class_name'] == 'person'
    assert d['bbox'] == (100.0, 100.0, 200.0, 200.0)
    
    print("Detection dataclass test passed!")


def test_latent_roi_dataclass():
    """Test LatentROI dataclass."""
    det = Detection(
        class_id=0,
        class_name='person',
        confidence=0.95,
        bbox=(100.0, 100.0, 200.0, 200.0),
        image_width=512,
        image_height=512
    )
    
    roi = LatentROI(
        latent_x1=12,
        latent_y1=12,
        latent_x2=25,
        latent_y2=25,
        detection=det
    )
    
    assert roi.latent_bbox == (12, 12, 25, 25)
    assert roi.latent_width == 13
    assert roi.latent_height == 13
    assert roi.latent_area == 169
    
    d = roi.to_dict()
    assert d['latent_x1'] == 12
    assert d['latent_y1'] == 12
    assert d['latent_x2'] == 25
    assert d['latent_y2'] == 25
    assert d['detection']['class_name'] == 'person'
    
    print("LatentROI dataclass test passed!")


def test_detector_initialization():
    """Test detector initialization with mocked YOLO."""
    with _mock_yolo_context():
        detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
        assert detector.model_path == 'yolov8n.pt'
        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.25
        assert detector.iou_threshold == 0.45
        assert detector.latent_size == 64
        assert detector.image_size == 512
        assert detector.downsample_factor == 8
    print("Detector initialization test passed!")


def test_detector_load():
    """Test detector model loading."""
    with _mock_yolo_context():
        detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
        success = detector.load()
        assert success
        assert detector.is_loaded()
    print("Detector load test passed!")


def test_detect():
    """Test detection on mock image."""
    with _mock_yolo_context():
        detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
        detector.load()
        
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        detections = detector.detect(img)
        
        assert len(detections) == 2
        assert detections[0].class_name == 'person'
        assert abs(detections[0].confidence - 0.9) < 1e-5
        assert detections[1].class_name == 'car'
        assert abs(detections[1].confidence - 0.8) < 1e-5
    
    print("Detection test passed!")


def test_get_latent_rois():
    """Test mapping image bboxes to latent ROIs."""
    with _mock_yolo_context():
        detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
        detector.load()
        
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        detections = detector.detect(img)
        
        latent_rois = detector.get_latent_rois(detections)
        
        assert len(latent_rois) == 2
        assert latent_rois[0].latent_bbox == (12, 12, 25, 25)
        # Second bbox: (300, 150, 400, 250) -> latent (37, 18, 50, 32)
        # 300/8=37.5->37, 150/8=18.75->18, ceil(400/8)=50, ceil(250/8)=32
        assert latent_rois[1].latent_bbox == (37, 18, 50, 32)
    
    print("Latent ROI mapping test passed!")


def test_extract_roi_crops():
    """Test ROI crop extraction."""
    with _mock_yolo_context():
        detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
        detector.load()
        
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        detections = detector.detect(img)
        
        crops = detector.extract_roi_crops(img, detections)
        
        assert len(crops) == 2
        assert crops[0].size == (100, 100)
        assert crops[1].size == (100, 100)
    
    print("ROI crop extraction test passed!")


def test_visualize_detections():
    """Test detection visualization."""
    with _mock_yolo_context():
        detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
        detector.load()
        
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        detections = detector.detect(img)
        
        vis_img = detector.visualize_detections(img, detections)
        
        assert vis_img.size == (512, 512)
        assert vis_img.mode == 'RGB'
    
    print("Visualization test passed!")


def test_update_thresholds():
    """Test threshold updates."""
    with _mock_yolo_context():
        detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
        detector.load()
        
        detector.update_thresholds(conf_threshold=0.5, iou_threshold=0.3)
        assert detector.conf_threshold == 0.5
        assert detector.iou_threshold == 0.3
    
    print("Threshold update test passed!")


def test_inference_stats():
    """Test inference statistics."""
    with _mock_yolo_context():
        detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
        detector.load()
        
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        detector.detect(img)
        detector.detect(img)
        
        stats = detector.get_inference_stats()
        assert stats['num_runs'] == 2
        assert 'mean_ms' in stats
        assert 'median_ms' in stats
        
        detector.reset_stats()
        stats = detector.get_inference_stats()
        assert stats == {}
    
    print("Inference stats test passed!")


def test_no_detections():
    """Test handling of images with no detections."""
    mock_yolo = Mock()
    mock_result = Mock()
    mock_result.boxes = None
    mock_yolo.predict.return_value = [mock_result]
    
    with patch('ultralytics.YOLO', return_value=mock_yolo):
        detector = YOLODetector(model_path='yolov8n.pt', device='cpu')
        detector.load()
        
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        detections = detector.detect(img)
        
        assert detections == []
        
        latent_rois = detector.get_latent_rois(detections)
        assert latent_rois == []
    
    print("No detections test passed!")


def test_create_detector_factory():
    """Test factory function."""
    with _mock_yolo_context():
        detector = create_detector('yolov8n.pt', conf_threshold=0.3)
        detector.load()
        
        assert detector.conf_threshold == 0.3
    
    print("Factory function test passed!")


def test_get_available_models():
    """Test available models list."""
    models = get_available_models()
    
    assert 'yolov8n.pt' in models
    assert 'yolo11n.pt' in models
    
    print("Available models test passed!")


if __name__ == "__main__":
    test_detection_dataclass()
    test_latent_roi_dataclass()
    test_detector_initialization()
    test_detector_load()
    test_detect()
    test_get_latent_rois()
    test_extract_roi_crops()
    test_visualize_detections()
    test_update_thresholds()
    test_inference_stats()
    test_no_detections()
    test_create_detector_factory()
    test_get_available_models()
    print("\nAll object detection tests passed!")