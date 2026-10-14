import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
import os


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    image_width: int
    image_height: int
    
    def to_dict(self) -> Dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'image_width': self.image_width,
            'image_height': self.image_height
        }
    
    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        return self.bbox
    
    @property
    def bbox_xywh(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class LatentROI:
    """Region of Interest in latent space (64x64)"""
    latent_x1: int
    latent_y1: int
    latent_x2: int
    latent_y2: int
    detection: Detection
    
    def to_dict(self) -> Dict:
        return {
            'latent_x1': self.latent_x1,
            'latent_y1': self.latent_y1,
            'latent_x2': self.latent_x2,
            'latent_y2': self.latent_y2,
            'detection': self.detection.to_dict()
        }
    
    @property
    def latent_bbox(self) -> Tuple[int, int, int, int]:
        return (self.latent_x1, self.latent_y1, self.latent_x2, self.latent_y2)
    
    @property
    def latent_width(self) -> int:
        return self.latent_x2 - self.latent_x1
    
    @property
    def latent_height(self) -> int:
        return self.latent_y2 - self.latent_y1
    
    @property
    def latent_area(self) -> int:
        return self.latent_width * self.latent_height


class YOLODetector:
    """
    Lightweight YOLO detector for object detection and ROI extraction.
    
    Supports YOLOv8/YOLOv11 nano models for real-time inference.
    Caches model for efficient repeated inference.
    """
    
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 100,
        input_size: int = 640,
        latent_size: int = 64,
        image_size: int = 512
    ):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.input_size = input_size
        self.latent_size = latent_size
        self.image_size = image_size
        
        self.model = None
        self._loaded = False
        self._inference_times = []
        
        self.downsample_factor = image_size // latent_size
    
    def load(self) -> bool:
        """Load the YOLO model. Returns True if successful."""
        if self._loaded:
            return True
        
        try:
            from ultralytics import YOLO
            print(f"[YOLODetector] Loading model from {self.model_path} on {self.device}...")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.model.fuse()
            self._loaded = True
            print(f"[YOLODetector] Model loaded successfully")
            return True
        except ImportError:
            print(f"[YOLODetector] ERROR: ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            print(f"[YOLODetector] ERROR: Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None
    
    def detect(self, image: Union[Image.Image, np.ndarray, str]) -> List[Detection]:
        """
        Run object detection on image.
        
        Args:
            image: PIL Image, numpy array (HWC RGB), or image path
            
        Returns:
            List of Detection objects
        """
        if not self.is_loaded():
            if not self.load():
                return []
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_width, original_height = image.size
        
        start_time = time.time()
        
        with torch.no_grad():
            results = self.model.predict(
                source=image,
                imgsz=self.input_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                device=self.device,
                verbose=False
            )
        
        inference_time = time.time() - start_time
        self._inference_times.append(inference_time)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    xyxy = box.xyxy[0].cpu().numpy().flatten()
                    conf = float(box.conf[0].cpu().numpy().item())
                    cls_id = int(box.cls[0].cpu().numpy().item())
                    
                    class_name = self.COCO_CLASSES[cls_id] if cls_id < len(self.COCO_CLASSES) else f'class_{cls_id}'
                    
                    x1, y1, x2, y2 = xyxy
                    x1 = max(0, min(x1, original_width - 1))
                    y1 = max(0, min(y1, original_height - 1))
                    x2 = max(0, min(x2, original_width))
                    y2 = max(0, min(y2, original_height))
                    
                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        image_width=original_width,
                        image_height=original_height
                    ))
        
        return detections
    
    def detect_batch(self, images: List[Union[Image.Image, np.ndarray, str]]) -> List[List[Detection]]:
        """Run detection on a batch of images."""
        return [self.detect(img) for img in images]
    
    def get_latent_rois(self, detections: List[Detection], 
                        clamp_to_bounds: bool = True,
                        min_size: int = 1) -> List[LatentROI]:
        """
        Map image-space bounding boxes to latent-space ROIs.
        
        The VAE downsamples by factor of 8 (512 -> 64).
        We map the image-space bbox to latent space using this factor.
        
        Note: This is an approximation. The VAE's receptive field means
        a latent pixel corresponds to a larger image region than just 8x8.
        For more accurate alignment, consider the VAE encoder's effective
        receptive field.
        
        Args:
            detections: List of Detection objects
            clamp_to_bounds: Clamp ROI to latent dimensions
            min_size: Minimum ROI size in latent pixels
            
        Returns:
            List of LatentROI objects
        """
        latent_rois = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            latent_x1 = int(x1 / self.downsample_factor)
            latent_y1 = int(y1 / self.downsample_factor)
            latent_x2 = int(np.ceil(x2 / self.downsample_factor))
            latent_y2 = int(np.ceil(y2 / self.downsample_factor))
            
            if clamp_to_bounds:
                latent_x1 = max(0, min(latent_x1, self.latent_size - 1))
                latent_y1 = max(0, min(latent_y1, self.latent_size - 1))
                latent_x2 = max(1, min(latent_x2, self.latent_size))
                latent_y2 = max(1, min(latent_y2, self.latent_size))
            
            if (latent_x2 - latent_x1) < min_size:
                latent_x2 = min(self.latent_size, latent_x1 + min_size)
            if (latent_y2 - latent_y1) < min_size:
                latent_y2 = min(self.latent_size, latent_y1 + min_size)
            
            latent_rois.append(LatentROI(
                latent_x1=latent_x1,
                latent_y1=latent_y1,
                latent_x2=latent_x2,
                latent_y2=latent_y2,
                detection=det
            ))
        
        return latent_rois
    
    def extract_roi_crops(self, image: Image.Image, detections: List[Detection]) -> List[Image.Image]:
        """Extract image crops for each detection."""
        crops = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
        return crops
    
    def visualize_detections(self, image: Image.Image, detections: List[Detection],
                            draw_labels: bool = True, draw_conf: bool = True,
                            color_map: Optional[Dict[int, Tuple[int, int, int]]] = None) -> Image.Image:
        """Draw detection bounding boxes on image."""
        import cv2
        
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            
            if color_map and det.class_id in color_map:
                color = color_map[det.class_id]
            else:
                np.random.seed(det.class_id)
                color = tuple(np.random.randint(0, 255, 3).tolist())
            
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            
            if draw_labels:
                label = det.class_name
                if draw_conf:
                    label += f" {det.confidence:.2f}"
                
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_np, (x1, y1 - label_h - 4), (x1 + label_w, y1), color, -1)
                cv2.putText(img_np, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_np)
    
    def get_inference_stats(self) -> Dict:
        """Get inference timing statistics."""
        if not self._inference_times:
            return {}
        
        times = self._inference_times
        return {
            'mean_ms': np.mean(times) * 1000,
            'median_ms': np.median(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'num_runs': len(times)
        }
    
    def reset_stats(self):
        self._inference_times = []
    
    def update_thresholds(self, conf_threshold: Optional[float] = None, 
                         iou_threshold: Optional[float] = None):
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold


def create_detector(model_name: str = 'yolov8n.pt', **kwargs) -> YOLODetector:
    """Factory function to create a YOLO detector."""
    return YOLODetector(model_path=model_name, **kwargs)


def get_available_models() -> List[str]:
    """Return list of available YOLO model names."""
    return [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
        'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'
    ]


if __name__ == "__main__":
    print("Testing YOLO Detector...")
    
    detector = create_detector('yolov8n.pt')
    
    if not detector.load():
        print("Failed to load model. Skipping test.")
    else:
        test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        
        detections = detector.detect(test_img)
        print(f"Found {len(detections)} detections")
        
        for det in detections[:3]:
            print(f"  {det.class_name}: {det.confidence:.3f} @ {det.bbox}")
        
        latent_rois = detector.get_latent_rois(detections)
        print(f"Mapped to {len(latent_rois)} latent ROIs")
        
        for roi in latent_rois[:3]:
            print(f"  Latent ROI: {roi.latent_bbox} for {roi.detection.class_name}")
        
        vis_img = detector.visualize_detections(test_img, detections)
        print(f"Visualization image size: {vis_img.size}")
        
        stats = detector.get_inference_stats()
        print(f"Inference stats: {stats}")
    
    print("YOLO Detector test complete!")