import os
import torch
import numpy as np
import time
from preprocessing.image_utils import load_image, preprocess_image, postprocess_image, save_image
from semantic.encoder import SemanticEncoder
from semantic.decoder import SemanticDecoder
from quantization.uniform_quantizer import quantize_uniform_8bit, dequantize_uniform_8bit
from quantization.attention_quantizer import quantize_attention_4bit, dequantize_attention_4bit
from quantization.adaptive_quantizer import (
    AdaptiveQuantizer, quantize_uniform_multi_bit, dequantize_uniform_multi_bit,
    calculate_payload_bits
)
from quantization.allocation_controller import (
    AllocationController, AllocationPolicy, ChannelState, ChannelCondition
)
from semantic.task_aware import (
    TaskType, TaskProfile, get_task_profile, TaskAwareAllocator, TaskEvaluator
)
from communication.channel_pipeline import ChannelPipeline
import argparse
import json


class VCSCPipeline:
    def __init__(
        self,
        use_turbo=True,
        channel_type='awgn',
        device=None,
        enable_object_detection=False,
        enable_retrieval=False,
        retrieval_index_path="faiss_index"
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SemanticEncoder(device=self.device)
        self.decoder = SemanticDecoder(device=self.device)
        self.channel = ChannelPipeline(use_turbo=use_turbo, channel_type=channel_type)
        
        # Object detection
        self.object_detector = None
        self.enable_object_detection = enable_object_detection
        if enable_object_detection:
            self._init_object_detector()
        
        # Retrieval
        self.search_engine = None
        self.enable_retrieval = enable_retrieval
        if enable_retrieval:
            self._init_retrieval(retrieval_index_path)
        
        # Task-aware components
        self.task_allocator = None
        self.task_evaluator = None
        self.current_task_profile = None
        
        # Allocation controller
        self.allocation_controller = None
        self._init_allocation_controller()
        self.working_mode = '8bit'
    
    def _init_object_detector(self):
        try:
            from object_detection.detector import create_detector
            self.object_detector = create_detector('yolov8n.pt')
            self.object_detector.load()
            print("[VCSCPipeline] Object detector initialized")
        except Exception as e:
            print(f"[VCSCPipeline] Object detector initialization failed: {e}")
            self.object_detector = None
            self.enable_object_detection = False
    
    def _init_retrieval(self, index_path):
        try:
            from retrieval.search import create_search_engine
            self.search_engine = create_search_engine(index_path=index_path)
            print("[VCSCPipeline] Retrieval engine initialized")
        except Exception as e:
            print(f"[VCSCPipeline] Retrieval initialization failed: {e}")
            self.search_engine = None
            self.enable_retrieval = False
    
    def _init_allocation_controller(self):
        policy = AllocationPolicy(
            target_payload_bits=None,
            min_bits=2,
            max_bits=8,
            use_attention=True,
            attention_threshold=0.2
        )
        self.allocation_controller = AllocationController(policy=policy)
    
    def _create_detection_importance_map(self, latent_shape, latent_rois):
        """Create an importance map for attention-based quantization.
        
        The importance map highlights regions of interest (ROIs) from object detection
        or provides uniform importance when no ROIs are available.
        
        Args:
            latent_shape: Shape of latent tensor (B, C, H, W)
            latent_rois: List of LatentROI objects from object detection
            
        Returns:
            Tensor of importance weights with same spatial dimensions as latent
        """
        batch_size, channels, height, width = latent_shape
        importance = torch.ones(batch_size, height, width, device=self.device)
        
        if latent_rois:
            for roi in latent_rois:
                x1, y1, x2, y2 = roi.latent_bbox
                # Expand ROI area to 1.5x size for better attention coverage
                h = y2 - y1
                w = x2 - x1
                new_h = int(h * 1.5)
                new_w = int(w * 1.5)
                center_y = (y1 + y2) // 2
                center_x = (x1 + x2) // 2
                y1_new = max(0, center_y - new_h // 2)
                x1_new = max(0, center_x - new_w // 2)
                y2_new = min(height, center_y + new_h // 2)
                x2_new = min(width, center_x + new_w // 2)
                if y2_new > y1_new and x2_new > x1_new:
                    importance[:, y1_new:y2_new, x1_new:x2_new] = 2.0
        
        return importance
    
    def set_task_profile(self, task_type: TaskType, custom_profile: TaskProfile = None):
        """Set the task profile for task-aware allocation."""
        if custom_profile:
            self.current_task_profile = custom_profile
        else:
            self.current_task_profile = get_task_profile(task_type)
        
        self.task_allocator = TaskAwareAllocator(
            self.current_task_profile,
            min_bits=2,
            max_bits=8
        )
        
        # Initialize task evaluator
        from metrics.evaluator import Evaluator
        evaluator = Evaluator(device=self.device)
        self.task_evaluator = TaskEvaluator(
            evaluator,
            object_detector=self.object_detector,
            search_engine=self.search_engine
        )
        
        print(f"[VCSCPipeline] Task profile set: {self.current_task_profile.name}")
    
    def update_channel_state(self, snr_db: float, channel_type: str = 'awgn', bandwidth_mhz: float = 5.0):
        """Update channel state for adaptive allocation."""
        self.channel.channel_type = channel_type
        if self.allocation_controller:
            self.allocation_controller.update_channel_state(snr_db, channel_type, bandwidth_mhz)
    
    def process_image(
        self,
        image_input,
        quant_mode: str,
        snr_db: float,
        threshold: float = 0.2,
        save_dir: str = None,
        task_type: str = None,
        return_detections: bool = False,
        return_embeddings: bool = False
    ):
        if isinstance(image_input, str):
            print(f"--- Processing {image_input} ---")
            img = load_image(image_input)
        else:
            print("--- Processing Uploaded Image ---")
            img = image_input
            
        # Normalize quantization mode
        qmap = {
            'task_aware': 'task_aware',
            'adaptive-multi': 'adaptive',
            'adaptive': 'adaptive',
            'multi-adaptive': 'adaptive',
            '4bit-attn': '4bit-attn',
            '4-bit-attn': '4bit-attn',
            '2bit': '2bit',
            '4bit': '4bit',
            '6bit': '6bit',
            '8bit': '8bit'
        }
        quant_mode = qmap.get(quant_mode, quant_mode)
        
        print(f"Quantization Mode: {quant_mode}, Channel: {self.channel.channel_type}, SNR: {snr_db}dB, Turbo: {self.channel.use_turbo}")
        
        # Update channel state for adaptive allocation
        if self.allocation_controller:
            self.allocation_controller.update_channel_state(snr_db, self.channel.channel_type)
        
        # 1. Preprocess
        img_tensor = preprocess_image(img).to(self.device)
        
        # 2. Encode
        start_enc = time.time()
        z = self.encoder.encode(img_tensor)
        enc_time = time.time() - start_enc
        print(f"Latent shape: {z.shape}, Encoding time: {enc_time:.4f}s")
        if save_dir:
            torch.save(z.cpu(), os.path.join(save_dir, 'latent.pt'))
        
        # 3. Object Detection (if enabled)
        detections = None
        latent_rois = None
        detection_importance = None
        
        if self.enable_object_detection and self.object_detector:
            detections = self.object_detector.detect(img)
            latent_rois = self.object_detector.get_latent_rois(detections)
            detection_importance = self._create_detection_importance_map(z.shape, latent_rois)
            print(f"Detected {len(detections)} objects")
            if save_dir:
                vis_img = self.object_detector.visualize_detections(img, detections)
                vis_img.save(os.path.join(save_dir, 'detections.png'))
        
        # 4. Task-aware priority map
        task_priority = None
        if self.current_task_profile and self.task_allocator:
            task_priority = self.current_task_profile.compute_priority(
                attention=None,  # Will be computed inside
                detection_confidence=detection_importance,
                class_names=self._get_class_names_per_location(detections, z.shape) if detections else None,
                task_relevance=None
            )
        
        # 5. Quantization
        if quant_mode == '8bit':
            q, meta = quantize_uniform_8bit(z)
        elif quant_mode == '4bit-attn':
            q, meta = quantize_attention_4bit(z, threshold=threshold)
        elif quant_mode in ['2bit', '4bit', '6bit']:
            bits = int(quant_mode.replace('bit', ''))
            q, meta = quantize_uniform_multi_bit(z, bits)
        elif quant_mode == 'task_aware':
            if not self.task_allocator:
                self.set_task_profile(TaskType.OBJECT_IDENTIFICATION)
            allocation_map, alloc_info = self.task_allocator.compute_allocation(
                z,
                detection_confidence=detection_importance,
                class_names=self._get_class_names_per_location(detections, z.shape) if detections else None,
                task_relevance=task_priority
            )
            aq = AdaptiveQuantizer(allocation_mode='adaptive', min_bits=2, max_bits=8)
            q, meta = aq.quantize(z, attention_map=allocation_map.float())
            meta['task_aware'] = True
            meta['allocation_info'] = alloc_info
        elif quant_mode in ['adaptive', 'attention', 'channel_aware']:
            aq = AdaptiveQuantizer(
                allocation_mode=quant_mode,
                min_bits=2,
                max_bits=8,
                attention_threshold=threshold,
                snr_db=snr_db
            )
            q, meta = aq.quantize(z)
            meta['adaptive_multi'] = True
            meta['bits_per_position'] = q.shape[-1]  # approximated
        else:
            raise ValueError(f"Unknown quantization mode: {quant_mode}")
        
        if save_dir:
            if hasattr(q, 'cpu'):
                q.cpu().numpy().tofile(os.path.join(save_dir, 'quantized_stream.bin'))
            else:
                q.tofile(os.path.join(save_dir, 'quantized_stream.bin'))
        
        original_bits = 512 * 512 * 3 * 8
        compressed_bits = meta.get('packed_bits', q.numel() * 8)
        compression_rate = (compressed_bits / original_bits) * 100
        print(f"Original bits: {original_bits}, Compressed bits: {compressed_bits}, Rate: {compression_rate:.2f}%")
        
        # 6. Transmit
        start_tx = time.time()
        q_tilde = self.channel.transmit(q, snr_db, meta)
        tx_time = time.time() - start_tx
        print(f"Transmission & Channel time: {tx_time:.4f}s")
        
        # 7. Dequantization
        if quant_mode == '8bit':
            z_tilde = dequantize_uniform_8bit(q_tilde, meta)
        elif quant_mode == '4bit-attn':
            z_tilde = dequantize_attention_4bit(q_tilde, meta, device=q_tilde.device)
        elif quant_mode in ['2bit', '4bit', '6bit']:
            bits = int(quant_mode.replace('bit', ''))
            z_tilde = dequantize_uniform_multi_bit(q_tilde, meta, device=q_tilde.device)
        elif quant_mode in ['adaptive', 'attention', 'channel_aware', 'task_aware']:
            aq = AdaptiveQuantizer(allocation_mode='adaptive')
            z_tilde = aq.dequantize(q_tilde, meta, device=q_tilde.device)
        else:
            raise ValueError(f"Unknown quantization mode: {quant_mode}")
        
        z_tilde = z_tilde.to(self.device)
        
        # 8. Decode
        start_dec = time.time()
        img_recon_tensor = self.decoder.decode(z_tilde)
        dec_time = time.time() - start_dec
        print(f"Decoding time: {dec_time:.4f}s")
        
        # 9. Postprocess
        img_recon = postprocess_image(img_recon_tensor)
        
        # 10. Task evaluation (if enabled)
        task_eval_result = None
        if self.task_evaluator and self.current_task_profile:
            original_embedding = None
            reconstructed_embedding = None
            
            if return_embeddings and self.search_engine:
                original_embedding = self.search_engine.feature_extractor.extract_image_embedding(img).embedding
                reconstructed_embedding = self.search_engine.feature_extractor.extract_image_embedding(img_recon).embedding
            
            original_detections = detections
            reconstructed_detections = None
            if return_detections and self.object_detector:
                reconstructed_detections = self.object_detector.detect(img_recon)
            
            task_eval_result = self.task_evaluator.evaluate(
                original_image=img,
                reconstructed_image=img_recon,
                task_profile=self.current_task_profile,
                original_detections=original_detections,
                reconstructed_detections=reconstructed_detections,
                original_embedding=original_embedding,
                reconstructed_embedding=reconstructed_embedding,
                payload_bits=compressed_bits
            )
        
        # 11. Retrieval (if enabled and we have a reconstruction)
        retrieval_results = None
        if self.enable_retrieval and self.search_engine and return_embeddings:
            recon_embedding = self.search_engine.feature_extractor.extract_image_embedding(img_recon).embedding
            retrieval_results = self.search_engine.search_text(
                "",  # Would use actual query
                top_k=5
            )
        
        stats = {
            'enc_time': enc_time,
            'tx_time': tx_time,
            'dec_time': dec_time,
            'compression_rate': compression_rate,
            'compressed_bits': compressed_bits,
            'payload_info': calculate_payload_bits(meta) if 'region_metadata' in meta or meta.get('mode', '').startswith('bit') else {}
        }
        
        result = {
            'image': img_recon,
            'stats': stats,
            'detections': detections if return_detections else None,
            'latent_rois': latent_rois if return_detections else None,
            'task_eval': task_eval_result.to_dict() if task_eval_result else None,
            'retrieval_results': [r.to_dict() for r in retrieval_results] if retrieval_results else None
        }
        
        return img_recon, result
    
    def _create_detection_importance_map(self, z_shape, latent_rois):
        """Create importance map from latent ROIs."""
        B, C, H, W = z_shape
        importance_map = torch.zeros((B, 1, H, W), device=self.device)
        
        for roi in latent_rois:
            x1, y1, x2, y2 = roi.latent_bbox
            conf = roi.detection.confidence
            importance_map[0, 0, y1:y2, x1:x2] = max(
                importance_map[0, 0, y1:y2, x1:x2].max().item() if (y2 > y1 and x2 > x1) else 0,
                conf
            )
        
        return importance_map
    
    def _get_class_names_per_location(self, detections, z_shape):
        """Map detections to spatial locations for class priority."""
        if not detections:
            return None
        
        B, C, H, W = z_shape
        class_names = [[[] for _ in range(W)] for _ in range(H)]
        
        for det in detections:
            latent_rois = self.object_detector.get_latent_rois([det])
            for roi in latent_rois:
                x1, y1, x2, y2 = roi.latent_bbox
                for y in range(y1, min(y2, H)):
                    for x in range(x1, min(x2, W)):
                        class_names[y][x].append(det.class_name)
        
        # Convert to list of lists format expected by TaskProfile
        result = []
        for y in range(H):
            row = []
            for x in range(W):
                names = class_names[y][x]
                row.append(names[0] if names else 'background')
            result.append(row)
        
        return result
    
    def add_to_retrieval(
        self,
        image,
        record_id: str = None,
        object_class: str = None,
        confidence: float = None,
        snr_db: float = None,
        quantization_mode: str = None,
        source: str = 'vcsc'
    ):
        """Add an image to the retrieval database."""
        if not self.enable_retrieval or not self.search_engine:
            return False
        
        return self.search_engine.add_image_record(
            image=image,
            record_id=record_id,
            object_class=object_class,
            confidence=confidence,
            channel_snr_db=snr_db,
            channel_type=self.channel.channel_type,
            quantization_mode=quantization_mode,
            source=source
        )
    
    def search_retrieval(self, query_text: str, top_k: int = 10, **filters):
        """Search the retrieval database."""
        if not self.enable_retrieval or not self.search_engine:
            return []
        return self.search_engine.search_text(query_text, top_k=top_k, **filters)
    
    def get_retrieval_stats(self):
        """Get retrieval database statistics."""
        if not self.enable_retrieval or not self.search_engine:
            return {}
        return self.search_engine.get_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input image path")
    parser.add_argument('--output', type=str, default="results/reconstructed.png", help="Output image path")
    parser.add_argument('--quantization', type=str, 
        choices=['8bit', '4bit-attn', '2bit', '4bit', '6bit', 'adaptive', 'attention', 'channel_aware', 'task_aware'],
        default='8bit')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--channel', type=str, choices=['awgn', 'rayleigh'], default='awgn')
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--no-turbo', action='store_true')
    parser.add_argument('--task', type=str, choices=['object_identification', 'object_retrieval', 'scene_understanding'], default=None)
    parser.add_argument('--detect', action='store_true', help="Enable object detection")
    parser.add_argument('--retrieval', action='store_true', help="Enable retrieval")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    pipeline = VCSCPipeline(
        use_turbo=not args.no_turbo,
        channel_type=args.channel,
        enable_object_detection=args.detect,
        enable_retrieval=args.retrieval
    )
    
    if args.task:
        pipeline.set_task_profile(TaskType(args.task))
    
    recon_img, result = pipeline.process_image(
        args.input,
        args.quantization,
        args.snr,
        args.threshold,
        return_detections=args.detect,
        return_embeddings=args.retrieval
    )
    
    save_image(recon_img, args.output)
    print(f"Saved reconstructed image to {args.output}")
    
    if result.get('task_eval'):
        print(f"Task Evaluation: {json.dumps(result['task_eval'], indent=2)}")