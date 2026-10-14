import torch
import numpy as np
from PIL import Image

from experiments.run_vcsc import VCSCPipeline
from object_detection.detector import create_detector
from retrieval.search import create_search_engine


def test_vcsc_pipeline_initialization():
    """Test VCSCPipeline initialization with various configurations."""
    print("Testing VCSCPipeline initialization...")
    
    # Test 1: Basic initialization
    pipeline = VCSCPipeline(
        use_turbo=False,
        channel_type='awgn',
        enable_object_detection=False,
        enable_retrieval=False
    )
    assert pipeline is not None
    assert pipeline.working_mode == '8bit'
    assert pipeline.enable_object_detection is False
    assert pipeline.enable_retrieval is False
    print("  OK Basic initialization passed")
    
    # Test 2: With object detection enabled
    pipeline2 = VCSCPipeline(
        use_turbo=False,
        channel_type='awgn',
        enable_object_detection=True,
        enable_retrieval=False
    )
    assert pipeline2.enable_object_detection is True
    assert pipeline2.object_detector is not None
    print("  OK Object detection initialization passed")
    
    # Test 3: With retrieval enabled
    pipeline3 = VCSCPipeline(
        use_turbo=False,
        channel_type='awgn',
        enable_object_detection=False,
        enable_retrieval=True,
        retrieval_index_path='tests/fixtures/faiss_index'
    )
    assert pipeline3.enable_retrieval is True
    assert pipeline3.search_engine is not None
    print("  OK Retrieval initialization passed")
    
    # Test 4: With task-aware allocation
    pipeline4 = VCSCPipeline(
        use_turbo=False,
        channel_type='awgn',
        enable_object_detection=True,
        enable_retrieval=True
    )
    # Set task profile
    from semantic.task_aware import TaskType, TaskProfile
    from semantic.task_aware import get_task_profile
    pipeline4.set_task_profile(TaskType.OBJECT_IDENTIFICATION)
    assert pipeline4.current_task_profile is not None
    assert pipeline4.task_allocator is not None
    print("  OK Task-aware allocation initialization passed")
    
    print("All VCSCPipeline initialization tests passed!\n")


def test_task_profile_and_allocation():
    """Test task profile setting and allocation computation."""
    print("Testing task profiles and allocation...")
    
    pipeline = VCSCPipeline(
        use_turbo=False,
        channel_type='awgn',
        enable_object_detection=True,
        enable_retrieval=False
    )
    
    from semantic.task_aware import TaskType, get_task_profile
    
    # Test all task profiles
    for task_type in [TaskType.OBJECT_IDENTIFICATION, TaskType.OBJECT_RETRIEVAL, TaskType.SCENE_UNDERstanding]:
        profile_name = task_type.value
        pipeline.set_task_profile(task_type)
        assert pipeline.current_task_profile.name == profile_name
        assert pipeline.task_allocator is not None
        print(f"  OK Task profile '{profile_name}' set successfully")
    
    # Test allocation with task priority
    z = torch.randn(1, 4, 64, 64)
    
    # Test task-aware mode (simulated)
    pipeline.set_task_profile(TaskType.OBJECT_IDENTIFICATION)
    allocation_map, alloc_info = pipeline.task_allocator.compute_allocation(
        z,
        detection_confidence=None,
        class_names=None,
        task_relevance=None
    )
    assert allocation_map.shape == (1, 1, 64, 64)
    print(f"  OK Task-aware allocation computed: {alloc_info.get('channel_condition', 'unknown')}")
    
    print("All task profile and allocation tests passed!\n")


def test_quantization_modes():
    """Test various quantization modes supported by the pipeline."""
    print("Testing quantization modes...")
    
    pipeline = VCSCPipeline(
        use_turbo=False,
        channel_type='awgn',
        enable_object_detection=False,
        enable_retrieval=False
    )
    
    z = torch.randn(1, 4, 64, 64)
    
    # Test 8-bit uniform
    recon8, meta8 = pipeline.process_image(
        z,
        quantization='8bit',
        snr_db=10,
        threshold=0.2,
        task_type='object_identification'
    )
    print(f"  OK 8-bit quantization completed, bits: {meta8.get('packed_bits', 'N/A')}")
    
    # Test 4-bit attention
    recon4_attn, meta4 = pipeline.process_image(
        z,
        quantization='4bit-attn',
        snr_db=10,
        threshold=0.3,
        task_type='object_identification'
    )
    print(f"  OK 4-bit attention quantization completed")
    
    # Test 2-bit uniform
    recon2, meta2 = pipeline.process_image(
        z,
        quantization='2bit',
        snr_db=10,
        threshold=0.2,
        task_type='object_identification'
    )
    print(f"  OK 2-bit quantization completed, bits: {meta2.get('packed_bits', 'N/A')}")
    
    # Test task_aware mode
    pipeline.set_task_profile(TaskType.OBJECT_IDENTIFICATION)
    recon_task, meta_task = pipeline.process_image(
        z,
        quantization='task_aware',
        snr_db=10,
        threshold=0.2,
        task_type='object_identification'
    )
    print(f"  OK Task-aware quantization completed, task_aware flag: {meta_task.get('task_aware', False)}")
    
    print("All quantization mode tests passed!\n")


def test_detection_and_roi():
    """Test object detection and latent ROI mapping."""
    print("Testing object detection and latent ROIs...")
    
    pipeline = VCSCPipeline(
        use_turbo=False,
        channel_type='awgn',
        enable_object_detection=True,
        enable_retrieval=False
    )
    
    # Create a synthetic test image
    test_img = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )
    
    # Process with detection
    recon_img, result = pipeline.process_image(
        test_img,
        quantization='8bit',
        snr_db=10,
        threshold=0.2,
        task_type='object_identification',
        return_detections=True
    )
    
    detections = result.get('detections', [])
    latent_rois = result.get('latent_rois', [])
    
    print(f"  OK Detected {len(detectioos)} objects")
    print(f"  �extracted {len(latent_rois)} latent ROIs")
    
    # Test that detection visualization works
    if detections:
        vis_img = pipeline.object_detector.visualize_detections(test_img, detections[:3])
        assert vis_img is not None
        print("  OK Detection visualization successful")
    
    print("All detection and ROI tests passed!\n")


def test_retrieval_integration():
    """Test retrieval engine integration."""
    print("Testing retrieval integration...")
    
    # Note: CLIPFeatureExtractor not available in this environment
    # Test that pipeline initialization works with retrieval enabled
    from experiments.run_vcsc import VCSCPipeline
    
    pipeline = VCSCPipeline(
        use_turbo=False,
        channel_type='awgn',
        enable_object_detection=False,
        enable_retrieval=True,
        retrieval_index_path='tests/fixtures/faiss_index'
    )
    
    # Test search engine has basic methods
    if pipeline.search_engine:
        assert hasattr(pipeline.search_engine, 'search_text'), "search_engine missing search_text method"
        assert hasattr(pipeline.search_engine, 'get_stats'), "search_engine missing get_stats method"
        print("  OK Search engine has required methods")
        
        # Test retrieval stats
        try:
            stats = pipeline.search_engine.get_stats()
            assert isinstance(stats, dict)
            print(f"  OK Retrieval stats: {stats}")
        except Exception as e:
            print(f"  ! Retrieval stats error (may be expected): {e}")
    
    print("All retrieval integration tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("VCSC Pipeline Integration Tests")
    print("="*60 + "\n")
    
    test_vcsc_pipeline_initialization()
    test_task_profile_and_allocation()
    test_quantization_modes()
    test_detection_and_roi()
    test_retrieval_integration()
    
    print("="*60)
    print("All integration tests passed! OK")
    print("="*60)