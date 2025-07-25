#!/usr/bin/env python3
"""
Test Enhanced Memory Tracking

Demonstrates the enhanced memory tracking capabilities using tracemalloc and memory_profiler.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.memory_tracker import MemoryTracker, create_memory_tracker, get_system_memory_info
from ocr_models.base import BaseOCRModel
from evaluation.evaluator import OCREvaluator
import numpy as np


def test_memory_tracker():
    """Test the MemoryTracker class."""
    print("Testing MemoryTracker...")
    
    # Create memory tracker
    tracker = create_memory_tracker(enable_tracemalloc=True, enable_memory_profiler=True)
    
    # Test basic memory snapshot
    snapshot = tracker.get_memory_snapshot()
    print(f"Initial memory snapshot: {json.dumps(snapshot, indent=2)}")
    
    # Test memory tracking context manager
    with tracker.track_memory_usage("test_operation") as memory_info:
        # Simulate some memory-intensive operation
        large_list = [i for i in range(1000000)]  # ~8MB of memory
        time.sleep(0.1)  # Simulate processing time
        
        print(f"Memory info during operation: {json.dumps(memory_info, indent=2)}")
    
    # Test top memory allocations
    top_allocations = tracker.get_top_memory_allocations(limit=5)
    print(f"Top memory allocations: {json.dumps(top_allocations, indent=2)}")
    
    print("MemoryTracker test completed.\n")


def test_system_memory_info():
    """Test system memory information gathering."""
    print("Testing system memory info...")
    
    memory_info = get_system_memory_info()
    print(f"System memory info: {json.dumps(memory_info, indent=2)}")
    
    print("System memory info test completed.\n")


class MockOCRModel(BaseOCRModel):
    """Mock OCR model for testing memory tracking."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.model_data = None
    
    def initialize(self):
        """Initialize the mock model."""
        # Simulate model loading with memory allocation
        self.model_data = np.random.rand(1000, 1000)  # ~8MB
        self.is_initialized = True
    
    def recognize_text(self, image: np.ndarray) -> str:
        """Mock text recognition."""
        # Simulate OCR processing
        time.sleep(0.01)  # Simulate processing time
        return "Mock OCR result"


def test_ocr_model_memory_tracking():
    """Test memory tracking in OCR models."""
    print("Testing OCR model memory tracking...")
    
    # Create mock OCR model
    model = MockOCRModel("MockOCR")
    
    # Test initialization memory tracking
    print("Initializing model...")
    model.initialize()
    
    # Test single image recognition with memory tracking
    test_image = np.random.rand(100, 300, 3)  # RGB image
    
    print("Performing OCR with memory tracking...")
    result = model.recognize_text_with_metrics(test_image)
    
    print(f"OCR result: {json.dumps(result, indent=2)}")
    
    # Test batch recognition
    print("Testing batch recognition...")
    test_images = [np.random.rand(100, 300, 3) for _ in range(3)]
    batch_results = model.batch_recognize(test_images)
    
    print(f"Batch results: {json.dumps(batch_results, indent=2)}")
    
    # Get memory profile
    memory_profile = model.get_memory_profile()
    print(f"Memory profile: {json.dumps(memory_profile, indent=2)}")
    
    print("OCR model memory tracking test completed.\n")


def test_evaluator_memory_tracking():
    """Test memory tracking in the evaluator."""
    print("Testing evaluator memory tracking...")
    
    # Create mock models
    models = {
        "MockOCR1": MockOCRModel("MockOCR1"),
        "MockOCR2": MockOCRModel("MockOCR2")
    }
    
    # Create evaluator
    evaluator = OCREvaluator(models)
    
    # Test single image evaluation
    test_image_path = "test_image.png"
    ground_truth = "Test ground truth text"
    
    # Create a test image file
    test_image = np.random.rand(100, 300, 3)
    import cv2
    cv2.imwrite(test_image_path, test_image)
    
    print("Evaluating single image...")
    results = evaluator.evaluate_single_image(test_image_path, ground_truth)
    
    print(f"Single image evaluation results: {json.dumps(results, indent=2)}")
    
    # Test memory comparison
    memory_comparison = evaluator.get_memory_comparison()
    print(f"Memory comparison: {json.dumps(memory_comparison, indent=2)}")
    
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    
    print("Evaluator memory tracking test completed.\n")


def test_memory_report_generation():
    """Test memory report generation."""
    print("Testing memory report generation...")
    
    # Create mock models
    models = {
        "MockOCR1": MockOCRModel("MockOCR1"),
        "MockOCR2": MockOCRModel("MockOCR2")
    }
    
    # Create evaluator
    evaluator = OCREvaluator(models)
    
    # Generate memory report
    report_path = "memory_report.json"
    evaluator.generate_memory_report(report_path)
    
    # Read and display the report
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print(f"Memory report: {json.dumps(report, indent=2)}")
    
    # Clean up
    if os.path.exists(report_path):
        os.remove(report_path)
    
    print("Memory report generation test completed.\n")


def main():
    """Run all memory tracking tests."""
    print("Enhanced Memory Tracking Test Suite")
    print("==================================")
    
    try:
        # Test basic memory tracker
        test_memory_tracker()
        
        # Test system memory info
        test_system_memory_info()
        
        # Test OCR model memory tracking
        test_ocr_model_memory_tracking()
        
        # Test evaluator memory tracking
        test_evaluator_memory_tracking()
        
        # Test memory report generation
        test_memory_report_generation()
        
        print("All memory tracking tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 