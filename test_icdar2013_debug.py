#!/usr/bin/env python3
"""
Debug script for ICDAR 2013 dataset evaluation.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_icdar2013_loading():
    """Test ICDAR 2013 dataset loading."""
    print("Testing ICDAR 2013 dataset loading...")
    
    # Check if files exist
    images_dir = "data/datasets/icdar2013/images"
    gt_file = "data/datasets/icdar2013/ground_truth/gt.txt"
    
    print(f"Images directory: {images_dir}")
    print(f"Ground truth file: {gt_file}")
    
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} image files")
        if image_files:
            print(f"Sample images: {image_files[:5]}")
    else:
        print("❌ Images directory not found")
    
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        print(f"Found {len(lines)} ground truth entries")
        if lines:
            print(f"Sample ground truth: {lines[:3]}")
    else:
        print("❌ Ground truth file not found")
    
    # Test the ground truth loading function
    try:
        from src.evaluation.evaluator import OCREvaluator
        
        # Create a dummy evaluator to test the loading function
        evaluator = OCREvaluator({})
        
        # Test the ICDAR 2013 ground truth loading
        ground_truth_map = evaluator._load_icdar2013_ground_truth(gt_file)
        print(f"Loaded {len(ground_truth_map)} ground truth entries")
        
        if ground_truth_map:
            sample_items = list(ground_truth_map.items())[:3]
            print("Sample ground truth mapping:")
            for filename, text in sample_items:
                print(f"  {filename}: {text}")
        
        # Check if images match ground truth
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            matched = 0
            for img_file in image_files:
                if img_file in ground_truth_map:
                    matched += 1
            
            print(f"Images matched with ground truth: {matched}/{len(image_files)}")
            
    except Exception as e:
        print(f"❌ Error testing ground truth loading: {e}")
        import traceback
        traceback.print_exc()

def test_ocr_models():
    """Test OCR model initialization."""
    print("\nTesting OCR model initialization...")
    
    try:
        from src.ocr_models import TesseractOCR, PaddleOCR, EasyOCR
        
        models = {}
        
        # Test Tesseract
        try:
            tesseract = TesseractOCR()
            models['tesseract'] = tesseract
            print("✅ Tesseract initialized successfully")
        except Exception as e:
            print(f"❌ Tesseract initialization failed: {e}")
        
        # Test PaddleOCR
        try:
            paddleocr = PaddleOCR()
            models['paddleocr'] = paddleocr
            print("✅ PaddleOCR initialized successfully")
        except Exception as e:
            print(f"❌ PaddleOCR initialization failed: {e}")
        
        # Test EasyOCR
        try:
            easyocr = EasyOCR()
            models['easyocr'] = easyocr
            print("✅ EasyOCR initialized successfully")
        except Exception as e:
            print(f"❌ EasyOCR initialization failed: {e}")
        
        print(f"Successfully initialized {len(models)} models")
        
        # Test single image processing
        if models and os.path.exists("data/datasets/icdar2013/images/word_1.png"):
            test_image = "data/datasets/icdar2013/images/word_1.png"
            print(f"\nTesting single image processing with {test_image}")
            
            # Load image as numpy array
            import cv2
            image = cv2.imread(test_image)
            if image is None:
                print("❌ Failed to load test image")
                return models
            
            for model_name, model in models.items():
                try:
                    result = model.recognize_text(image)
                    print(f"✅ {model_name}: '{result}'")
                except Exception as e:
                    print(f"❌ {model_name} failed: {e}")
        
        return models
        
    except Exception as e:
        print(f"❌ Error testing OCR models: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    test_icdar2013_loading()
    test_ocr_models() 