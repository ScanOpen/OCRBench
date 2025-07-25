#!/usr/bin/env python3
"""
Debug script for PaddleOCR issues.
"""

import os
import sys
import cv2
import numpy as np

# Add src to path
sys.path.append('src')

def test_paddleocr():
    """Test PaddleOCR initialization and basic functionality."""
    print("Testing PaddleOCR...")
    
    try:
        from src.ocr_models.paddle_ocr import PaddleOCR
        
        # Test initialization
        print("1. Testing PaddleOCR initialization...")
        model = PaddleOCR()
        model.initialize()
        print("✅ PaddleOCR initialized successfully")
        
        # Test with a simple image
        test_image_path = "data/datasets/icdar2013/images/word_1.png"
        if os.path.exists(test_image_path):
            print(f"2. Testing with image: {test_image_path}")
            
            # Load image
            image = cv2.imread(test_image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(f"✅ Image loaded successfully, shape: {image.shape}")
                
                # Test basic recognition
                try:
                    result = model.recognize_text(image)
                    print(f"✅ PaddleOCR result: '{result}'")
                except Exception as e:
                    print(f"❌ PaddleOCR recognition failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Test with confidence
                try:
                    result_with_conf = model.recognize_text_with_confidence(image)
                    print(f"✅ PaddleOCR with confidence: {result_with_conf}")
                except Exception as e:
                    print(f"❌ PaddleOCR confidence recognition failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"❌ Could not load image: {test_image_path}")
        else:
            print(f"❌ Test image not found: {test_image_path}")
            
    except ImportError as e:
        print(f"❌ PaddleOCR import failed: {e}")
    except Exception as e:
        print(f"❌ PaddleOCR test failed: {e}")
        import traceback
        traceback.print_exc()

def test_paddleocr_raw():
    """Test PaddleOCR directly without our wrapper."""
    print("\nTesting PaddleOCR directly...")
    
    try:
        from paddleocr import PaddleOCR
        
        # Initialize directly
        print("1. Testing direct PaddleOCR initialization...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("✅ Direct PaddleOCR initialized successfully")
        
        # Test with image
        test_image_path = "data/datasets/icdar2013/images/word_1.png"
        if os.path.exists(test_image_path):
            print(f"2. Testing direct PaddleOCR with image: {test_image_path}")
            
            # Load image
            image = cv2.imread(test_image_path)
            if image is not None:
                print(f"✅ Image loaded successfully, shape: {image.shape}")
                
                # Test OCR
                try:
                    result = ocr.ocr(image, cls=True)
                    print(f"✅ Direct PaddleOCR result: {result}")
                    
                    # Extract text
                    if result and len(result) > 0:
                        text_parts = []
                        for line in result:
                            if line and len(line) > 0:
                                for detection in line:
                                    if len(detection) >= 2:
                                        text_parts.append(detection[1][0])
                        text = ' '.join(text_parts)
                        print(f"✅ Extracted text: '{text}'")
                    else:
                        print("❌ No text detected")
                        
                except Exception as e:
                    print(f"❌ Direct PaddleOCR failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"❌ Could not load image: {test_image_path}")
        else:
            print(f"❌ Test image not found: {test_image_path}")
            
    except ImportError as e:
        print(f"❌ Direct PaddleOCR import failed: {e}")
    except Exception as e:
        print(f"❌ Direct PaddleOCR test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_paddleocr()
    test_paddleocr_raw() 