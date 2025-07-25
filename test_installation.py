#!/usr/bin/env python3
"""
Test script to verify OCR evaluation framework installation.

This script checks if all required components can be imported
and basic functionality works.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    # Add src to path
    sys.path.append('src')
    
    try:
        # Test OCR models
        from src.ocr_models import TesseractOCR, PaddleOCR, EasyOCR
        print("✓ OCR models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OCR models: {e}")
        return False
    
    try:
        # Test evaluation
        from src.evaluation import OCREvaluator
        print("✓ Evaluation module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation module: {e}")
        return False
    
    try:
        # Test metrics
        from src.evaluation.metrics import calculate_cer, calculate_wer
        print("✓ Metrics module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import metrics module: {e}")
        return False
    
    try:
        # Test visualization
        from src.visualization import plot_accuracy_comparison
        print("✓ Visualization module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import visualization module: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of the framework."""
    print("\nTesting basic functionality...")
    
    try:
        # Test metrics calculation
        from src.evaluation.metrics import calculate_cer, calculate_wer
        
        # Test with simple examples
        cer = calculate_cer("hello world", "hello world")
        if cer == 0.0:
            print("✓ CER calculation working correctly")
        else:
            print(f"✗ CER calculation failed: expected 0.0, got {cer}")
            return False
        
        wer = calculate_wer("hello world", "hello world")
        if wer == 0.0:
            print("✓ WER calculation working correctly")
        else:
            print(f"✗ WER calculation failed: expected 0.0, got {wer}")
            return False
        
        # Test with different text
        cer = calculate_cer("hello", "world")
        if cer > 0.0:
            print("✓ CER calculation working correctly with different text")
        else:
            print(f"✗ CER calculation failed with different text: got {cer}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_model_initialization():
    """Test model initialization (without actually loading models)."""
    print("\nTesting model initialization...")
    
    try:
        from src.ocr_models import TesseractOCR, PaddleOCR, EasyOCR
        
        # Test Tesseract initialization (this will fail if Tesseract is not installed)
        try:
            tesseract = TesseractOCR()
            print("✓ TesseractOCR class can be instantiated")
        except Exception as e:
            print(f"⚠ TesseractOCR initialization failed (expected if Tesseract not installed): {e}")
        
        # Test PaddleOCR initialization
        try:
            paddleocr = PaddleOCR()
            print("✓ PaddleOCR class can be instantiated")
        except Exception as e:
            print(f"⚠ PaddleOCR initialization failed (expected if PaddleOCR not installed): {e}")
        
        # Test EasyOCR initialization
        try:
            easyocr = EasyOCR()
            print("✓ EasyOCR class can be instantiated")
        except Exception as e:
            print(f"⚠ EasyOCR initialization failed (expected if EasyOCR not installed): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization test failed: {e}")
        return False


def test_directory_structure():
    """Test if the project directory structure is correct."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'src',
        'src/ocr_models',
        'src/evaluation',
        'src/visualization',
        'src/cli',
        'data',
        'data/images',
        'data/ground_truth',
        'data/results',
        'notebooks',
        'tests',
        'config',
        'docs'
    ]
    
    required_files = [
        'README.md',
        'Pipfile',
        'setup.py',
        'src/__init__.py',
        'src/ocr_models/__init__.py',
        'src/evaluation/__init__.py',
        'config/evaluation_config.yaml'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ File missing: {file_path}")
            all_good = False
    
    return all_good


def main():
    """Main test function."""
    print("OCR Evaluation Framework - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Model Initialization", test_model_initialization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The OCR evaluation framework is ready to use.")
        print("\nNext steps:")
        print("1. Install required OCR engines (Tesseract, PaddleOCR, EasyOCR)")
        print("2. Run: make evaluate")
        print("3. Explore the notebooks/ directory")
        print("4. Check the README.md for detailed usage instructions")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the installation.")
        print("Make sure all required dependencies are installed:")
        print("pipenv install")


if __name__ == "__main__":
    main() 