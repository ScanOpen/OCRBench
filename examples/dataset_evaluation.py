#!/usr/bin/env python3
"""
Example script demonstrating dataset evaluation with the three supported datasets.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.evaluation import OCREvaluator
from src.ocr_models import TesseractOCR, PaddleOCR, EasyOCR


def main():
    """Demonstrate dataset evaluation with all three datasets."""
    
    print("🔍 OCR Dataset Evaluation Example")
    print("=" * 50)
    
    # Initialize models
    print("\n📦 Initializing OCR models...")
    models = {}
    
    try:
        models['Tesseract'] = TesseractOCR()
        print("✅ Tesseract initialized")
    except Exception as e:
        print(f"❌ Tesseract initialization failed: {e}")
    
    try:
        models['PaddleOCR'] = PaddleOCR()
        print("✅ PaddleOCR initialized")
    except Exception as e:
        print(f"❌ PaddleOCR initialization failed: {e}")
    
    try:
        models['EasyOCR'] = EasyOCR()
        print("✅ EasyOCR initialized")
    except Exception as e:
        print(f"❌ EasyOCR initialization failed: {e}")
    
    if not models:
        print("❌ No models could be initialized. Please check your installation.")
        return
    
    # Create evaluator
    evaluator = OCREvaluator(models)
    
    # Available datasets
    datasets = {
        'icdar2013': {
            'name': 'ICDAR 2013',
            'description': 'Printed text dataset',
            'path': 'data/datasets/icdar2013'
        },
        'iam': {
            'name': 'IAM Handwriting',
            'description': 'Handwritten text dataset',
            'path': 'data/datasets/iam'
        },
        'multilingual': {
            'name': 'Multilingual',
            'description': 'Mixed language dataset',
            'path': 'data/datasets/multilingual'
        }
    }
    
    print(f"\n📊 Available datasets:")
    for key, dataset in datasets.items():
        print(f"  - {key}: {dataset['name']} ({dataset['description']})")
    
    # Check dataset availability
    print(f"\n🔍 Checking dataset availability...")
    for dataset_key, dataset_info in datasets.items():
        images_dir = os.path.join(dataset_info['path'], 'images')
        gt_dir = os.path.join(dataset_info['path'], 'ground_truth')
        
        if os.path.exists(images_dir) and os.listdir(images_dir):
            print(f"✅ {dataset_info['name']}: Images found")
        else:
            print(f"⚠️  {dataset_info['name']}: No images found (place images in {images_dir})")
        
        if os.path.exists(gt_dir) and os.listdir(gt_dir):
            print(f"✅ {dataset_info['name']}: Ground truth found")
        else:
            print(f"⚠️  {dataset_info['name']}: No ground truth found (place annotations in {gt_dir})")
    
    print(f"\n💡 To evaluate datasets:")
    print(f"  1. Download the datasets and place them in the appropriate directories")
    print(f"  2. Run: make evaluate-icdar2013")
    print(f"  3. Run: make evaluate-iam")
    print(f"  4. Run: make evaluate-multilingual")
    print(f"  5. Or run all: make evaluate-all-datasets")
    
    print(f"\n📋 Dataset Information:")
    print(f"  - ICDAR 2013: Printed text in natural images (XML annotations)")
    print(f"  - IAM: Handwritten English text (TXT annotations)")
    print(f"  - Multilingual: Mixed languages (JSON annotations)")
    
    print(f"\n🎯 Example evaluation commands:")
    print(f"  python -m src.cli.main evaluate-dataset --dataset icdar2013")
    print(f"  python -m src.cli.main evaluate-dataset --dataset iam --models tesseract easyocr")
    print(f"  python -m src.cli.main evaluate-dataset --dataset multilingual")


if __name__ == "__main__":
    main() 