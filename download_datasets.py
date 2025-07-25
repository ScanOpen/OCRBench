#!/usr/bin/env python3
"""
Dataset Download Helper Script

This script helps you download and set up the datasets used in the OCR evaluation framework.
"""

import os
import sys
import json
import requests
from pathlib import Path
from faker import Faker


def create_directory_structure():
    """Create the dataset directory structure."""
    datasets = ['icdar2013', 'iam', 'multilingual']
    
    for dataset in datasets:
        for subdir in ['images', 'ground_truth', 'results']:
            Path(f'data/datasets/{dataset}/{subdir}').mkdir(parents=True, exist_ok=True)
    
    print("✅ Dataset directory structure created")


def download_icdar2013_info():
    """Provide information about downloading ICDAR 2013."""
    print("\n📥 ICDAR 2013 Dataset Download Instructions:")
    print("=" * 50)
    print("1. Visit: http://dag.cvc.uab.es/icdar2013competition/")
    print("2. Download 'Challenge 2: Focused Scene Text'")
    print("3. Extract files to:")
    print("   - Images: data/datasets/icdar2013/images/")
    print("   - Annotations: data/datasets/icdar2013/ground_truth/")
    print("\nAlternative sources:")
    print("- Roboflow: https://universe.roboflow.com/roboflow-universe-projects/icdar-2013-text-localization")
    print("- Kaggle: Search for 'ICDAR 2013' datasets")


def download_iam_info():
    """Provide information about downloading IAM dataset."""
    print("\n📥 IAM Handwriting Database Download Instructions:")
    print("=" * 50)
    print("1. Visit: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
    print("2. Register for an account")
    print("3. Accept terms and request access")
    print("4. Download the IAM Handwriting Database")
    print("5. Extract files to:")
    print("   - Images: data/datasets/iam/images/")
    print("   - Annotations: data/datasets/iam/ground_truth/")
    print("\nAlternative sources:")
    print("- Kaggle: Search for 'IAM Handwriting Database'")


def generate_synthetic_multilingual():
    """Generate synthetic multilingual dataset."""
    print("\n🔧 Generating Synthetic Multilingual Dataset...")
    
    # Create Faker instances for different languages
    fake_en = Faker('en_US')
    fake_fr = Faker('fr_FR')
    fake_de = Faker('de_DE')
    fake_es = Faker('es_ES')
    
    # Generate sample data
    data = []
    languages = [
        ('en', fake_en, 'English'),
        ('fr', fake_fr, 'French'),
        ('de', fake_de, 'German'),
        ('es', fake_es, 'Spanish'),
        ('zh', None, 'Chinese')
    ]
    
    for i in range(50):  # Generate 50 samples
        lang_code, fake, lang_name = languages[i % len(languages)]
        
        if lang_code == 'zh':
            # Chinese text examples
            chinese_texts = [
                "这是一个中文文本示例。",
                "人工智能技术正在快速发展。",
                "机器学习是计算机科学的重要分支。",
                "深度学习在图像识别方面表现出色。",
                "自然语言处理技术越来越成熟。"
            ]
            text = chinese_texts[i % len(chinese_texts)]
        else:
            # Generate text in other languages
            text = fake.text(max_nb_chars=200)
        
        sample_data = {
            'id': f'sample_{i:03d}',
            'text': text,
            'language': lang_code,
            'language_name': lang_name,
            'image_path': f'images/sample_{i:03d}.png',
            'bbox': [10, 10, 500, 50],  # Example bounding box
            'confidence': 0.95
        }
        data.append(sample_data)
    
    # Save ground truth
    gt_path = 'data/datasets/multilingual/ground_truth/sample_ground_truth.json'
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Synthetic multilingual dataset created: {gt_path}")
    print(f"   - {len(data)} samples generated")
    print(f"   - Languages: English, French, German, Spanish, Chinese")
    print(f"   - Ground truth saved to: {gt_path}")
    print("\n💡 Note: You'll need to create corresponding images for these samples")


def check_dataset_status():
    """Check the status of downloaded datasets."""
    print("\n🔍 Dataset Status Check:")
    print("=" * 30)
    
    datasets = {
        'ICDAR 2013': 'data/datasets/icdar2013',
        'IAM Handwriting': 'data/datasets/iam',
        'Multilingual': 'data/datasets/multilingual'
    }
    
    for name, path in datasets.items():
        images_dir = os.path.join(path, 'images')
        gt_dir = os.path.join(path, 'ground_truth')
        
        images_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(images_dir) else 0
        gt_count = len([f for f in os.listdir(gt_dir) if f.lower().endswith(('.txt', '.xml', '.json'))]) if os.path.exists(gt_dir) else 0
        
        print(f"\n{name}:")
        print(f"  📁 Images: {images_count} files")
        print(f"  📄 Ground Truth: {gt_count} files")
        
        if images_count > 0 and gt_count > 0:
            print(f"  ✅ Ready for evaluation")
        elif images_count == 0 and gt_count == 0:
            print(f"  ⚠️  Not downloaded")
        else:
            print(f"  ⚠️  Partially downloaded")


def main():
    """Main function."""
    print("🔍 OCR Dataset Download Helper")
    print("=" * 40)
    
    # Create directory structure
    create_directory_structure()
    
    # Provide download instructions
    download_icdar2013_info()
    download_iam_info()
    
    # Generate synthetic multilingual dataset
    try:
        generate_synthetic_multilingual()
    except ImportError:
        print("\n⚠️  Faker not installed. Install with: pipenv install faker")
        print("   Or manually create the multilingual dataset structure.")
    
    # Check current status
    check_dataset_status()
    
    print("\n🎯 Next Steps:")
    print("1. Download the datasets using the instructions above")
    print("2. Place files in the appropriate directories")
    print("3. Run: make evaluate-all-datasets")
    print("4. Check results in data/datasets/*/results/")
    
    print("\n💡 Quick Commands:")
    print("  make evaluate-icdar2013  # Evaluate ICDAR 2013")
    print("  make evaluate-iam        # Evaluate IAM")
    print("  make evaluate-multilingual # Evaluate multilingual")
    print("  make evaluate-all-datasets # Evaluate all datasets")


if __name__ == "__main__":
    main() 