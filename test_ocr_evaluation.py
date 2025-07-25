#!/usr/bin/env python3
"""
Test script for OCR evaluation framework.

This script demonstrates how to use the OCR evaluation framework
to compare different OCR models on sample data.
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.append('src')

from src.ocr_models import TesseractOCR, PaddleOCR, EasyOCR
from src.evaluation import OCREvaluator, calculate_all_metrics


def create_test_images():
    """Create sample test images for evaluation."""
    print("Creating test images...")
    
    # Test texts
    test_texts = [
        "Hello World! This is a sample text for OCR evaluation.",
        "The quick brown fox jumps over the lazy dog.",
        "1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "Mixed case: Hello WORLD, this is a TEST.",
        "Numbers and text: 42 is the answer to life, the universe, and everything.",
        "Longer text with multiple sentences. This is the second sentence. And here is the third sentence.",
        "Technical terms: Machine Learning, Artificial Intelligence, Deep Learning",
        "Punctuation test: Hello, world! How are you? I'm doing well.",
        "Empty space test:   Multiple   spaces   between   words   "
    ]
    
    # Create images directory
    os.makedirs('data/images', exist_ok=True)
    
    # Create test images
    for i, text in enumerate(test_texts, 1):
        # Create image
        image = Image.new('RGB', (600, 100), color='white')
        draw = ImageDraw.Draw(image)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        # Draw text
        draw.text((10, 30), text, fill='black', font=font)
        
        # Save image
        filename = f"sample_text_{i}.png"
        image.save(f"data/images/{filename}")
        print(f"Created: {filename}")
    
    print("Test images created successfully!")


def initialize_models():
    """Initialize OCR models."""
    print("Initializing OCR models...")
    
    models = {}
    
    # Initialize Tesseract
    try:
        models['Tesseract'] = TesseractOCR()
        print("✓ Tesseract initialized")
    except Exception as e:
        print(f"✗ Tesseract initialization failed: {e}")
    
    # Initialize PaddleOCR
    try:
        models['PaddleOCR'] = PaddleOCR()
        print("✓ PaddleOCR initialized")
    except Exception as e:
        print(f"✗ PaddleOCR initialization failed: {e}")
    
    # Initialize EasyOCR
    try:
        models['EasyOCR'] = EasyOCR()
        print("✓ EasyOCR initialized")
    except Exception as e:
        print(f"✗ EasyOCR initialization failed: {e}")
    
    return models


def run_evaluation():
    """Run the OCR evaluation."""
    print("\n" + "="*60)
    print("OCR Model Evaluation")
    print("="*60)
    
    # Initialize models
    models = initialize_models()
    
    if not models:
        print("Error: No models could be initialized!")
        return
    
    # Create evaluator
    evaluator = OCREvaluator(models)
    
    # Load ground truth
    ground_truth_file = "data/ground_truth/sample_ground_truth.json"
    if not os.path.exists(ground_truth_file):
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return
    
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    # Evaluate each image
    all_results = {model_name: [] for model_name in models.keys()}
    
    images_dir = "data/images"
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nEvaluating {len(image_files)} images with {len(models)} models...")
    
    for image_file in image_files:
        print(f"\nProcessing: {image_file}")
        
        # Get ground truth
        if image_file not in ground_truth_data:
            print(f"Warning: No ground truth for {image_file}")
            continue
        
        ground_truth = ground_truth_data[image_file]
        image_path = os.path.join(images_dir, image_file)
        
        # Evaluate all models on this image
        results = evaluator.evaluate_single_image(image_path, ground_truth)
        
        # Store results
        for model_name, result in results.items():
            all_results[model_name].append(result)
    
    # Aggregate results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for model_name, results in all_results.items():
        if results:
            # Calculate aggregated metrics
            cer_values = [r['cer'] for r in results]
            wer_values = [r['wer'] for r in results]
            time_values = [r['processing_time'] for r in results]
            accuracy_values = [r['accuracy'] for r in results]
            
            print(f"\n{model_name}:")
            print(f"  Total Images: {len(results)}")
            print(f"  Success Rate: {sum(1 for r in results if r['success'])}/{len(results)}")
            print(f"  Average CER: {np.mean(cer_values):.4f} ± {np.std(cer_values):.4f}")
            print(f"  Average WER: {np.mean(wer_values):.4f} ± {np.std(wer_values):.4f}")
            print(f"  Average Accuracy: {np.mean(accuracy_values):.4f} ± {np.std(accuracy_values):.4f}")
            print(f"  Average Processing Time: {np.mean(time_values):.4f}s")
    
    # Find best model
    print("\n" + "="*60)
    print("BEST MODEL ANALYSIS")
    print("="*60)
    
    best_models = {}
    for model_name, results in all_results.items():
        if results:
            cer_values = [r['cer'] for r in results]
            wer_values = [r['wer'] for r in results]
            time_values = [r['processing_time'] for r in results]
            
            best_models[model_name] = {
                'avg_cer': np.mean(cer_values),
                'avg_wer': np.mean(wer_values),
                'avg_time': np.mean(time_values)
            }
    
    if best_models:
        # Best by CER
        best_cer = min(best_models.items(), key=lambda x: x[1]['avg_cer'])
        print(f"Best by CER: {best_cer[0]} (CER: {best_cer[1]['avg_cer']:.4f})")
        
        # Best by WER
        best_wer = min(best_models.items(), key=lambda x: x[1]['avg_wer'])
        print(f"Best by WER: {best_wer[0]} (WER: {best_wer[1]['avg_wer']:.4f})")
        
        # Fastest
        fastest = min(best_models.items(), key=lambda x: x[1]['avg_time'])
        print(f"Fastest: {fastest[0]} (Time: {fastest[1]['avg_time']:.4f}s)")
    
    print("\nEvaluation completed!")


def main():
    """Main function."""
    print("OCR Model Evaluation Framework")
    print("="*40)
    
    # Check if test images exist
    if not os.path.exists("data/images") or not os.listdir("data/images"):
        print("Creating test images...")
        create_test_images()
    
    # Run evaluation
    run_evaluation()


if __name__ == "__main__":
    main() 