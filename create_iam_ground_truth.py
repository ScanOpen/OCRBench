#!/usr/bin/env python3
"""
Script to create sample ground truth files for IAM dataset images.
This is useful for testing when you have images but no ground truth files.
"""

import os
import glob
from pathlib import Path

def create_sample_ground_truth():
    """Create sample ground truth files for IAM dataset images."""
    
    # IAM dataset paths
    images_dir = "data/datasets/iam/images"
    ground_truth_dir = "data/datasets/iam/ground_truth"
    
    # Sample text snippets for different types of content
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "In the beginning was the Word, and the Word was with God.",
        "To be or not to be, that is the question.",
        "All the world's a stage, and all the men and women merely players.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago—never mind how long precisely.",
        "The only way to do great work is to love what you do.",
        "Success is not final, failure is not fatal.",
        "The future belongs to those who believe in the beauty of their dreams."
    ]
    
    # Find all image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        pattern = os.path.join(images_dir, "**", f"*{ext}")
        image_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(image_files)} image files")
    
    # Create ground truth directory structure
    os.makedirs(ground_truth_dir, exist_ok=True)
    
    # Create sample ground truth files
    created_count = 0
    for i, image_file in enumerate(image_files):
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        
        # Create the same directory structure in ground truth
        rel_path = os.path.relpath(image_file, images_dir)
        gt_dir = os.path.join(ground_truth_dir, os.path.dirname(rel_path))
        os.makedirs(gt_dir, exist_ok=True)
        
        # Create ground truth file
        gt_file = os.path.join(gt_dir, f"{base_name}.txt")
        
        # Use a sample text (cycle through the list)
        sample_text = sample_texts[i % len(sample_texts)]
        
        with open(gt_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        created_count += 1
        
        if created_count % 1000 == 0:
            print(f"Created {created_count} ground truth files...")
    
    print(f"Created {created_count} sample ground truth files")
    print(f"Ground truth files are in: {ground_truth_dir}")
    print("\nNote: These are sample texts for testing purposes only.")
    print("For actual evaluation, you should download the real IAM ground truth files.")

if __name__ == "__main__":
    create_sample_ground_truth() 