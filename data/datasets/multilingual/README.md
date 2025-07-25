# Synthetic Mixed-Language Dataset

## Overview
A synthetic dataset containing text in multiple languages including English, French, German, Spanish, and Chinese. This dataset is designed to test OCR performance across different languages and scripts.

## Dataset Information
- **Type**: Mixed (printed and synthetic)
- **Languages**: English, French, German, Spanish, Chinese
- **Format**: Images with JSON ground truth annotations
- **Source**: Synthetic generation

## Directory Structure
```
multilingual/
├── images/          # Image files (.png, .jpg, etc.)
├── ground_truth/    # JSON annotation files
└── results/         # Evaluation results
```

## Usage
1. Generate or download the synthetic multilingual dataset
2. Place images in the `images/` directory
3. Place corresponding JSON annotations in the `ground_truth/` directory
4. Run evaluation: `make evaluate-dataset multilingual`

## Ground Truth Format
JSON files containing:
- Text annotations
- Language labels
- Bounding boxes
- Confidence scores

## Language Support
- **English**: Latin script
- **French**: Latin script with accents
- **German**: Latin script with umlauts
- **Spanish**: Latin script with special characters
- **Chinese**: Simplified Chinese characters

## Citation
This is a synthetic dataset. If you use this framework for multilingual evaluation, please cite the original datasets used for training and the OCR models being evaluated. 