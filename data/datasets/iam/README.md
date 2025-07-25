# IAM Handwriting Database

## Overview
The IAM Handwriting Database is a large dataset of handwritten English text. It contains forms of handwritten English text which can be used to train and test handwritten text recognition systems.

## Dataset Information
- **Type**: Handwritten text
- **Language**: English
- **Format**: Images with text line annotations
- **Source**: Institute of Computer Science and Applied Mathematics, University of Bern

## Directory Structure
```
iam/
├── images/          # Image files (.png, .jpg, etc.)
├── ground_truth/    # Text line annotations (.txt files)
└── results/         # Evaluation results
```

## Usage
1. Download the IAM Handwriting Database
2. Place images in the `images/` directory
3. Place corresponding text annotations in the `ground_truth/` directory
4. Run evaluation: `make evaluate-dataset iam`

## Ground Truth Format
Text files containing line-by-line transcriptions of the handwritten text in the images.

## Preprocessing
This dataset benefits from preprocessing:
- Denoising: Enabled by default
- Contrast enhancement: Enabled by default
- These settings are configured in the evaluation config

## Citation
If you use this dataset, please cite:
```
@article{marti2002iam,
  title={The IAM-database: an English sentence database for offline handwriting recognition},
  author={Marti, U-V and Bunke, Horst},
  journal={International Journal on Document Analysis and Recognition},
  volume={5},
  number={1},
  pages={39--46},
  year={2002},
  publisher={Springer}
}
``` 