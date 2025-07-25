# ICDAR 2013 Dataset

## Overview
The ICDAR 2013 dataset is a benchmark dataset for text recognition in natural images. It contains printed text images from various sources including signs, advertisements, and documents.

## Dataset Information
- **Type**: Printed text
- **Language**: English
- **Format**: Images with XML ground truth annotations
- **Source**: ICDAR 2013 Robust Reading Competition

## Directory Structure
```
icdar2013/
├── images/          # Image files (.png, .jpg, etc.)
├── ground_truth/    # XML annotation files
└── results/         # Evaluation results
```

## Usage
1. Download the ICDAR 2013 dataset
2. Place images in the `images/` directory
3. Place corresponding XML annotations in the `ground_truth/` directory
4. Run evaluation: `make evaluate-dataset icdar2013`

## Ground Truth Format
Single txt file (`gt.txt`) containing comma-separated values with format:
```
filename.png, "ground_truth_text"
```

## Citation
If you use this dataset, please cite:
```
@inproceedings{karatzas2013icdar,
  title={ICDAR 2013 robust reading competition},
  author={Karatzas, Dimosthenis and Shafait, Faisal and Uchida, Seiichi and Iwamura, Masakazu and i Bigorda, Lluis Gomez and Mestre, Sergi Robles and Mas, Joan and Mota, David Fernandez and Almazan, Jon Almazan and de las Heras, Lluis Pere},
  booktitle={2013 12th International Conference on Document Analysis and Recognition},
  pages={1484--1493},
  year={2013},
  organization={IEEE}
}
``` 