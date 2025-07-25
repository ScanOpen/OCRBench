# OCR Model Evaluation Summary - IAM Handwriting Dataset

## Overview

This document presents the evaluation results for OCR models on the IAM Handwriting Dataset, containing 13,353 handwritten text images.

## Overall Statistics

| Model | Total Images | Successful Images | Success Rate | Avg Processing Time | Avg Memory Used |
|-------|-------------|------------------|--------------|-------------------|-----------------|
| **PaddleOCR** | 13,353 | 13,353 | 100.00% | 2.7798s | -2.95MB |
| **EasyOCR** | 13,353 | 13,353 | 100.00% | 1.3332s | 7.34MB |
| **Tesseract** | 13,353 | 13,353 | 100.00% | 0.5020s | -3.92MB |

## Accuracy Metrics

### Character Error Rate (CER)
| Model | CER |
|-------|-----|
| **PaddleOCR** | 0.7294 ± 0.0707 |
| **EasyOCR** | 0.5746 ± 0.1694 |
| **Tesseract** | 0.5405 ± 0.2755 |

### Word Error Rate (WER)
| Model | WER |
|-------|-----|
| **PaddleOCR** | 0.9997 ± 0.0047 |
| **EasyOCR** | 0.9421 ± 0.1126 |
| **Tesseract** | 0.8739 ± 0.1788 |

### Accuracy
| Model | Accuracy |
|-------|----------|
| **PaddleOCR** | 0.0000 ± 0.0000 |
| **EasyOCR** | 0.0001 ± 0.0087 |
| **Tesseract** | 0.0007 ± 0.0274 |

### Precision
| Model | Precision |
|-------|-----------|
| **PaddleOCR** | 0.4991 ± 0.1144 |
| **EasyOCR** | 0.5780 ± 0.1432 |
| **Tesseract** | 0.5537 ± 0.2845 |

### Recall
| Model | Recall |
|-------|--------|
| **PaddleOCR** | 0.3157 ± 0.0456 |
| **EasyOCR** | 0.5089 ± 0.1519 |
| **Tesseract** | 0.5114 ± 0.2792 |

### F1 Score
| Model | F1 Score |
|-------|----------|
| **PaddleOCR** | 0.3797 ± 0.0562 |
| **EasyOCR** | 0.5385 ± 0.1455 |
| **Tesseract** | 0.5267 ± 0.2801 |

## Performance Analysis

### Speed Performance
- **Fastest**: Tesseract (0.5020s average)
- **Medium**: EasyOCR (1.3332s average)
- **Slowest**: PaddleOCR (2.7798s average)

### Memory Usage
- **Lowest Memory**: Tesseract (-3.92MB average)
- **Medium Memory**: PaddleOCR (-2.95MB average)
- **Highest Memory**: EasyOCR (7.34MB average)

### Accuracy Performance
- **Best CER**: Tesseract (0.5405 ± 0.2755)
- **Best WER**: Tesseract (0.8739 ± 0.1788)
- **Best F1 Score**: EasyOCR (0.5385 ± 0.1455)

## Key Findings

1. **All models achieved 100% success rate** on the IAM dataset
2. **Tesseract shows the best character-level accuracy** with the lowest CER
3. **EasyOCR demonstrates the best overall F1 score** for balanced precision and recall
4. **PaddleOCR struggles with handwritten text**, showing poor accuracy metrics
5. **Memory usage varies significantly** between models, with EasyOCR using the most memory

## Recommendations

- **For handwritten text recognition**: Use Tesseract for best character accuracy
- **For balanced performance**: Use EasyOCR for best F1 score
- **For memory-constrained environments**: Use Tesseract or PaddleOCR
- **For speed-critical applications**: Use Tesseract for fastest processing

---

*Generated on: $(date)*  
*Dataset: IAM Handwriting Dataset*  
*Total Images: 13,353* 