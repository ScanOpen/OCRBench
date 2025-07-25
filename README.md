# Evaluating the Accuracy of OCR Models: A Comparative Study

This project provides a comprehensive evaluation framework for comparing the accuracy of three popular OCR models: Tesseract, PaddleOCR, and EasyOCR. Built with Python 3.13+ for optimal performance and latest features.

## Project Overview

The goal of this study is to systematically evaluate and compare the performance of different OCR models across various scenarios including:
- Text recognition accuracy
- Processing speed
- Error analysis
- Performance on different text types (printed, handwritten, mixed)
- Robustness to image quality variations

## Models Evaluated

1. **Tesseract** - Open-source OCR engine by Google
2. **PaddleOCR** - Baidu's lightweight OCR toolkit
3. **EasyOCR** - Python library for OCR

## Project Structure

```
Accuracy-of-OCR-Models/
├── README.md                 # This file
├── Pipfile                  # Python dependencies (pipenv)
├── setup.py                 # Project setup
├── data/                    # Dataset directory
│   ├── images/             # Test images
│   ├── ground_truth/       # Ground truth annotations
│   └── results/            # Evaluation results
├── src/                    # Source code
│   ├── __init__.py
│   ├── ocr_models/         # OCR model implementations
│   ├── evaluation/         # Evaluation metrics and tools
│   ├── preprocessing/      # Image preprocessing utilities
│   └── visualization/      # Results visualization
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests
├── config/                 # Configuration files
└── docs/                   # Documentation
```

## Features

- **Multi-model comparison**: Evaluate Tesseract, PaddleOCR, and EasyOCR
- **Comprehensive metrics**: Character-level and word-level accuracy, WER, CER
- **Error analysis**: Detailed analysis of common error types
- **Performance benchmarking**: Speed and memory usage comparison
- **Visualization**: Interactive charts and error visualization
- **Batch processing**: Process large datasets efficiently
- **Configurable evaluation**: Customizable evaluation parameters
- **Python 3.13+ optimized**: Latest Python features for better performance
- **Multiple datasets**: Support for ICDAR 2013, IAM Handwriting, and Multilingual datasets

## Installation

### Prerequisites

- Python 3.13+ (recommended) or Python 3.9+
- pipenv (install with `pip install pipenv`)
- Tesseract OCR engine
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Accuracy-of-OCR-Models
```

2. Install Python dependencies with pipenv:
```bash
pipenv install
```

3. Install Tesseract:
   - **macOS**: `brew install tesseract`
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

4. Install the project in development mode:
```bash
pipenv install -e .
```

5. Activate the virtual environment:
```bash
pipenv shell
```

### Quick Start

For the fastest setup, use the provided quick start script:

```bash
./quick_start.sh
```

This script will:
- Install pipenv if not available
- Install all dependencies
- Run installation tests
- Provide next steps

### Alternative: Using Makefile

For convenience, you can use the provided Makefile:

```bash
# Check Python version compatibility
make check-python

# Install dependencies
make install

# Install development dependencies
make install-dev

# Run tests
make test

# Run evaluation
make evaluate

# Evaluate specific datasets
make evaluate-icdar2013    # ICDAR 2013 printed text dataset
make evaluate-iam          # IAM handwriting dataset
make evaluate-multilingual # Multilingual dataset
make evaluate-all-datasets # Evaluate all datasets

# Run example
make example

# Activate shell
make shell

# See all available commands
make help
```

## Usage

### Quick Start

```python
from src.evaluation import OCREvaluator
from src.ocr_models import TesseractOCR, PaddleOCR, EasyOCR

# Initialize models
models = {
    'tesseract': TesseractOCR(),
    'paddleocr': PaddleOCR(),
    'easyocr': EasyOCR()
}

# Create evaluator
evaluator = OCREvaluator(models)

# Run evaluation
results = evaluator.evaluate_dataset('data/images/', 'data/ground_truth/')

# Generate report
evaluator.generate_report(results, 'data/results/')
```

### Command Line Interface

```bash
# Run evaluation on a single image
pipenv run python -m src.cli evaluate --image path/to/image.jpg --ground-truth "expected text"

# Run evaluation on a dataset
pipenv run python -m src.cli evaluate_dataset --images-dir data/images/ --ground-truth-dir data/ground_truth/

# Generate comparison report
pipenv run python -m src.cli generate_report --results-dir data/results/
```

### Dataset Evaluation

The framework supports three main datasets for comprehensive OCR evaluation:

#### 1. ICDAR 2013 Dataset (Printed Text)
- **Type**: Printed text in natural images (signs, advertisements, documents)
- **Language**: English
- **Format**: Images with XML annotations
- **Usage**: `make evaluate-icdar2013`
- **Citation**: ICDAR 2013 Robust Reading Competition

#### 2. IAM Handwriting Database (Handwritten Text)
- **Type**: Handwritten English text
- **Language**: English
- **Format**: Images with text line annotations
- **Usage**: `make evaluate-iam`
- **Preprocessing**: Denoising and contrast enhancement enabled
- **Citation**: IAM Database for English Handwriting

#### 3. Synthetic Mixed-Language Dataset (Multilingual)
- **Type**: Mixed languages and scripts
- **Languages**: English, French, German, Spanish, Chinese
- **Format**: Images with JSON annotations
- **Usage**: `make evaluate-multilingual`
- **Features**: Multi-script and multi-language evaluation

### Dataset Commands

```bash
# Evaluate specific datasets
make evaluate-icdar2013    # ICDAR 2013 printed text dataset
make evaluate-iam          # IAM handwriting dataset
make evaluate-multilingual # Multilingual dataset
make evaluate-all-datasets # Evaluate all datasets

# Using CLI directly
python -m src.cli.main evaluate-dataset --dataset icdar2013
python -m src.cli.main evaluate-dataset --dataset iam --models tesseract easyocr
python -m src.cli.main evaluate-dataset --dataset multilingual
```

## Evaluation Metrics

- **Character Error Rate (CER)**: Percentage of character-level errors
- **Word Error Rate (WER)**: Percentage of word-level errors
- **Accuracy**: Overall recognition accuracy
- **Precision/Recall**: For specific character classes
- **Processing Time**: Speed comparison between models
- **Memory Usage**: Resource consumption analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{ocr_evaluation_2024,
  title={Evaluating the Accuracy of OCR Models: A Comparative Study},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Accuracy-of-OCR-Models}
}
```

---

# Development Guide

This section is for developers who want to contribute to the OCR Evaluation Framework.

## Setup for Development

### Prerequisites

- Python 3.13+ (recommended) or Python 3.9+
- pipenv
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Accuracy-of-OCR-Models
   ```

2. **Install development dependencies:**
   ```bash
   pipenv install --dev
   pipenv install -e .
   ```

3. **Activate the virtual environment:**
   ```bash
   pipenv shell
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pipenv run pytest

# Run with coverage
pipenv run pytest --cov=src

# Run specific test file
pipenv run pytest tests/test_metrics.py
```

### Code Quality

```bash
# Format code
pipenv run black src/ tests/

# Lint code
pipenv run flake8 src/ tests/

# Type checking
pipenv run mypy src/

# Run all quality checks
make check
```

### Adding Dependencies

```bash
# Add production dependency
pipenv install package-name

# Add development dependency
pipenv install --dev package-name

# Add specific version
pipenv install package-name==1.2.3
```

### Running the Framework

```bash
# Run installation test
pipenv run python test_installation.py

# Run evaluation
pipenv run python test_ocr_evaluation.py

# Run example (shows usage information)
make example

# Run CLI commands
pipenv run python -m src.cli.main evaluate --help
```

## Project Structure

```
src/
├── ocr_models/          # OCR model implementations
│   ├── base.py         # Base class for OCR models
│   ├── tesseract_ocr.py
│   ├── paddle_ocr.py
│   └── easy_ocr.py
├── evaluation/          # Evaluation tools
│   ├── metrics.py      # Accuracy metrics
│   └── evaluator.py    # Main evaluator
├── visualization/       # Plotting and charts
│   └── plots.py
└── cli/                # Command-line interface
    └── main.py
```

## Adding New OCR Models

1. **Create a new model class** in `src/ocr_models/`
2. **Inherit from `BaseOCRModel`**
3. **Implement required methods:**
   - `initialize()`: Load the model
   - `recognize_text()`: Perform OCR
4. **Add to `src/ocr_models/__init__.py`**
5. **Update CLI in `src/cli/main.py`**
6. **Add tests**

Example:
```python
from .base import BaseOCRModel

class NewOCRModel(BaseOCRModel):
    def __init__(self, **kwargs):
        super().__init__("NewOCR", **kwargs)
    
    def initialize(self):
        # Load your model here
        pass
    
    def recognize_text(self, image):
        # Perform OCR here
        return "recognized text"
```

## Adding New Metrics

1. **Add metric function** in `src/evaluation/metrics.py`
2. **Update `calculate_all_metrics()`** to include new metric
3. **Add tests** for the new metric
4. **Update documentation**

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test
pipenv run pytest tests/test_metrics.py::test_calculate_cer

# Run with verbose output
pipenv run pytest -v

# Run with coverage
pipenv run pytest --cov=src --cov-report=html
```

### Writing Tests

- Tests should be in the `tests/` directory
- Use pytest for testing
- Follow naming convention: `test_*.py`
- Test both success and failure cases

Example:
```python
def test_calculate_cer():
    from src.evaluation.metrics import calculate_cer
    
    # Test exact match
    assert calculate_cer("hello", "hello") == 0.0
    
    # Test different text
    assert calculate_cer("hello", "world") > 0.0
```

## Documentation

### Updating Documentation

1. **Update README.md** for user-facing changes
2. **Update docstrings** in code
3. **Update DEVELOPMENT.md** for development changes
4. **Generate docs** (if using Sphinx):
   ```bash
   make docs
   ```

### Code Documentation

- Use Google-style docstrings
- Include type hints
- Document all public methods and classes

Example:
```python
def calculate_metric(predicted: str, ground_truth: str) -> float:
    """Calculate a custom metric.
    
    Args:
        predicted: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        The calculated metric value
    """
    # Implementation here
    pass
```

## Contributing

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Run tests:**
   ```bash
   make check
   ```
5. **Commit your changes:**
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push to your fork**
7. **Create a pull request**

### Commit Message Format

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test changes
- `refactor:` for code refactoring

Example:
```
feat: add new OCR model support
fix: resolve memory leak in evaluator
docs: update installation instructions
```

## Troubleshooting

### Common Issues

1. **pipenv not found:**
   ```bash
   pip install pipenv
   ```

2. **Virtual environment issues:**
   ```bash
   pipenv --rm
   pipenv install
   ```

3. **Import errors:**
   - Make sure you're in the virtual environment: `pipenv shell`
   - Check that the project is installed: `pipenv install -e .`

4. **OCR model initialization fails:**
   - Install the required OCR engines (Tesseract, PaddleOCR, EasyOCR)
   - Check system dependencies

### Getting Help

- Check the README.md for usage instructions
- Check the README.md for usage examples
- Run `make help` for available commands
- Check the test files for usage examples

---

# Dataset Documentation

This section provides comprehensive information about the three main datasets used for evaluating OCR model performance.

## Overview

The framework supports three distinct datasets to provide comprehensive evaluation across different text types and languages:

1. **ICDAR 2013** - Printed text in natural images
2. **IAM Handwriting Database** - Handwritten English text
3. **Synthetic Mixed-Language Dataset** - Multilingual text

## 1. ICDAR 2013 Dataset

### Description
The ICDAR 2013 dataset is a benchmark dataset for text recognition in natural images. It contains printed text images from various sources including signs, advertisements, and documents.

### Characteristics
- **Type**: Printed text in natural images
- **Language**: English
- **Format**: Images with XML ground truth annotations
- **Source**: ICDAR 2013 Robust Reading Competition
- **Size**: ~1,000 images
- **Difficulty**: High (natural scene text)

### Ground Truth Format
XML files containing bounding boxes and text annotations for each image.

### Usage
```bash
# Evaluate on ICDAR 2013
make evaluate-icdar2013

# Using CLI
python -m src.cli.main evaluate-dataset --dataset icdar2013
```

### Citation
```bibtex
@inproceedings{karatzas2013icdar,
  title={ICDAR 2013 robust reading competition},
  author={Karatzas, Dimosthenis and Shafait, Faisal and Uchida, Seiichi and Iwamura, Masakazu and i Bigorda, Lluis Gomez and Mestre, Sergi Robles and Mas, Joan and Mota, David Fernandez and Almazan, Jon Almazan and de las Heras, Lluis Pere},
  booktitle={2013 12th International Conference on Document Analysis and Recognition},
  pages={1484--1493},
  year={2013},
  organization={IEEE}
}
```

## 2. IAM Handwriting Database

### Description
The IAM Handwriting Database is a large dataset of handwritten English text. It contains forms of handwritten English text which can be used to train and test handwritten text recognition systems.

### Characteristics
- **Type**: Handwritten text
- **Language**: English
- **Format**: Images with text line annotations
- **Source**: Institute of Computer Science and Applied Mathematics, University of Bern
- **Size**: ~1,000+ handwritten forms
- **Difficulty**: Medium-High (handwritten text)

### Ground Truth Format
Text files containing line-by-line transcriptions of the handwritten text in the images.

### Preprocessing
This dataset benefits from preprocessing:
- **Denoising**: Enabled by default
- **Contrast enhancement**: Enabled by default
- **Resizing**: Optional, configurable

### Usage
```bash
# Evaluate on IAM
make evaluate-iam

# Using CLI
python -m src.cli.main evaluate-dataset --dataset iam
```

### Citation
```bibtex
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

## 3. Synthetic Mixed-Language Dataset

### Description
A synthetic dataset containing text in multiple languages including English, French, German, Spanish, and Chinese. This dataset is designed to test OCR performance across different languages and scripts.

### Characteristics
- **Type**: Mixed (printed and synthetic)
- **Languages**: English, French, German, Spanish, Chinese
- **Format**: Images with JSON ground truth annotations
- **Source**: Synthetic generation
- **Size**: Variable (configurable)
- **Difficulty**: Medium (mixed languages and scripts)

### Ground Truth Format
JSON files containing:
- Text annotations
- Language labels
- Bounding boxes
- Confidence scores

### Language Support
- **English**: Latin script
- **French**: Latin script with accents
- **German**: Latin script with umlauts
- **Spanish**: Latin script with special characters
- **Chinese**: Simplified Chinese characters

### Usage
```bash
# Evaluate on Multilingual dataset
make evaluate-multilingual

# Using CLI
python -m src.cli.main evaluate-dataset --dataset multilingual
```

## Dataset Configuration

All datasets are configured in `config/evaluation_config.yaml`:

```yaml
datasets:
  icdar2013:
    name: "ICDAR 2013"
    description: "Printed text dataset from ICDAR 2013 competition"
    path: "data/datasets/icdar2013"
    type: "printed"
    languages: ["english"]
    ground_truth_format: "xml"
    
  iam:
    name: "IAM Handwriting Database"
    description: "Handwritten text dataset from IAM"
    path: "data/datasets/iam"
    type: "handwritten"
    languages: ["english"]
    ground_truth_format: "txt"
    
  multilingual:
    name: "Synthetic Mixed-Language Dataset"
    description: "Synthetic dataset with mixed languages"
    path: "data/datasets/multilingual"
    type: "mixed"
    languages: ["english", "french", "german", "spanish", "chinese"]
    ground_truth_format: "json"
```

## Directory Structure

Each dataset follows this structure:
```
data/datasets/
├── icdar2013/
│   ├── images/          # Image files
│   ├── ground_truth/    # Annotations
│   └── results/         # Evaluation results
├── iam/
│   ├── images/          # Image files
│   ├── ground_truth/    # Annotations
│   └── results/         # Evaluation results
└── multilingual/
    ├── images/          # Image files
    ├── ground_truth/    # Annotations
    └── results/         # Evaluation results
```

## Evaluation Commands

### Individual Datasets
```bash
# Evaluate specific datasets
make evaluate-icdar2013
make evaluate-iam
make evaluate-multilingual
```

### All Datasets
```bash
# Evaluate all datasets
make evaluate-all-datasets
```

### CLI Commands
```bash
# Using the command line interface
python -m src.cli.main evaluate-dataset --dataset icdar2013
python -m src.cli.main evaluate-dataset --dataset iam --models tesseract easyocr
python -m src.cli.main evaluate-dataset --dataset multilingual
```

## Expected Results

### Performance Expectations

1. **ICDAR 2013**:
   - Tesseract: CER ~15-25%, WER ~30-40%
   - PaddleOCR: CER ~8-15%, WER ~20-30%
   - EasyOCR: CER ~10-18%, WER ~25-35%

2. **IAM Handwriting**:
   - Tesseract: CER ~25-35%, WER ~40-50%
   - PaddleOCR: CER ~15-25%, WER ~30-40%
   - EasyOCR: CER ~20-30%, WER ~35-45%

3. **Multilingual**:
   - Performance varies by language
   - English: Best performance
   - Chinese: Moderate performance
   - European languages: Good performance

### Key Metrics
- **Character Error Rate (CER)**: Character-level accuracy
- **Word Error Rate (WER)**: Word-level accuracy
- **Processing Time**: Speed comparison
- **Memory Usage**: Resource consumption
- **Success Rate**: Percentage of successful recognitions

## Adding New Datasets

To add a new dataset:

1. Create directory structure in `data/datasets/`
2. Add configuration to `config/evaluation_config.yaml`
3. Update the CLI with new dataset option
4. Add Makefile commands
5. Update documentation

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure images and ground truth are in correct directories
2. **Format errors**: Check ground truth format matches configuration
3. **Memory issues**: Reduce batch size in configuration
4. **Performance issues**: Enable GPU acceleration if available

### Dataset Downloads

- **ICDAR 2013**: Available from ICDAR competition website
- **IAM**: Available from IAM database website
- **Multilingual**: Generate synthetic data or use existing multilingual datasets

## Contributing

When contributing new datasets:

1. Follow the established directory structure
2. Provide clear documentation
3. Include citation information
4. Test with all supported OCR models
5. Update configuration files
6. Add appropriate tests 