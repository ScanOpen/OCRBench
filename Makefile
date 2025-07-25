.PHONY: install install-dev test evaluate example clean shell

# Install production dependencies
install:
	pipenv install

# Install development dependencies
install-dev:
	pipenv install --dev

# Check Python version
check-python:
	pipenv run python check_python_version.py

# Run tests
test:
	pipenv run python test_installation.py

# Run evaluation
evaluate:
	pipenv run python test_ocr_evaluation.py

# Evaluate specific datasets
evaluate-icdar2013:
	pipenv run python -m src.cli.main evaluate-dataset --dataset icdar2013 --output-dir data/datasets/icdar2013/results

evaluate-iam:
	pipenv run python -m src.cli.main evaluate-dataset --dataset iam --output-dir data/datasets/iam/results

evaluate-multilingual:
	pipenv run python -m src.cli.main evaluate-dataset --dataset multilingual --output-dir data/datasets/multilingual/results

# Evaluate all datasets
evaluate-all-datasets: evaluate-icdar2013 evaluate-iam evaluate-multilingual

# Download dataset helper
download-datasets:
	pipenv run python download_datasets.py

# Run example (shows usage information)
example:
	@echo "OCR Evaluation Framework Usage Examples:"
	@echo "========================================"
	@echo ""
	@echo "1. Quick Start:"
	@echo "   ./quick_start.sh"
	@echo ""
	@echo "2. Manual Setup:"
	@echo "   pipenv install"
	@echo "   pipenv install -e ."
	@echo "   pipenv shell"
	@echo ""
	@echo "3. Run Evaluation:"
	@echo "   make evaluate"
	@echo ""
	@echo "4. Run Tests:"
	@echo "   make test"
	@echo ""
	@echo "5. See all commands:"
	@echo "   make help"

# Activate virtual environment
shell:
	pipenv shell

# Clean up
clean:
	pipenv --rm
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install the project in development mode
install-e:
	pipenv install -e .

# Run linting
lint:
	pipenv run black src/ tests/
	pipenv run flake8 src/ tests/

# Run type checking
type-check:
	pipenv run mypy src/

# Run all checks
check: lint type-check test

# Generate documentation
docs:
	@echo "Documentation is now integrated into README.md"
	@echo "No separate documentation generation needed."

# Help
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  check-python - Check Python version compatibility"
	@echo "  test         - Run installation tests"
	@echo "  evaluate     - Run OCR evaluation"
	@echo "  evaluate-icdar2013 - Evaluate ICDAR 2013 dataset"
	@echo "  evaluate-iam - Evaluate IAM handwriting dataset"
	@echo "  evaluate-multilingual - Evaluate multilingual dataset"
	@echo "  evaluate-all-datasets - Evaluate all datasets"
	@echo "  download-datasets - Download and setup datasets"
	@echo "  example      - Run example usage"
	@echo "  shell        - Activate virtual environment"
	@echo "  clean        - Clean up virtual environment and cache"
	@echo "  install-e    - Install project in development mode"
	@echo "  lint         - Run code linting"
	@echo "  type-check   - Run type checking"
	@echo "  check        - Run all checks (lint + type-check + test)"
	@echo "  docs         - Generate documentation" 