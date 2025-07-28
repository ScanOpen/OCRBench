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
	pipenv run python -m src.cli.main generate-text-comparison --results-dir data/datasets/icdar2013/results --output data/datasets/icdar2013/results/text_comparison_report.txt --num-samples 20
	pipenv run python -m src.cli.main generate-detailed-text-report --results-dir data/datasets/icdar2013/results --output data/datasets/icdar2013/results/detailed_text_report.txt
	@echo "ICDAR 2013 evaluation completed with comparison reports generated!"
	@echo "Note: If detailed per-image results are not available, summary reports will be generated instead."

evaluate-iam:
	pipenv run python -m src.cli.main evaluate-dataset --dataset iam --output-dir data/datasets/iam/results
	pipenv run python -m src.cli.main generate-text-comparison --results-dir data/datasets/iam/results --output data/datasets/iam/results/text_comparison_report.txt --num-samples 20
	pipenv run python -m src.cli.main generate-detailed-text-report --results-dir data/datasets/iam/results --output data/datasets/iam/results/detailed_text_report.txt
	@echo "IAM evaluation completed with comparison reports generated!"

evaluate-multilingual:
	pipenv run python -m src.cli.main evaluate-dataset --dataset multilingual --output-dir data/datasets/multilingual/results

# Test single image evaluation
evaluate-test-single:
	pipenv run python -m src.cli.main evaluate-dataset --dataset test_single --output-dir data/test_single_image/results
	pipenv run python -m src.cli.main generate-text-comparison --results-dir data/test_single_image/results --output data/test_single_image/results/text_comparison_report.txt --num-samples 1
	pipenv run python -m src.cli.main generate-detailed-text-report --results-dir data/test_single_image/results --output data/test_single_image/results/detailed_text_report.txt
	@echo "Test single image evaluation completed with comparison reports generated!"

# Evaluate all datasets
evaluate-all-datasets: evaluate-icdar2013 evaluate-iam evaluate-multilingual

# Generate text comparison reports
generate-text-comparison-icdar2013:
	pipenv run python -m src.cli.main generate-text-comparison --results-dir data/datasets/icdar2013/results --output data/datasets/icdar2013/results/text_comparison_report.txt --num-samples 20

generate-text-comparison-iam:
	pipenv run python -m src.cli.main generate-text-comparison --results-dir data/datasets/iam/results --output data/datasets/iam/results/text_comparison_report.txt --num-samples 20

generate-detailed-text-report-icdar2013:
	pipenv run python -m src.cli.main generate-detailed-text-report --results-dir data/datasets/icdar2013/results --output data/datasets/icdar2013/results/detailed_text_report.txt

generate-detailed-text-report-iam:
	pipenv run python -m src.cli.main generate-detailed-text-report --results-dir data/datasets/iam/results --output data/datasets/iam/results/detailed_text_report.txt

# Generate all text comparison reports
generate-all-text-reports: generate-text-comparison-icdar2013 generate-text-comparison-iam generate-detailed-text-report-icdar2013 generate-detailed-text-report-iam

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
	@echo "  evaluate-icdar2013 - Evaluate ICDAR 2013 dataset (with comparison reports)"
	@echo "  evaluate-iam - Evaluate IAM handwriting dataset (with comparison reports)"
	@echo "  evaluate-multilingual - Evaluate multilingual dataset"
	@echo "  evaluate-test-single - Evaluate single test image (with comparison reports)"
	@echo "  evaluate-all-datasets - Evaluate all datasets"
	@echo "  generate-text-comparison-icdar2013 - Generate sample text comparisons for ICDAR 2013"
	@echo "  generate-text-comparison-iam - Generate sample text comparisons for IAM"
	@echo "  generate-detailed-text-report-icdar2013 - Generate detailed text report for ICDAR 2013"
	@echo "  generate-detailed-text-report-iam - Generate detailed text report for IAM"
	@echo "  generate-all-text-reports - Generate all text comparison reports"
	@echo "  download-datasets - Download and setup datasets"
	@echo "  example      - Run example usage"
	@echo "  shell        - Activate virtual environment"
	@echo "  clean        - Clean up virtual environment and cache"
	@echo "  install-e    - Install project in development mode"
	@echo "  lint         - Run code linting"
	@echo "  type-check   - Run type checking"
	@echo "  check        - Run all checks (lint + type-check + test)"
	@echo "  docs         - Generate documentation" 