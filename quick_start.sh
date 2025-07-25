#!/bin/bash

# OCR Evaluation Framework - Quick Start Script
# This script sets up the project with pipenv

set -e  # Exit on any error

echo "🚀 OCR Evaluation Framework - Quick Start"
echo "========================================"

# Check if pipenv is installed
if ! command -v pipenv &> /dev/null; then
    echo "❌ pipenv is not installed. Installing pipenv..."
    pip install pipenv
fi

echo "✅ pipenv is available"

# Check Python version
echo "🐍 Checking Python version..."
python3 check_python_version.py

# Check if we're in the right directory
if [ ! -f "Pipfile" ]; then
    echo "❌ Pipfile not found. Please run this script from the project root directory."
    exit 1
fi

echo "📦 Installing dependencies..."
pipenv install

echo "🔧 Installing project in development mode..."
pipenv install -e .

echo "🧪 Running installation test..."
pipenv run python test_installation.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   pipenv shell"
echo ""
echo "2. Run the example:"
echo "   make evaluate"
echo "   # or use: make example for usage info"
echo ""
echo "3. Run evaluation:"
echo "   pipenv run python test_ocr_evaluation.py"
echo "   # or use: make evaluate"
echo ""
echo "4. Explore the documentation:"
echo "   # All documentation is now in README.md"
echo ""
echo "5. See all available commands:"
echo "   make help"
echo ""
echo "Happy OCR evaluating! 🎯" 