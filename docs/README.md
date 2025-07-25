# Documentation

This directory contains documentation for the OCR Evaluation Framework.

## Contents

- API documentation (generated with Sphinx)
- User guides
- Development documentation

## Building Documentation

```bash
# Install documentation dependencies
pipenv install --dev

# Build documentation
make docs
```

## Documentation Structure

- `README.md` - This file
- `_build/` - Generated documentation (not in version control)
- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index 