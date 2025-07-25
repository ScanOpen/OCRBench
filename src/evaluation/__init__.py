"""
Evaluation Package

Contains evaluation metrics and tools for OCR model comparison.
"""

from .metrics import (
    calculate_cer,
    calculate_wer,
    calculate_accuracy,
    calculate_precision_recall,
    calculate_edit_distance
)

from .evaluator import OCREvaluator

__all__ = [
    "calculate_cer",
    "calculate_wer", 
    "calculate_accuracy",
    "calculate_precision_recall",
    "calculate_edit_distance",
    "OCREvaluator"
] 