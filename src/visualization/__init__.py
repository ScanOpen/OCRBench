"""
Visualization Package

Contains tools for visualizing OCR evaluation results.
"""

from .plots import (
    plot_accuracy_comparison,
    plot_error_rates,
    plot_processing_times,
    plot_confusion_matrix,
    create_dashboard
)

__all__ = [
    "plot_accuracy_comparison",
    "plot_error_rates", 
    "plot_processing_times",
    "plot_confusion_matrix",
    "create_dashboard"
] 