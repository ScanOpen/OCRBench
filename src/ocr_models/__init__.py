"""
OCR Models Package

Contains implementations for different OCR models.
"""

from .base import BaseOCRModel
from .tesseract_ocr import TesseractOCR
from .paddle_ocr import PaddleOCR
from .easy_ocr import EasyOCR

__all__ = [
    "BaseOCRModel",
    "TesseractOCR", 
    "PaddleOCR",
    "EasyOCR"
] 