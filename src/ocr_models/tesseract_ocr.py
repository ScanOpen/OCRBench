"""
Tesseract OCR Model Implementation

Implementation of Tesseract OCR using pytesseract.
"""

import pytesseract
import numpy as np
from typing import Dict, Any
from .base import BaseOCRModel


class TesseractOCR(BaseOCRModel):
    """
    Tesseract OCR model implementation.
    
    Uses pytesseract to interface with Google's Tesseract OCR engine.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Tesseract OCR model.
        
        Args:
            **kwargs: Additional parameters for Tesseract configuration
        """
        super().__init__("Tesseract", **kwargs)
        
        # Default configuration
        self.default_config = {
            'lang': 'eng',
            'config': '--psm 6',  # Assume uniform block of text
            'oem': 3,  # Default OCR Engine Mode
        }
        
        # Update with user-provided config
        self.default_config.update(kwargs)
        
    def initialize(self) -> None:
        """
        Initialize the Tesseract model.
        
        Note: Tesseract doesn't require explicit initialization,
        but we check if it's available.
        """
        try:
            # Check if Tesseract is available
            pytesseract.get_tesseract_version()
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Tesseract not available: {e}")
    
    def recognize_text(self, image: np.ndarray) -> str:
        """
        Recognize text in the given image using Tesseract.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Recognized text as string
        """
        if not self.is_initialized:
            self.initialize()
        
        # Convert image to PIL Image if needed
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        # Perform OCR
        text = pytesseract.image_to_string(
            image,
            lang=self.default_config['lang'],
            config=self.default_config['config'],
            output_type=pytesseract.Output.STRING
        )
        
        return text.strip()
    
    def recognize_text_with_confidence(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Recognize text with confidence scores.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing text and confidence information
        """
        if not self.is_initialized:
            self.initialize()
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(
            image,
            lang=self.default_config['lang'],
            config=self.default_config['config'],
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text and confidence
        text_parts = []
        confidences = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Filter out low confidence results
                text_parts.append(data['text'][i])
                confidences.append(int(data['conf'][i]))
        
        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'word_confidences': list(zip(text_parts, confidences))
        }
    
    def get_available_languages(self) -> list:
        """
        Get list of available languages for Tesseract.
        
        Returns:
            List of available language codes
        """
        try:
            return pytesseract.get_languages()
        except Exception:
            return ['eng']  # Default to English if languages can't be retrieved
    
    def set_language(self, lang: str) -> None:
        """
        Set the language for OCR.
        
        Args:
            lang: Language code (e.g., 'eng', 'fra', 'deu')
        """
        self.default_config['lang'] = lang
    
    def set_page_segmentation_mode(self, psm: int) -> None:
        """
        Set the page segmentation mode.
        
        Args:
            psm: Page segmentation mode (0-13)
        """
        self.default_config['config'] = f'--psm {psm}'
    
    def set_ocr_engine_mode(self, oem: int) -> None:
        """
        Set the OCR engine mode.
        
        Args:
            oem: OCR engine mode (0-3)
        """
        self.default_config['oem'] = oem 