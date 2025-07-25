"""
EasyOCR Model Implementation

Implementation of EasyOCR using the easyocr library.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from .base import BaseOCRModel

try:
    import easyocr
except ImportError:
    easyocr = None


class EasyOCR(BaseOCRModel):
    """
    EasyOCR model implementation.
    
    Uses the EasyOCR library for text recognition.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize EasyOCR model.
        
        Args:
            **kwargs: Additional parameters for EasyOCR configuration
        """
        super().__init__("EasyOCR", **kwargs)
        
        # Default configuration
        self.default_config = {
            'lang_list': ['en'],
            'gpu': False,
            'model_storage_directory': None,
            'user_network_directory': None,
            'recog_network': 'standard',
            'detect_network': 'craft',
            'download_enabled': True,
            'verbose': False,
            'quantize': True,
            'cudnn_benchmark': False,
        }
        
        # Update with user-provided config
        self.default_config.update(kwargs)
        
    def initialize(self) -> None:
        """
        Initialize the EasyOCR model.
        """
        if easyocr is None:
            raise ImportError("EasyOCR is not installed. Please install it with: pip install easyocr")
        
        try:
            self.model = easyocr.Reader(
                lang_list=self.default_config['lang_list'],
                gpu=self.default_config['gpu'],
                model_storage_directory=self.default_config['model_storage_directory'],
                user_network_directory=self.default_config['user_network_directory'],
                recog_network=self.default_config['recog_network'],
                detect_network=self.default_config['detect_network'],
                download_enabled=self.default_config['download_enabled'],
                verbose=self.default_config['verbose'],
                quantize=self.default_config['quantize'],
                cudnn_benchmark=self.default_config['cudnn_benchmark']
            )
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR: {e}")
    
    def recognize_text(self, image: np.ndarray) -> str:
        """
        Recognize text in the given image using EasyOCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Recognized text as string
        """
        if not self.is_initialized:
            self.initialize()
        
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        # Perform OCR
        result = self.model.readtext(image)
        
        # Extract text from result
        if not result:
            return ""
        
        # EasyOCR returns a list of tuples: (bbox, text, confidence)
        text_parts = [detection[1] for detection in result]
        
        return ' '.join(text_parts)
    
    def recognize_text_with_confidence(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Recognize text with confidence scores and bounding boxes.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing text, confidence, and bounding box information
        """
        if not self.is_initialized:
            self.initialize()
        
        # Perform OCR
        result = self.model.readtext(image)
        
        if not result:
            return {
                'text': '',
                'confidence': 0.0,
                'detections': []
            }
        
        # Extract text and confidence information
        text_parts = []
        confidences = []
        detections = []
        
        for detection in result:
            bbox, text, confidence = detection
            
            text_parts.append(text)
            confidences.append(confidence)
            detections.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'detections': detections,
            'word_confidences': list(zip(text_parts, confidences))
        }
    
    def get_available_languages(self) -> List[str]:
        """
        Get list of available languages for EasyOCR.
        
        Returns:
            List of available language codes
        """
        return [
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'th', 'vi', 'ar', 'hi', 'bn', 'ur', 'te', 'ta', 'gu', 'kn', 'ml', 'si', 'my', 'ne', 'fa', 'he', 'id', 'ms', 'bn', 'ur', 'te', 'ta', 'gu', 'kn', 'ml', 'si', 'my', 'ne', 'fa', 'he', 'id', 'ms'
        ]
    
    def set_language(self, lang: str) -> None:
        """
        Set the language for OCR.
        
        Args:
            lang: Language code (e.g., 'en', 'ch_sim', 'ja')
        """
        self.default_config['lang_list'] = [lang]
        # Reinitialize model with new language
        if self.is_initialized:
            self.initialize()
    
    def set_languages(self, languages: List[str]) -> None:
        """
        Set multiple languages for OCR.
        
        Args:
            languages: List of language codes
        """
        self.default_config['lang_list'] = languages
        # Reinitialize model with new languages
        if self.is_initialized:
            self.initialize()
    
    def set_gpu(self, use_gpu: bool) -> None:
        """
        Set whether to use GPU acceleration.
        
        Args:
            use_gpu: Whether to use GPU
        """
        self.default_config['gpu'] = use_gpu
        # Reinitialize model with new GPU setting
        if self.is_initialized:
            self.initialize()
    
    def set_recognition_network(self, network: str) -> None:
        """
        Set the recognition network.
        
        Args:
            network: Network type ('standard', 'arabic_g2', 'chinese_sim', etc.)
        """
        self.default_config['recog_network'] = network
        # Reinitialize model with new network
        if self.is_initialized:
            self.initialize()
    
    def set_detector_network(self, network: str) -> None:
        """
        Set the detector network.
        
        Args:
            network: Network type ('craft', 'dbnet18', 'dbnet50', etc.)
        """
        self.default_config['detector_network'] = network
        # Reinitialize model with new network
        if self.is_initialized:
            self.initialize() 