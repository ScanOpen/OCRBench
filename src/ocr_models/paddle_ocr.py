"""
PaddleOCR Model Implementation

Implementation of PaddleOCR using the paddleocr library.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from .base import BaseOCRModel

try:
    from paddleocr import PaddleOCR as PaddleOCREngine
except ImportError:
    PaddleOCREngine = None


class PaddleOCR(BaseOCRModel):
    """
    PaddleOCR model implementation.
    
    Uses Baidu's PaddleOCR library for text recognition.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize PaddleOCR model.
        
        Args:
            **kwargs: Additional parameters for PaddleOCR configuration
        """
        super().__init__("PaddleOCR", **kwargs)
        
        # Default configuration
        self.default_config = {
            'use_textline_orientation': False,
            'lang': 'en',
            'text_det_thresh': 0.1,
            'text_det_box_thresh': 0.1,
            'text_det_unclip_ratio': 2.0,
            'rec_batch_num': 6,
            'cls_batch_num': 6,
        }
        
        # Update with user-provided config
        self.default_config.update(kwargs)
        
    def initialize(self) -> None:
        """
        Initialize the PaddleOCR model.
        """
        if PaddleOCREngine is None:
            raise ImportError("PaddleOCR is not installed. Please install it with: pip install paddleocr")
        
        try:
            self.model = PaddleOCREngine(
                use_textline_orientation=self.default_config['use_textline_orientation'],
                lang=self.default_config['lang'],
                text_det_thresh=self.default_config['text_det_thresh'],
                text_det_box_thresh=self.default_config['text_det_box_thresh'],
                text_det_unclip_ratio=self.default_config['text_det_unclip_ratio'],
                rec_batch_num=self.default_config['rec_batch_num'],
                cls_batch_num=self.default_config['cls_batch_num']
            )
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {e}")
    
    def recognize_text(self, image: np.ndarray) -> str:
        """
        Recognize text in the given image using PaddleOCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Recognized text as string
        """
        if not self.is_initialized:
            self.initialize()
        
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        # Perform OCR using the new API
        result = self.model.predict(image)
        
        # Extract text from result
        if result is None or len(result) == 0:
            return ""
        
        # Handle the new PaddleOCR result structure
        text_parts = []
        for page_result in result:
            if isinstance(page_result, dict):
                # New API format - extract from rec_texts
                rec_texts = page_result.get('rec_texts', [])
                if rec_texts:
                    text_parts.extend(rec_texts)
            elif isinstance(page_result, list):
                # Old API format - extract from nested lists
                for line in page_result:
                    if line and len(line) > 0:
                        for detection in line:
                            if len(detection) >= 2:
                                text_parts.append(detection[1][0])  # Extract text from detection
        
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
        
        # Perform OCR using the new API
        result = self.model.predict(image)
        
        if result is None or len(result) == 0:
            return {
                'text': '',
                'confidence': 0.0,
                'detections': []
            }
        
        # Extract text and confidence information
        text_parts = []
        confidences = []
        detections = []
        
        for page_result in result:
            if isinstance(page_result, dict):
                # New API format - extract from rec_texts and rec_scores
                rec_texts = page_result.get('rec_texts', [])
                rec_scores = page_result.get('rec_scores', [])
                rec_boxes = page_result.get('rec_boxes', [])
                
                for i, text in enumerate(rec_texts):
                    confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                    bbox = rec_boxes[i] if i < len(rec_boxes) else []
                    
                    text_parts.append(text)
                    confidences.append(confidence)
                    detections.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            elif isinstance(page_result, list):
                # Old API format - extract from nested lists
                for line in page_result:
                    if line and len(line) > 0:
                        for detection in line:
                            if len(detection) >= 2:
                                text = detection[1][0] if isinstance(detection[1], (list, tuple)) else detection[1]
                                # Handle different confidence formats
                                if isinstance(detection[1], (list, tuple)) and len(detection[1]) > 1:
                                    confidence = detection[1][1]
                                else:
                                    confidence = 1.0  # Default confidence if not available
                                bbox = detection[0]
                                
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
        Get list of available languages for PaddleOCR.
        
        Returns:
            List of available language codes
        """
        return ['en', 'ch', 'french', 'german', 'korean', 'japan']
    
    def set_language(self, lang: str) -> None:
        """
        Set the language for OCR.
        
        Args:
            lang: Language code (e.g., 'en', 'ch', 'french')
        """
        self.default_config['lang'] = lang
        # Reinitialize model with new language
        if self.is_initialized:
            self.initialize()
    
    def set_use_gpu(self, use_gpu: bool) -> None:
        """
        Set whether to use GPU acceleration.
        
        Args:
            use_gpu: Whether to use GPU
        """
        # Note: GPU setting is handled automatically by PaddleOCR
        # based on available hardware
        pass
    
    def set_detection_thresholds(self, det_thresh: float, det_box_thresh: float) -> None:
        """
        Set detection thresholds.
        
        Args:
            det_thresh: Detection threshold
            det_box_thresh: Detection box threshold
        """
        self.default_config['text_det_thresh'] = det_thresh
        self.default_config['text_det_box_thresh'] = det_box_thresh 