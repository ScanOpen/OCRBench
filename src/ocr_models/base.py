"""
Base OCR Model

Abstract base class for OCR models with common functionality.
"""

import time
import psutil
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class BaseOCRModel(ABC):
    """
    Abstract base class for OCR models.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the OCR model.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.is_initialized = False
        self.config = kwargs
        self.memory_tracker = None
        self._initialize_memory_tracker()
    
    def _initialize_memory_tracker(self):
        """Initialize memory tracker lazily to avoid circular imports."""
        # Memory tracking has been disabled - always set to None
        self.memory_tracker = None
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model. This method should load the model into memory.
        """
        pass
    
    @abstractmethod
    def recognize_text(self, image: np.ndarray) -> str:
        """
        Recognize text in the given image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Recognized text as string
        """
        pass
    
    def recognize_text_with_metrics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Recognize text and return performance metrics.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing recognized text and performance metrics
        """
        if not self.is_initialized:
            self.initialize()
        
        # Record start time
        start_time = time.time()
        
        # Perform OCR
        try:
            text = self.recognize_text(image)
            success = True
            error = None
        except Exception as e:
            text = ""
            success = False
            error = str(e)
        
        # Record end time
        end_time = time.time()
        processing_time = end_time - start_time
        
        result = {
            'text': text,
            'processing_time': processing_time,
            'success': success
        }
        
        if not success:
            result['error'] = error
            
        return result
    
    def _recognize_text_with_enhanced_memory_tracking(self, image: np.ndarray) -> Dict[str, Any]:
        """Recognize text with enhanced memory tracking."""
        with self.memory_tracker.track_memory_usage(f"{self.model_name}_ocr") as memory_info:
            start_time = time.time()
            
            # Perform OCR
            try:
                text = self.recognize_text(image)
                success = True
                error = None
            except Exception as e:
                text = ""
                success = False
                error = str(e)
            
            end_time = time.time()
            processing_time = end_time - start_time
        
        # Extract memory metrics from the tracking results
        memory_metrics = self._extract_memory_metrics(memory_info)
        
        result = {
            'text': text,
            'processing_time': processing_time,
            'memory_used': memory_metrics['total_memory_used_mb'],
            'success': success,
            'memory_details': memory_metrics
        }
        
        if not success:
            result['error'] = error
            
        return result
    
    def _recognize_text_with_basic_memory_tracking(self, image: np.ndarray) -> Dict[str, Any]:
        """Recognize text with basic memory tracking (fallback)."""
        # Record start time and memory usage
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform OCR
        try:
            text = self.recognize_text(image)
            success = True
        except Exception as e:
            text = ""
            success = False
            error = str(e)
        
        # Record end time and memory usage
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        processing_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        result = {
            'text': text,
            'processing_time': processing_time,
            'memory_used': memory_used,
            'success': success,
            'memory_details': {
                'total_memory_used_mb': memory_used,
                'peak_memory_used_mb': memory_used,
                'psutil_rss_diff_mb': memory_used,
                'psutil_vms_diff_mb': 0.0,
                'psutil_percent_diff': 0.0,
                'tracemalloc_current_diff_mb': 0.0,
                'tracemalloc_peak_diff_mb': 0.0,
                'duration': processing_time
            }
        }
        
        if not success:
            result['error'] = error
            
        return result
    
    def _extract_memory_metrics(self, memory_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format memory metrics from memory tracking results.
        
        Args:
            memory_info: Memory tracking information
            
        Returns:
            Dictionary with formatted memory metrics
        """
        # Get basic memory differences
        psutil_diff = memory_info.get('psutil_diff', {})
        tracemalloc_diff = memory_info.get('tracemalloc_diff', {})
        
        # Calculate total memory used (prefer tracemalloc if available, fallback to psutil)
        total_memory_used_mb = 0.0
        if tracemalloc_diff and 'current_diff_mb' in tracemalloc_diff:
            total_memory_used_mb = tracemalloc_diff['current_diff_mb']
        elif psutil_diff and 'rss_diff_mb' in psutil_diff:
            total_memory_used_mb = psutil_diff['rss_diff_mb']
        
        # Get peak memory usage
        peak_memory_used_mb = 0.0
        if tracemalloc_diff and 'peak_diff_mb' in tracemalloc_diff:
            peak_memory_used_mb = tracemalloc_diff['peak_diff_mb']
        
        return {
            'total_memory_used_mb': total_memory_used_mb,
            'peak_memory_used_mb': peak_memory_used_mb,
            'psutil_rss_diff_mb': psutil_diff.get('rss_diff_mb', 0.0),
            'psutil_vms_diff_mb': psutil_diff.get('vms_diff_mb', 0.0),
            'psutil_percent_diff': psutil_diff.get('percent_diff', 0.0),
            'tracemalloc_current_diff_mb': tracemalloc_diff.get('current_diff_mb', 0.0),
            'tracemalloc_peak_diff_mb': tracemalloc_diff.get('peak_diff_mb', 0.0),
            'duration': memory_info.get('duration', 0.0)
        }
    
    def batch_recognize(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Recognize text in multiple images with enhanced memory tracking.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of dictionaries containing recognition results and metrics
        """
        results = []
        
        if self.memory_tracker is not None:
            # Track overall batch memory usage
            with self.memory_tracker.track_memory_usage(f"{self.model_name}_batch_ocr") as batch_memory_info:
                for i, image in enumerate(images):
                    # Track individual image processing
                    with self.memory_tracker.track_memory_usage(f"{self.model_name}_image_{i}") as image_memory_info:
                        result = self.recognize_text_with_metrics(image)
                        
                        # Add batch context to individual results
                        result['batch_index'] = i
                        result['batch_memory_context'] = {
                            'batch_total_images': len(images),
                            'batch_current_image': i + 1
                        }
                        
                        results.append(result)
                
                # Add batch-level memory metrics
                batch_memory_metrics = self._extract_memory_metrics(batch_memory_info)
                for result in results:
                    result['batch_memory_metrics'] = batch_memory_metrics
        else:
            # Fallback to basic batch processing
            for i, image in enumerate(images):
                result = self.recognize_text_with_metrics(image)
                result['batch_index'] = i
                results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'is_initialized': self.is_initialized,
            'config': self.config,
            'memory_tracker_enabled': self.memory_tracker is not None
        }
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """
        Get detailed memory profile information.
        
        Returns:
            Dictionary with memory profile information
        """
        if not self.memory_tracker:
            return {'error': 'Memory tracker not available'}
        
        return {
            'current_memory': self.memory_tracker.get_memory_snapshot(),
            'top_allocations': self.memory_tracker.get_top_memory_allocations(limit=5),
            'system_memory': self.memory_tracker.get_psutil_memory()
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.model_name} (initialized: {self.is_initialized})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', initialized={self.is_initialized})" 