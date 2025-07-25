"""
Image processing utilities for OCR evaluation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Args:
        image: Input image as numpy array
        target_size: Target (width, height)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: float = 0) -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        image: Input image as numpy array
        alpha: Contrast factor (1.0 = no change)
        beta: Brightness offset
        
    Returns:
        Enhanced image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def denoise_image(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """
    Remove noise from image.
    
    Args:
        image: Input image as numpy array
        method: Denoising method ("bilateral", "gaussian", "median")
        
    Returns:
        Denoised image
    """
    if method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(image, 5)
    else:
        return image


def binarize_image(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Convert image to binary (black and white).
    
    Args:
        image: Input image as numpy array
        method: Binarization method ("otsu", "adaptive", "simple")
        
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == "simple":
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = gray
    
    return binary


def preprocess_for_ocr(image: np.ndarray, 
                      resize_to: Optional[Tuple[int, int]] = None,
                      do_enhance_contrast: bool = True,
                      do_denoise: bool = True,
                      do_binarize: bool = False) -> np.ndarray:
    """
    Apply a complete preprocessing pipeline for OCR.
    
    Args:
        image: Input image as numpy array
        resize_to: Target size for resizing (None = no resize)
        enhance_contrast: Whether to enhance contrast
        denoise: Whether to denoise the image
        binarize: Whether to binarize the image
        
    Returns:
        Preprocessed image
    """
    processed = image.copy()
    
    if resize_to:
        processed = resize_image(processed, resize_to)
    
    if do_enhance_contrast:
        processed = enhance_contrast(processed, alpha=1.5, beta=0)
    
    if do_denoise:
        processed = denoise_image(processed)
    
    if do_binarize:
        processed = binarize_image(processed)
    
    return processed 