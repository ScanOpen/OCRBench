"""
Tests for evaluation metrics.
"""

import pytest
from src.evaluation.metrics import calculate_cer, calculate_wer, calculate_accuracy


def test_calculate_cer():
    """Test Character Error Rate calculation."""
    # Test exact match
    assert calculate_cer("hello", "hello") == 0.0
    
    # Test different text
    assert calculate_cer("hello", "world") > 0.0
    
    # Test empty strings
    assert calculate_cer("", "") == 0.0
    assert calculate_cer("hello", "") == 1.0
    assert calculate_cer("", "hello") == 0.0


def test_calculate_wer():
    """Test Word Error Rate calculation."""
    # Test exact match
    assert calculate_wer("hello world", "hello world") == 0.0
    
    # Test different text
    assert calculate_wer("hello world", "goodbye world") > 0.0
    
    # Test empty strings
    assert calculate_wer("", "") == 0.0
    assert calculate_wer("hello world", "") == 1.0


def test_calculate_accuracy():
    """Test accuracy calculation."""
    # Test exact match
    assert calculate_accuracy("hello", "hello") == 1.0
    
    # Test different text
    assert calculate_accuracy("hello", "world") == 0.0
    
    # Test empty strings
    assert calculate_accuracy("", "") == 1.0
    assert calculate_accuracy("hello", "") == 0.0 