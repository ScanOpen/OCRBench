"""
Evaluation Metrics

Contains various metrics for evaluating OCR model performance.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Any
from editdistance import eval as edit_distance
import jiwer


def calculate_cer(predicted: str, ground_truth: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Character Error Rate as a float between 0 and 1
    """
    if not ground_truth:
        return 1.0 if predicted else 0.0
    
    # Normalize text
    predicted = _normalize_text(predicted)
    ground_truth = _normalize_text(ground_truth)
    
    # Calculate edit distance
    distance = edit_distance(predicted, ground_truth)
    
    # Calculate CER
    cer = distance / len(ground_truth)
    
    return min(cer, 1.0)  # Cap at 1.0


def calculate_wer(predicted: str, ground_truth: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Word Error Rate as a float between 0 and 1
    """
    if not ground_truth:
        return 1.0 if predicted else 0.0
    
    # Normalize text
    predicted = _normalize_text(predicted)
    ground_truth = _normalize_text(ground_truth)
    
    # Split into words
    predicted_words = predicted.split()
    ground_truth_words = ground_truth.split()
    
    if not ground_truth_words:
        return 1.0 if predicted_words else 0.0
    
    # Calculate WER using jiwer
    wer = jiwer.wer(ground_truth, predicted)
    
    return min(wer, 1.0)  # Cap at 1.0


def calculate_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculate overall accuracy.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    
    # Normalize text
    predicted = _normalize_text(predicted)
    ground_truth = _normalize_text(ground_truth)
    
    # Calculate accuracy
    if predicted == ground_truth:
        return 1.0
    else:
        return 0.0


def calculate_precision_recall(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate precision and recall for character-level matching.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Dictionary containing precision and recall scores
    """
    if not ground_truth:
        return {'precision': 0.0, 'recall': 0.0}
    
    # Normalize text
    predicted = _normalize_text(predicted)
    ground_truth = _normalize_text(ground_truth)
    
    # Convert to character lists
    predicted_chars = list(predicted)
    ground_truth_chars = list(ground_truth)
    
    # Calculate true positives, false positives, false negatives
    tp = 0
    fp = 0
    fn = 0
    
    # Use dynamic programming to find longest common subsequence
    lcs_length = _longest_common_subsequence(predicted_chars, ground_truth_chars)
    
    tp = lcs_length
    fp = len(predicted_chars) - lcs_length
    fn = len(ground_truth_chars) - lcs_length
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    }


def calculate_edit_distance(predicted: str, ground_truth: str) -> int:
    """
    Calculate edit distance between predicted and ground truth text.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Edit distance as integer
    """
    # Normalize text
    predicted = _normalize_text(predicted)
    ground_truth = _normalize_text(ground_truth)
    
    return edit_distance(predicted, ground_truth)


def calculate_all_metrics(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a single prediction.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['cer'] = calculate_cer(predicted, ground_truth)
    metrics['wer'] = calculate_wer(predicted, ground_truth)
    metrics['accuracy'] = calculate_accuracy(predicted, ground_truth)
    metrics['edit_distance'] = calculate_edit_distance(predicted, ground_truth)
    
    # Precision/Recall
    pr_metrics = calculate_precision_recall(predicted, ground_truth)
    metrics.update(pr_metrics)
    
    # Additional metrics
    metrics['length_ratio'] = len(predicted) / len(ground_truth) if ground_truth else 0.0
    
    return metrics


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation (optional, can be made configurable)
    # text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()


def _longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """
    Calculate the length of the longest common subsequence.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Length of longest common subsequence
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple evaluations.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary containing aggregated metrics
    """
    if not results:
        return {}
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric names from the first result
    metric_names = [key for key in results[0].keys() if isinstance(results[0][key], (int, float))]
    
    for metric in metric_names:
        values = [result[metric] for result in results if metric in result]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
            aggregated[f'{metric}_median'] = np.median(values)
    
    return aggregated 