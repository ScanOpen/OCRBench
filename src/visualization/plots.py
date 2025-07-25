"""
Plotting functions for OCR evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


def plot_accuracy_comparison(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot accuracy comparison between models.
    
    Args:
        results: Dictionary containing evaluation results
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Extract data
    model_names = list(results.keys())
    cer_values = [results[name].get('cer_mean', 0) for name in model_names]
    wer_values = [results[name].get('wer_mean', 0) for name in model_names]
    accuracy_values = [results[name].get('accuracy_mean', 0) for name in model_names]
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # CER comparison
    bars1 = ax1.bar(model_names, cer_values, color='skyblue', alpha=0.7)
    ax1.set_title('Character Error Rate (CER)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('CER', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # WER comparison
    bars2 = ax2.bar(model_names, wer_values, color='lightcoral', alpha=0.7)
    ax2.set_title('Word Error Rate (WER)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('WER', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Accuracy comparison
    bars3 = ax3.bar(model_names, accuracy_values, color='lightgreen', alpha=0.7)
    ax3.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_error_rates(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot error rates comparison.
    
    Args:
        results: Dictionary containing evaluation results
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    model_names = list(results.keys())
    cer_values = [results[name].get('cer_mean', 0) for name in model_names]
    wer_values = [results[name].get('wer_mean', 0) for name in model_names]
    
    # Create grouped bar plot
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, cer_values, width, label='CER', color='skyblue', alpha=0.7)
    plt.bar(x + width/2, wer_values, width, label='WER', color='lightcoral', alpha=0.7)
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('Character vs Word Error Rates', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (cer, wer) in enumerate(zip(cer_values, wer_values)):
        plt.text(i - width/2, cer + 0.01, f'{cer:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, wer + 0.01, f'{wer:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_processing_times(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot processing time comparison.
    
    Args:
        results: Dictionary containing evaluation results
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    model_names = list(results.keys())
    time_values = [results[name].get('avg_processing_time', 0) for name in model_names]
    memory_values = [results[name].get('avg_memory_used', 0) for name in model_names]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Processing time
    bars1 = ax1.bar(model_names, time_values, color='lightblue', alpha=0.7)
    ax1.set_title('Average Processing Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}s', ha='center', va='bottom')
    
    # Memory usage
    bars2 = ax2.bar(model_names, memory_values, color='lightgreen', alpha=0.7)
    ax2.set_title('Average Memory Usage', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}MB', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(predicted: str, ground_truth: str, save_path: Optional[str] = None):
    """
    Plot character-level confusion matrix.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        save_path: Optional path to save the plot
    """
    # Normalize texts
    predicted = predicted.lower().strip()
    ground_truth = ground_truth.lower().strip()
    
    # Get unique characters
    all_chars = set(predicted + ground_truth)
    char_to_idx = {char: i for i, char in enumerate(sorted(all_chars))}
    
    # Create confusion matrix
    matrix = np.zeros((len(all_chars), len(all_chars)))
    
    # Count character occurrences
    for char in predicted:
        if char in char_to_idx:
            matrix[char_to_idx[char], char_to_idx[char]] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=sorted(all_chars), yticklabels=sorted(all_chars))
    plt.title('Character Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Characters', fontsize=12)
    plt.ylabel('Ground Truth Characters', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_dashboard(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Create a comprehensive dashboard of evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
        save_path: Optional path to save the dashboard
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Extract data
    model_names = list(results.keys())
    cer_values = [results[name].get('cer_mean', 0) for name in model_names]
    wer_values = [results[name].get('wer_mean', 0) for name in model_names]
    accuracy_values = [results[name].get('accuracy_mean', 0) for name in model_names]
    time_values = [results[name].get('avg_processing_time', 0) for name in model_names]
    memory_values = [results[name].get('avg_memory_used', 0) for name in model_names]
    success_rates = [results[name].get('successful_images', 0) / results[name].get('total_images', 1) 
                    for name in model_names]
    
    # 1. CER comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(model_names, cer_values, color='skyblue', alpha=0.7)
    ax1.set_title('Character Error Rate', fontweight='bold')
    ax1.set_ylabel('CER')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. WER comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(model_names, wer_values, color='lightcoral', alpha=0.7)
    ax2.set_title('Word Error Rate', fontweight='bold')
    ax2.set_ylabel('WER')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Accuracy comparison
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(model_names, accuracy_values, color='lightgreen', alpha=0.7)
    ax3.set_title('Accuracy', fontweight='bold')
    ax3.set_ylabel('Accuracy')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Processing time
    ax4 = fig.add_subplot(gs[0, 3])
    bars = ax4.bar(model_names, time_values, color='gold', alpha=0.7)
    ax4.set_title('Processing Time', fontweight='bold')
    ax4.set_ylabel('Time (s)')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Memory usage
    ax5 = fig.add_subplot(gs[1, 0])
    bars = ax5.bar(model_names, memory_values, color='lightblue', alpha=0.7)
    ax5.set_title('Memory Usage', fontweight='bold')
    ax5.set_ylabel('Memory (MB)')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Success rate
    ax6 = fig.add_subplot(gs[1, 1])
    bars = ax6.bar(model_names, success_rates, color='lightpink', alpha=0.7)
    ax6.set_title('Success Rate', fontweight='bold')
    ax6.set_ylabel('Success Rate')
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. Radar chart for overall performance
    ax7 = fig.add_subplot(gs[1, 2:], projection='polar')
    
    # Normalize metrics for radar chart
    metrics = ['CER', 'WER', 'Accuracy', 'Speed', 'Memory']
    normalized_values = []
    
    for i, name in enumerate(model_names):
        values = [
            1 - cer_values[i],  # Lower CER is better
            1 - wer_values[i],  # Lower WER is better
            accuracy_values[i],  # Higher accuracy is better
            1 - (time_values[i] / max(time_values)),  # Lower time is better
            1 - (memory_values[i] / max(memory_values))  # Lower memory is better
        ]
        normalized_values.append(values)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, name in enumerate(model_names):
        values = normalized_values[i] + [normalized_values[i][0]]  # Complete the circle
        ax7.plot(angles, values, 'o-', linewidth=2, label=name)
        ax7.fill(angles, values, alpha=0.25)
    
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(metrics)
    ax7.set_title('Overall Performance Comparison', fontweight='bold')
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 8. Summary table
    ax8 = fig.add_subplot(gs[2, :])
    ax8.axis('tight')
    ax8.axis('off')
    
    # Create summary table
    table_data = []
    for name in model_names:
        table_data.append([
            name,
            f"{cer_values[model_names.index(name)]:.4f}",
            f"{wer_values[model_names.index(name)]:.4f}",
            f"{accuracy_values[model_names.index(name)]:.4f}",
            f"{time_values[model_names.index(name)]:.4f}s",
            f"{memory_values[model_names.index(name)]:.2f}MB"
        ])
    
    table = ax8.table(cellText=table_data,
                     colLabels=['Model', 'CER', 'WER', 'Accuracy', 'Time', 'Memory'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add title
    fig.suptitle('OCR Model Evaluation Dashboard', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_error_analysis(error_data: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot error analysis results.
    
    Args:
        error_data: Dictionary containing error analysis data
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different error types
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Character substitution errors
    if 'substitutions' in error_data:
        substitutions = error_data['substitutions']
        chars = list(substitutions.keys())[:10]  # Top 10
        counts = list(substitutions.values())[:10]
        
        ax1.bar(chars, counts, color='red', alpha=0.7)
        ax1.set_title('Most Common Character Substitutions', fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. Insertion errors
    if 'insertions' in error_data:
        insertions = error_data['insertions']
        chars = list(insertions.keys())[:10]  # Top 10
        counts = list(insertions.values())[:10]
        
        ax2.bar(chars, counts, color='blue', alpha=0.7)
        ax2.set_title('Most Common Insertions', fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Deletion errors
    if 'deletions' in error_data:
        deletions = error_data['deletions']
        chars = list(deletions.keys())[:10]  # Top 10
        counts = list(deletions.values())[:10]
        
        ax3.bar(chars, counts, color='green', alpha=0.7)
        ax3.set_title('Most Common Deletions', fontweight='bold')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Error type distribution
    if 'error_types' in error_data:
        error_types = error_data['error_types']
        types = list(error_types.keys())
        counts = list(error_types.values())
        
        ax4.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Error Type Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 