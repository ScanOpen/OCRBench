"""
Main CLI interface for OCR evaluation.
"""

import click
import os
import json
from typing import Dict, Any

from ..evaluation import OCREvaluator
from ..ocr_models import TesseractOCR, PaddleOCR, EasyOCR


@click.group()
def main():
    """OCR Model Evaluation Tool"""
    pass


@main.command()
@click.option('--image', '-i', required=True, help='Path to input image')
@click.option('--ground-truth', '-g', required=True, help='Ground truth text')
@click.option('--output', '-o', default='results.json', help='Output file path')
@click.option('--models', '-m', multiple=True, default=['tesseract', 'paddleocr', 'easyocr'], 
              help='Models to evaluate')
def evaluate(image, ground_truth, output, models):
    """Evaluate OCR models on a single image."""
    
    # Initialize models
    model_instances = {}
    
    if 'tesseract' in models:
        try:
            model_instances['Tesseract'] = TesseractOCR()
        except Exception as e:
            click.echo(f"Warning: Could not initialize Tesseract: {e}")
    
    if 'paddleocr' in models:
        try:
            model_instances['PaddleOCR'] = PaddleOCR()
        except Exception as e:
            click.echo(f"Warning: Could not initialize PaddleOCR: {e}")
    
    if 'easyocr' in models:
        try:
            model_instances['EasyOCR'] = EasyOCR()
        except Exception as e:
            click.echo(f"Warning: Could not initialize EasyOCR: {e}")
    
    if not model_instances:
        click.echo("Error: No models could be initialized")
        return
    
    # Create evaluator
    evaluator = OCREvaluator(model_instances)
    
    # Evaluate
    try:
        results = evaluator.evaluate_single_image(image, ground_truth)
        
        # Save results
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"Results saved to {output}")
        
        # Print summary
        click.echo("\nEvaluation Summary:")
        click.echo("=" * 50)
        for model_name, result in results.items():
            click.echo(f"\n{model_name}:")
            click.echo(f"  Predicted: {result['predicted_text']}")
            click.echo(f"  CER: {result['cer']:.4f}")
            click.echo(f"  WER: {result['wer']:.4f}")
            click.echo(f"  Processing Time: {result['processing_time']:.4f}s")
            
    except Exception as e:
        click.echo(f"Error during evaluation: {e}")


@main.command()
@click.option('--dataset', '-d', required=True, 
              type=click.Choice(['icdar2013', 'iam', 'multilingual']),
              help='Dataset to evaluate')
@click.option('--output-dir', '-o', default='results', help='Output directory')
@click.option('--models', '-m', multiple=True, default=['tesseract', 'paddleocr', 'easyocr'], 
              help='Models to evaluate')
def evaluate_dataset(dataset, output_dir, models):
    """Evaluate OCR models on a specific dataset."""
    
    # Initialize models
    model_instances = {}
    
    if 'tesseract' in models:
        try:
            model_instances['Tesseract'] = TesseractOCR()
            click.echo("✓ Tesseract initialized")
        except Exception as e:
            click.echo(f"✗ Could not initialize Tesseract: {e}")
    
    if 'paddleocr' in models:
        try:
            model_instances['PaddleOCR'] = PaddleOCR()
            click.echo("✓ PaddleOCR initialized")
        except Exception as e:
            click.echo(f"✗ Could not initialize PaddleOCR: {e}")
    
    if 'easyocr' in models:
        try:
            model_instances['EasyOCR'] = EasyOCR()
            click.echo("✓ EasyOCR initialized")
        except Exception as e:
            click.echo(f"✗ Could not initialize EasyOCR: {e}")
    
    if not model_instances:
        click.echo("Error: No models could be initialized")
        return
    
    # Create evaluator
    evaluator = OCREvaluator(model_instances)
    
    # Evaluate dataset
    try:
        click.echo(f"\nEvaluating {dataset} dataset...")
        raw_results = evaluator.evaluate_dataset_by_name(dataset)
        
        # Aggregate results
        results = evaluator._aggregate_results(raw_results)
        
        # Generate report
        report_path = evaluator.generate_report(output_dir, results)
        click.echo(f"Report generated: {report_path}")
        
        # Print summary
        click.echo("\nEvaluation Summary:")
        click.echo("=" * 50)
        for model_name, metrics in results.items():
            click.echo(f"\n{model_name}:")
            click.echo(f"  Total Images: {metrics.get('total_images', 0)}")
            click.echo(f"  Success Rate: {metrics.get('successful_images', 0) / metrics.get('total_images', 1) * 100:.2f}%")
            click.echo(f"  CER: {metrics.get('cer_mean', 0):.4f} ± {metrics.get('cer_std', 0):.4f}")
            click.echo(f"  WER: {metrics.get('wer_mean', 0):.4f} ± {metrics.get('wer_std', 0):.4f}")
            click.echo(f"  Avg Processing Time: {metrics.get('avg_processing_time', 0):.4f}s")
        
        # Find best model
        best_model = evaluator.get_best_model('cer_mean')
        if best_model:
            click.echo(f"\nBest model (by CER): {best_model}")
            
    except Exception as e:
        click.echo(f"Error during evaluation: {e}")


@main.command()
@click.option('--images-dir', '-i', required=True, help='Directory containing images')
@click.option('--ground-truth-file', '-g', required=True, help='Path to ground truth JSON file')
@click.option('--output-dir', '-o', default='results', help='Output directory')
@click.option('--models', '-m', multiple=True, default=['tesseract', 'paddleocr', 'easyocr'], 
              help='Models to evaluate')
def evaluate_custom_dataset(images_dir, ground_truth_file, output_dir, models):
    """Evaluate OCR models on a dataset."""
    
    # Check inputs
    if not os.path.exists(images_dir):
        click.echo(f"Error: Images directory does not exist: {images_dir}")
        return
    
    if not os.path.exists(ground_truth_file):
        click.echo(f"Error: Ground truth file does not exist: {ground_truth_file}")
        return
    
    # Initialize models
    model_instances = {}
    
    if 'tesseract' in models:
        try:
            model_instances['Tesseract'] = TesseractOCR()
            click.echo("✓ Tesseract initialized")
        except Exception as e:
            click.echo(f"✗ Could not initialize Tesseract: {e}")
    
    if 'paddleocr' in models:
        try:
            model_instances['PaddleOCR'] = PaddleOCR()
            click.echo("✓ PaddleOCR initialized")
        except Exception as e:
            click.echo(f"✗ Could not initialize PaddleOCR: {e}")
    
    if 'easyocr' in models:
        try:
            model_instances['EasyOCR'] = EasyOCR()
            click.echo("✓ EasyOCR initialized")
        except Exception as e:
            click.echo(f"✗ Could not initialize EasyOCR: {e}")
    
    if not model_instances:
        click.echo("Error: No models could be initialized")
        return
    
    # Create evaluator
    evaluator = OCREvaluator(model_instances)
    
    # Evaluate dataset
    try:
        click.echo(f"\nEvaluating dataset...")
        results = evaluator.evaluate_dataset(images_dir, ground_truth_file)
        
        # Generate report
        report_path = evaluator.generate_report(output_dir, results)
        click.echo(f"Report generated: {report_path}")
        
        # Print summary
        click.echo("\nEvaluation Summary:")
        click.echo("=" * 50)
        for model_name, metrics in results.items():
            click.echo(f"\n{model_name}:")
            click.echo(f"  Total Images: {metrics.get('total_images', 0)}")
            click.echo(f"  Success Rate: {metrics.get('successful_images', 0) / metrics.get('total_images', 1) * 100:.2f}%")
            click.echo(f"  CER: {metrics.get('cer_mean', 0):.4f} ± {metrics.get('cer_std', 0):.4f}")
            click.echo(f"  WER: {metrics.get('wer_mean', 0):.4f} ± {metrics.get('wer_std', 0):.4f}")
            click.echo(f"  Avg Processing Time: {metrics.get('avg_processing_time', 0):.4f}s")
        
        # Find best model
        best_model = evaluator.get_best_model('cer_mean')
        if best_model:
            click.echo(f"\nBest model (by CER): {best_model}")
            
    except Exception as e:
        click.echo(f"Error during evaluation: {e}")


@main.command()
@click.option('--results-dir', '-r', required=True, help='Directory containing evaluation results')
@click.option('--output', '-o', default='comparison_report.html', help='Output HTML report path')
def generate_report(results_dir, output):
    """Generate an HTML comparison report from evaluation results."""
    
    if not os.path.exists(results_dir):
        click.echo(f"Error: Results directory does not exist: {results_dir}")
        return
    
    # Load results
    report_file = os.path.join(results_dir, 'evaluation_report.json')
    if not os.path.exists(report_file):
        click.echo(f"Error: No evaluation report found in {results_dir}")
        return
    
    try:
        with open(report_file, 'r') as f:
            results = json.load(f)
        
        # Generate HTML report
        html_content = _generate_html_report(results)
        
        with open(output, 'w') as f:
            f.write(html_content)
        
        click.echo(f"HTML report generated: {output}")
        
    except Exception as e:
        click.echo(f"Error generating report: {e}")


def _generate_html_report(results: Dict[str, Any]) -> str:
    """Generate HTML report from evaluation results."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCR Model Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .model-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .metric { margin: 10px 0; }
            .metric-label { font-weight: bold; }
            .metric-value { color: #007bff; }
            .best { background-color: #d4edda; border-color: #c3e6cb; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>OCR Model Evaluation Report</h1>
            <p>Comparative analysis of OCR model performance</p>
        </div>
    """
    
    # Find best model for each metric
    metrics_to_compare = ['cer_mean', 'wer_mean', 'accuracy_mean', 'avg_processing_time']
    
    for model_name, metrics in results.items():
        html += f"""
        <div class="model-section">
            <h2>{model_name}</h2>
            <div class="metric">
                <span class="metric-label">Total Images:</span>
                <span class="metric-value">{metrics.get('total_images', 0)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Success Rate:</span>
                <span class="metric-value">{metrics.get('successful_images', 0) / metrics.get('total_images', 1) * 100:.2f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Character Error Rate (CER):</span>
                <span class="metric-value">{metrics.get('cer_mean', 0):.4f} ± {metrics.get('cer_std', 0):.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Word Error Rate (WER):</span>
                <span class="metric-value">{metrics.get('wer_mean', 0):.4f} ± {metrics.get('wer_std', 0):.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Accuracy:</span>
                <span class="metric-value">{metrics.get('accuracy_mean', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Average Processing Time:</span>
                <span class="metric-value">{metrics.get('avg_processing_time', 0):.4f}s</span>
            </div>
            <div class="metric">
                <span class="metric-label">Average Memory Used:</span>
                <span class="metric-value">{metrics.get('avg_memory_used', 0):.2f}MB</span>
            </div>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    return html


if __name__ == '__main__':
    main() 