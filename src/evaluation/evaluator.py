"""
OCR Evaluator

Main class for evaluating OCR models and comparing their performance.
"""

import os
import json
import time
import glob
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
from tqdm import tqdm

from ..ocr_models import BaseOCRModel
from .metrics import calculate_all_metrics, aggregate_metrics


class OCREvaluator:
    """
    Main evaluator class for comparing OCR models with enhanced memory tracking.
    """
    
    def __init__(self, models: Dict[str, BaseOCRModel], config_path: str = "config/evaluation_config.yaml"):
        """
        Initialize the OCR evaluator.
        
        Args:
            models: Dictionary mapping model names to OCR model instances
            config_path: Path to the configuration file
        """
        self.models = models
        self.results = {}
        self.config = self._load_config(config_path)
        self.datasets = self._initialize_datasets()
        
    def evaluate_single_image(self, image_path: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate all models on a single image with enhanced memory tracking.
        
        Args:
            image_path: Path to the image file
            ground_truth: Ground truth text
            
        Returns:
            Dictionary containing results for all models
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {}
        
        for model_name, model in self.models.items():
                try:
                    # Perform OCR with enhanced memory tracking
                    ocr_result = model.recognize_text_with_metrics(image)
                    predicted_text = ocr_result['text']
                    
                    # Calculate metrics
                    metrics = calculate_all_metrics(predicted_text, ground_truth)
                    
                    # Combine results
                    result = {
                        'predicted_text': predicted_text,
                        'ground_truth': ground_truth,
                        'processing_time': ocr_result['processing_time'],
                        'success': ocr_result['success'],
                        **metrics
                    }
                    
                    if not ocr_result['success']:
                        result['error'] = ocr_result.get('error', 'Unknown error')
                    
                    results[model_name] = result
                    
                except Exception as e:
                    results[model_name] = {
                        'predicted_text': '',
                        'ground_truth': ground_truth,
                        'processing_time': 0.0,
                        'success': False,
                        'error': str(e),
                        'cer': 1.0,
                        'wer': 1.0,
                        'accuracy': 0.0,
                        'edit_distance': len(ground_truth),
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0,
                        'length_ratio': 0.0
                    }
            

        
        return results
    
    def evaluate_dataset(self, images_dir: str, ground_truth_file: str) -> Dict[str, Any]:
        """
        Evaluate all models on a dataset with enhanced memory tracking.
        
        Args:
            images_dir: Directory containing images
            ground_truth_file: Path to ground truth file
            
        Returns:
            Dictionary containing aggregated results for all models
        """
        # Load ground truth data
        ground_truth_data = self._load_ground_truth(ground_truth_file)
        
        # Get list of image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")
        
        # Initialize results storage
        all_results = {model_name: [] for model_name in self.models.keys()}
        
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(images_dir, image_file)
            
            # Get ground truth for this image
            if image_file in ground_truth_data:
                ground_truth = ground_truth_data[image_file]
            else:
                print(f"Warning: No ground truth found for {image_file}")
                continue
            
            # Evaluate all models on this image
            results = self.evaluate_single_image(image_path, ground_truth)
            
            # Store results
            for model_name, result in results.items():
                all_results[model_name].append(result)
        
        # Aggregate results
        aggregated_results = {}
        for model_name, results in all_results.items():
            if results:
                aggregated_results[model_name] = self._aggregate_results(results)
                aggregated_results[model_name]['total_images'] = len(results)
                aggregated_results[model_name]['successful_images'] = sum(1 for r in results if r['success'])
        
        self.results = aggregated_results
        return aggregated_results
    

    
    def generate_report(self, output_dir: str, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive evaluation report with ground truth and OCR text comparisons.
        
        Args:
            output_dir: Directory to save the report
            results: Evaluation results (uses self.results if None)
            
        Returns:
            Path to the generated report
        """
        if results is None:
            results = self.results
        
        if not results:
            raise ValueError("No results available for report generation")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if results are aggregated (have 'total_images' key) or raw per-image results
        is_aggregated = False
        if results and isinstance(results, dict):
            # Check if this looks like aggregated results
            for model_name, model_data in results.items():
                if isinstance(model_data, dict) and 'total_images' in model_data:
                    is_aggregated = True
                    break
        
        if is_aggregated:
            # Results are already aggregated, use raw results from self.results
            raw_results = self.results
        else:
            # Results are raw per-image results
            raw_results = results
        
        # Generate detailed report with text comparisons
        detailed_report_path = os.path.join(output_dir, 'detailed_evaluation_report.json')
        self._generate_detailed_report(detailed_report_path, raw_results)
        
        # Generate summary
        summary_path = os.path.join(output_dir, 'summary.txt')
        self._generate_summary(summary_path, results)  # Use aggregated results for summary
        
        # Generate text comparison report
        text_comparison_path = os.path.join(output_dir, 'text_comparison_report.txt')
        self._generate_text_comparison_report(text_comparison_path, raw_results)
        
        return detailed_report_path
    
    def _generate_detailed_report(self, output_path: str, results: Dict[str, Any]) -> None:
        """
        Generate a detailed evaluation report including ground truth and predicted text.
        
        Args:
            output_path: Path to save the detailed report
            results: Evaluation results
        """
        detailed_results = {
            'summary': {},
            'detailed_results': results,
            'text_comparisons': {}
        }
        
        # Add aggregated summary
        if results:
            # Check if results are already aggregated
            is_aggregated = False
            for model_name, model_data in results.items():
                if isinstance(model_data, dict) and 'total_images' in model_data:
                    is_aggregated = True
                    break
            
            if is_aggregated:
                # Results are already aggregated, use them directly
                detailed_results['summary'] = results
            else:
                # Results are raw, need to aggregate them
                detailed_results['summary'] = self._aggregate_results(results)
        
        # Add text comparisons for each image
        for image_name, image_results in results.items():
            detailed_results['text_comparisons'][image_name] = {}
            for model_name, model_result in image_results.items():
                detailed_results['text_comparisons'][image_name][model_name] = {
                    'ground_truth': model_result.get('ground_truth', ''),
                    'predicted_text': model_result.get('predicted_text', ''),
                    'cer': model_result.get('cer', 1.0),
                    'wer': model_result.get('wer', 1.0),
                    'processing_time': model_result.get('processing_time', 0.0),
                    'success': model_result.get('success', False)
                }
        
        with open(output_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
    
    def _generate_text_comparison_report(self, output_path: str, results: Dict[str, Any]) -> None:
        """
        Generate a human-readable text comparison report.
        
        Args:
            output_path: Path to save the text comparison report
            results: Evaluation results
        """
        with open(output_path, 'w') as f:
            f.write("OCR Text Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Group by model for easier comparison
            model_names = set()
            for image_results in results.values():
                model_names.update(image_results.keys())
            
            # For each image, show all models' predictions
            for image_name, image_results in sorted(results.items()):
                f.write(f"Image: {image_name}\n")
                f.write("-" * 30 + "\n")
                
                # Check if image_results is valid
                if not isinstance(image_results, dict):
                    f.write(f"Error: Invalid results format for {image_name}\n\n")
                    continue
                
                # Get ground truth from first model (should be same for all)
                ground_truth = ""
                for model_result in image_results.values():
                    if isinstance(model_result, dict) and 'ground_truth' in model_result:
                        ground_truth = model_result['ground_truth']
                        break
                
                f.write(f"Ground Truth: {ground_truth}\n\n")
                
                # Show each model's prediction
                for model_name in sorted(model_names):
                    if model_name in image_results:
                        model_result = image_results[model_name]
                        if not isinstance(model_result, dict):
                            f.write(f"{model_name}: Error - Invalid result format\n\n")
                            continue
                            
                        predicted_text = model_result.get('predicted_text', '')
                        cer = model_result.get('cer', 1.0)
                        wer = model_result.get('wer', 1.0)
                        processing_time = model_result.get('processing_time', 0.0)
                        success = model_result.get('success', False)
                        
                        f.write(f"{model_name}:\n")
                        f.write(f"  Predicted: {predicted_text}\n")
                        f.write(f"  CER: {cer:.4f}, WER: {wer:.4f}\n")
                        f.write(f"  Processing Time: {processing_time:.4f}s\n")
                        f.write(f"  Success: {success}\n\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
    
    def generate_sample_comparisons(self, output_path: str, num_samples: int = 10) -> None:
        """
        Generate a report with sample text comparisons for quick analysis.
        
        Args:
            output_path: Path to save the sample comparisons
            num_samples: Number of sample images to include
        """
        if not self.results:
            raise ValueError("No results available for sample generation")
        
        # Get sample images (first n images)
        sample_images = list(self.results.keys())[:num_samples]
        
        with open(output_path, 'w') as f:
            f.write("Sample OCR Text Comparisons\n")
            f.write("=" * 40 + "\n\n")
            
            for i, image_name in enumerate(sample_images, 1):
                f.write(f"Sample {i}: {image_name}\n")
                f.write("-" * 30 + "\n")
                
                image_results = self.results[image_name]
                
                # Check if image_results is valid
                if not isinstance(image_results, dict):
                    f.write(f"Error: Invalid results format for {image_name}\n\n")
                    continue
                
                # Get ground truth
                ground_truth = ""
                for model_result in image_results.values():
                    if isinstance(model_result, dict) and 'ground_truth' in model_result:
                        ground_truth = model_result['ground_truth']
                        break
                
                f.write(f"Ground Truth: {ground_truth}\n\n")
                
                # Show predictions for each model
                for model_name, model_result in sorted(image_results.items()):
                    if not isinstance(model_result, dict):
                        f.write(f"{model_name}: Error - Invalid result format\n\n")
                        continue
                        
                    predicted_text = model_result.get('predicted_text', '')
                    cer = model_result.get('cer', 1.0)
                    wer = model_result.get('wer', 1.0)
                    
                    f.write(f"{model_name}:\n")
                    f.write(f"  {predicted_text}\n")
                    f.write(f"  CER: {cer:.4f}, WER: {wer:.4f}\n\n")
                
                f.write("\n" + "=" * 40 + "\n\n")
    
    def get_text_comparison_data(self, image_name: str = None) -> Dict[str, Any]:
        """
        Get text comparison data for analysis.
        
        Args:
            image_name: Specific image name (if None, returns all)
            
        Returns:
            Dictionary with text comparison data
        """
        if not self.results:
            return {}
        
        if image_name:
            if image_name not in self.results:
                return {}
            return {image_name: self.results[image_name]}
        
        return self.results
    
    def _generate_summary(self, output_path: str, results: Dict[str, Any]) -> None:
        """
        Generate a text summary of the evaluation results.
        
        Args:
            output_path: Path to save the summary
            results: Evaluation results
        """
        with open(output_path, 'w') as f:
            f.write("OCR Model Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            for model_name, metrics in results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Total Images: {metrics.get('total_images', 0)}\n")
                f.write(f"  Successful Images: {metrics.get('successful_images', 0)}\n")
                f.write(f"  Success Rate: {metrics.get('successful_images', 0) / metrics.get('total_images', 1) * 100:.2f}%\n")
                f.write(f"  Average Processing Time: {metrics.get('avg_processing_time', 0):.4f}s\n")
            
            # Accuracy metrics
            f.write("\n\nAccuracy Metrics:\n")
            f.write("-" * 20 + "\n")
            for model_name, metrics in results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Character Error Rate (CER): {metrics.get('cer_mean', 0):.4f} ± {metrics.get('cer_std', 0):.4f}\n")
                f.write(f"  Word Error Rate (WER): {metrics.get('wer_mean', 0):.4f} ± {metrics.get('wer_std', 0):.4f}\n")
                f.write(f"  Accuracy: {metrics.get('accuracy_mean', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}\n")
                f.write(f"  Precision: {metrics.get('precision_mean', 0):.4f} ± {metrics.get('precision_std', 0):.4f}\n")
                f.write(f"  Recall: {metrics.get('recall_mean', 0):.4f} ± {metrics.get('recall_std', 0):.4f}\n")
                f.write(f"  F1 Score: {metrics.get('f1_score_mean', 0):.4f} ± {metrics.get('f1_score_std', 0):.4f}\n")
    
    def compare_models(self, metric: str = 'cer_mean') -> List[tuple]:
        """
        Compare models based on a specific metric.
        
        Args:
            metric: Metric to compare (default: 'cer_mean')
            
        Returns:
            List of (model_name, metric_value) tuples sorted by performance
        """
        if not self.results:
            return []
        
        comparisons = []
        for model_name, metrics in self.results.items():
            if metric in metrics:
                comparisons.append((model_name, metrics[metric]))
        
        # Sort by metric value (lower is better for error rates)
        if 'error' in metric.lower() or 'cer' in metric.lower() or 'wer' in metric.lower():
            comparisons.sort(key=lambda x: x[1])
        else:
            comparisons.sort(key=lambda x: x[1], reverse=True)
        
        return comparisons
    
    def get_best_model(self, metric: str = 'cer_mean') -> Optional[str]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Name of the best performing model
        """
        comparisons = self.compare_models(metric)
        return comparisons[0][0] if comparisons else None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        import yaml
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize dataset configurations.
        
        Returns:
            Dictionary of dataset configurations
        """
        return self.config.get('datasets', {})
    
    def evaluate_dataset_by_name(self, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate models on a specific dataset by name.
        
        Args:
            dataset_name: Name of the dataset (icdar2013, iam, multilingual)
            
        Returns:
            Evaluation results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        dataset_config = self.datasets[dataset_name]
        images_dir = os.path.join(dataset_config['path'], 'images')
        ground_truth_dir = os.path.join(dataset_config['path'], 'ground_truth')
        
        return self.evaluate_dataset_with_config(images_dir, ground_truth_dir, dataset_config)
    
    def evaluate_dataset_with_config(self, images_dir: str, ground_truth_dir: str, 
                                   dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all models on a dataset with specific configuration.
        
        Args:
            images_dir: Directory containing image files
            ground_truth_dir: Directory containing ground truth files
            dataset_config: Dataset-specific configuration
            
        Returns:
            Dictionary containing evaluation results
        """
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        if not os.path.exists(ground_truth_dir):
            raise ValueError(f"Ground truth directory not found: {ground_truth_dir}")
        
        # Get image files (recursively search subfolders)
        image_files = []
        for ext in dataset_config.get('supported_formats', ['.png', '.jpg', '.jpeg']):
            # Search recursively in subfolders
            pattern = os.path.join(images_dir, "**", f"*{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
        
        if not image_files:
            raise ValueError(f"No image files found in {images_dir} or its subfolders")
        
        # Check if this is a single ground truth file format (like ICDAR 2013)
        gt_file = dataset_config.get('ground_truth_file')
        if gt_file:
            gt_file_path = os.path.join(ground_truth_dir, gt_file)
            if not os.path.exists(gt_file_path):
                raise ValueError(f"Ground truth file not found: {gt_file_path}")
            if gt_file == 'lines.txt':
                ground_truth_map = self._load_iam_lines_ground_truth(gt_file_path)
                print("[DEBUG] Sample image filenames:", [os.path.basename(f) for f in image_files[:10]])
                print("[DEBUG] Sample ground truth keys:", list(ground_truth_map.keys())[:10])
                # Build a mapping from lowercase base name to transcription
                base_gt_map = {k.lower().rsplit('.', 1)[0]: v for k, v in ground_truth_map.items()}
            else:
                ground_truth_map = self._load_icdar2013_ground_truth(gt_file_path)
            results = {}
            for image_file in tqdm(image_files, desc=f"Evaluating {dataset_config.get('name', 'dataset')}"):
                image_name = os.path.basename(image_file)
                image_base = image_name.lower().rsplit('.', 1)[0]
                # For IAM, use base_gt_map; for others, use normal map
                if gt_file == 'lines.txt':
                    if image_base not in base_gt_map:
                        print(f"Warning: No ground truth found for {image_name} (base: {image_base})")
                        continue
                    ground_truth = base_gt_map[image_base]
                else:
                    if image_name not in ground_truth_map:
                        print(f"Warning: No ground truth found for {image_name}")
                        continue
                    ground_truth = ground_truth_map[image_name]
                image_results = self.evaluate_single_image(image_file, ground_truth)
                results[image_name] = image_results
            return results
        
        # Original logic for individual ground truth files
        results = {}
        for image_file in tqdm(image_files, desc=f"Evaluating {dataset_config.get('name', 'dataset')}"):
            # Find corresponding ground truth file
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            gt_format = dataset_config.get('ground_truth_format', 'txt')
            
            # Try different ground truth file locations
            gt_file = None
            if gt_format == 'txt':
                # Get the relative path from images directory
                rel_path = os.path.relpath(image_file, images_dir)
                # Create the corresponding ground truth path
                gt_path = os.path.join(ground_truth_dir, os.path.dirname(rel_path), f"{base_name}.txt")
                
                # Try multiple possible locations
                possible_gt_files = [
                    gt_path,  # Exact mirror of image path
                    os.path.join(ground_truth_dir, f"{base_name}.txt"),
                    os.path.join(ground_truth_dir, "lines", f"{base_name}.txt"),
                    os.path.join(ground_truth_dir, base_name.split('/')[-1] + ".txt"),
                ]
                for possible_file in possible_gt_files:
                    if os.path.exists(possible_file):
                        gt_file = possible_file
                        break
            elif gt_format == 'xml':
                gt_file = os.path.join(ground_truth_dir, f"{base_name}.xml")
            elif gt_format == 'json':
                gt_file = os.path.join(ground_truth_dir, f"{base_name}.json")
            else:
                raise ValueError(f"Unsupported ground truth format: {gt_format}")
            
            if not gt_file or not os.path.exists(gt_file):
                print(f"Warning: Ground truth file not found for {image_file}")
                print(f"   Expected: {base_name}.{gt_format}")
                continue
            
            # Load ground truth
            ground_truth = self._load_ground_truth(gt_file, gt_format)
            
            # Evaluate image
            image_results = self.evaluate_single_image(image_file, ground_truth)
            
            # Store results
            image_name = os.path.basename(image_file)
            results[image_name] = image_results
        
        return results
    
    def _load_ground_truth(self, gt_file: str, format_type: str) -> str:
        """
        Load ground truth from file based on format.
        
        Args:
            gt_file: Path to ground truth file
            format_type: Format type (txt, xml, json)
            
        Returns:
            Ground truth text
        """
        if format_type == 'txt':
            with open(gt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        elif format_type == 'xml':
            import xml.etree.ElementTree as ET
            tree = ET.parse(gt_file)
            root = tree.getroot()
            # Extract text from XML (implementation depends on XML structure)
            # This is a simplified version - adjust based on actual XML format
            text_elements = root.findall('.//text')
            return ' '.join([elem.text for elem in text_elements if elem.text])
        
        elif format_type == 'json':
            with open(gt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract text from JSON (implementation depends on JSON structure)
                # This is a simplified version - adjust based on actual JSON format
                if isinstance(data, dict):
                    return data.get('text', '')
                elif isinstance(data, list):
                    return ' '.join([item.get('text', '') for item in data if isinstance(item, dict)])
                else:
                    return str(data)
        
        else:
            raise ValueError(f"Unsupported ground truth format: {format_type}")
    
    def _load_icdar2013_ground_truth(self, gt_file: str) -> Dict[str, str]:
        """
        Load ICDAR 2013 ground truth from the single txt file.
        
        Args:
            gt_file: Path to the gt.txt file
            
        Returns:
            Dictionary mapping image filenames to ground truth text
        """
        ground_truth_map = {}
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse the format: filename.png, "ground_truth_text"
                if ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        filename = parts[0].strip()
                        text = parts[1].strip()
                        
                        # Remove quotes if present
                        if text.startswith('"') and text.endswith('"'):
                            text = text[1:-1]
                        
                        ground_truth_map[filename] = text
        
        return ground_truth_map
    
    def _load_iam_lines_ground_truth(self, gt_file: str) -> Dict[str, str]:
        """
        Load IAM ground truth from the single lines.txt file.
        Args:
            gt_file: Path to the lines.txt file
        Returns:
            Dictionary mapping image filenames (e.g., a01-000u-00.png) to ground truth text
        """
        ground_truth_map = {}
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue  # Not a valid line
                line_id = parts[0]
                # Transcription is everything after the 8th field, joined and with | replaced by space
                transcription = ' '.join(parts[8:]).replace('|', ' ')
                # Map to image filename (e.g., a01-000u-00.png)
                image_filename = f"{line_id}.png"
                ground_truth_map[image_filename] = transcription
        return ground_truth_map

    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate individual image results into model-level statistics.
        
        Args:
            results: Dictionary with image names as keys and model results as values
            
        Returns:
            Aggregated results for each model
        """
        if not results:
            return {}
        
        # Initialize aggregated results for each model
        model_names = set()
        for image_results in results.values():
            model_names.update(image_results.keys())
        
        aggregated = {model_name: {
            'total_images': 0,
            'successful_images': 0,
            'processing_times': [],
            'cer_values': [],
            'wer_values': [],
            'accuracy_values': [],
            'precision_values': [],
            'recall_values': [],
            'f1_score_values': [],
            'edit_distances': [],
            'length_ratios': []
        } for model_name in model_names}
        
        # Aggregate results from each image
        for image_name, image_results in results.items():
            for model_name, model_result in image_results.items():
                if model_name not in aggregated:
                    continue
                
                aggregated[model_name]['total_images'] += 1
                
                if model_result.get('success', False):
                    aggregated[model_name]['successful_images'] += 1
                
                # Collect metrics
                aggregated[model_name]['processing_times'].append(model_result.get('processing_time', 0))
                aggregated[model_name]['cer_values'].append(model_result.get('cer', 1.0))
                aggregated[model_name]['wer_values'].append(model_result.get('wer', 1.0))
                aggregated[model_name]['accuracy_values'].append(model_result.get('accuracy', 0.0))
                aggregated[model_name]['precision_values'].append(model_result.get('precision', 0.0))
                aggregated[model_name]['recall_values'].append(model_result.get('recall', 0.0))
                aggregated[model_name]['f1_score_values'].append(model_result.get('f1_score', 0.0))
                aggregated[model_name]['edit_distances'].append(model_result.get('edit_distance', 0))
                aggregated[model_name]['length_ratios'].append(model_result.get('length_ratio', 0.0))
        
        # Calculate statistics
        import numpy as np
        for model_name, metrics in aggregated.items():
            if metrics['total_images'] > 0:
                # Processing time
                metrics['avg_processing_time'] = np.mean(metrics['processing_times'])
                
                # Error rates and accuracy
                metrics['cer_mean'] = np.mean(metrics['cer_values'])
                metrics['cer_std'] = np.std(metrics['cer_values'])
                metrics['wer_mean'] = np.mean(metrics['wer_values'])
                metrics['wer_std'] = np.std(metrics['wer_values'])
                metrics['accuracy_mean'] = np.mean(metrics['accuracy_values'])
                metrics['accuracy_std'] = np.std(metrics['accuracy_values'])
                
                # Precision, recall, F1
                metrics['precision_mean'] = np.mean(metrics['precision_values'])
                metrics['precision_std'] = np.std(metrics['precision_values'])
                metrics['recall_mean'] = np.mean(metrics['recall_values'])
                metrics['recall_std'] = np.std(metrics['recall_values'])
                metrics['f1_score_mean'] = np.mean(metrics['f1_score_values'])
                metrics['f1_score_std'] = np.std(metrics['f1_score_values'])
                
                # Other metrics
                metrics['avg_edit_distance'] = np.mean(metrics['edit_distances'])
                metrics['avg_length_ratio'] = np.mean(metrics['length_ratios'])
        
        return aggregated 