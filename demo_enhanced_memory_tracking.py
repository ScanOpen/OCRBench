#!/usr/bin/env python3
"""
Enhanced Memory Tracking Demo

Demonstrates the enhanced memory tracking concepts using tracemalloc and psutil.
"""

import tracemalloc
import time
import psutil
import json
import os
import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from evaluation.memory_tracker import MemoryTracker, create_memory_tracker, get_system_memory_info
    from ocr_models.base import BaseOCRModel
    from evaluation.evaluator import OCREvaluator
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    print("Note: Some dependencies are not available. Running simplified demo.")
    DEPENDENCIES_AVAILABLE = False


class SimpleMemoryTracker:
    """Simplified memory tracker using only tracemalloc and psutil."""
    
    def __init__(self):
        self.tracemalloc_started = False
    
    def start_tracemalloc(self):
        """Start tracemalloc if not already started."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_started = True
    
    def stop_tracemalloc(self):
        """Stop tracemalloc if we started it."""
        if self.tracemalloc_started and tracemalloc.is_tracing():
            tracemalloc.stop()
            self.tracemalloc_started = False
    
    def get_psutil_memory(self) -> Dict[str, float]:
        """Get memory usage using psutil."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent(),     # Memory usage percentage
            'available': psutil.virtual_memory().available / 1024 / 1024  # Available system memory in MB
        }
    
    def get_tracemalloc_memory(self) -> Dict[str, Any]:
        """Get memory usage using tracemalloc."""
        if not tracemalloc.is_tracing():
            return {'error': 'Tracemalloc not running'}
        
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'current_bytes': current,
            'peak_bytes': peak
        }
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive memory snapshot."""
        snapshot = {
            'timestamp': time.time(),
            'psutil': self.get_psutil_memory()
        }
        
        if tracemalloc.is_tracing():
            snapshot['tracemalloc'] = self.get_tracemalloc_memory()
        
        return snapshot
    
    @contextmanager
    def track_memory_usage(self, operation_name: str = "operation"):
        """Context manager for tracking memory usage during an operation."""
        # Force garbage collection before starting
        import gc
        gc.collect()
        
        # Get initial memory state
        initial_snapshot = self.get_memory_snapshot()
        start_time = time.time()
        
        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        try:
            yield {
                'operation_name': operation_name,
                'initial_snapshot': initial_snapshot,
                'start_time': start_time
            }
        finally:
            # Get final memory state
            end_time = time.time()
            final_snapshot = self.get_memory_snapshot()
            
            # Calculate differences
            duration = end_time - start_time
            
            # Calculate memory differences
            memory_diff = self._calculate_memory_difference(
                initial_snapshot, final_snapshot
            )
            
            # Stop tracemalloc if we started it
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            
            # Force garbage collection after operation
            gc.collect()
            
            # Update the yield dictionary with results
            memory_diff.update({
                'duration': duration,
                'final_snapshot': final_snapshot
            })
    
    def _calculate_memory_difference(self, initial: Dict[str, Any], final: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate memory usage differences between snapshots."""
        diff = {
            'psutil_diff': {
                'rss_diff_mb': final['psutil']['rss'] - initial['psutil']['rss'],
                'vms_diff_mb': final['psutil']['vms'] - initial['psutil']['vms'],
                'percent_diff': final['psutil']['percent'] - initial['psutil']['percent']
            }
        }
        
        if 'tracemalloc' in initial and 'tracemalloc' in final:
            if 'error' not in initial['tracemalloc'] and 'error' not in final['tracemalloc']:
                diff['tracemalloc_diff'] = {
                    'current_diff_mb': final['tracemalloc']['current_mb'] - initial['tracemalloc']['current_mb'],
                    'peak_diff_mb': final['tracemalloc']['peak_mb'] - initial['tracemalloc']['peak_mb']
                }
        
        return diff
    
    def get_top_memory_allocations(self, limit: int = 5) -> list:
        """Get top memory allocations using tracemalloc."""
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        allocations = []
        for stat in top_stats[:limit]:
            allocations.append({
                'filename': stat.traceback.format()[0],
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
        
        return allocations


def demo_basic_memory_tracking():
    """Demonstrate basic memory tracking."""
    print("=== Basic Memory Tracking Demo ===")
    
    tracker = SimpleMemoryTracker()
    
    # Test basic memory snapshot
    snapshot = tracker.get_memory_snapshot()
    print(f"Initial memory snapshot:")
    print(f"  RSS: {snapshot['psutil']['rss']:.2f} MB")
    print(f"  VMS: {snapshot['psutil']['vms']:.2f} MB")
    print(f"  Memory %: {snapshot['psutil']['percent']:.2f}%")
    
    # Test memory tracking context manager
    with tracker.track_memory_usage("memory_demo_operation") as memory_info:
        # Simulate some memory-intensive operation
        print("  Performing memory-intensive operation...")
        large_list = [i for i in range(1000000)]  # ~8MB of memory
        time.sleep(0.1)  # Simulate processing time
        
        print(f"  Memory info during operation:")
        print(f"    Operation: {memory_info['operation_name']}")
    
    # Test top memory allocations
    top_allocations = tracker.get_top_memory_allocations(limit=3)
    print(f"  Top memory allocations:")
    for i, allocation in enumerate(top_allocations, 1):
        print(f"    {i}. {allocation['filename']}: {allocation['size_mb']:.2f} MB ({allocation['count']} allocations)")
    
    print()


def demo_memory_patterns():
    """Demonstrate different memory usage patterns."""
    print("=== Memory Patterns Demo ===")
    
    tracker = SimpleMemoryTracker()
    
    # Pattern 1: Memory allocation
    print("Pattern 1: Memory Allocation")
    with tracker.track_memory_usage("memory_allocation") as memory_info:
        data = [i for i in range(500000)]  # Allocate memory
        time.sleep(0.05)
    
    if 'tracemalloc_diff' in memory_info:
        print(f"  Memory allocated: {memory_info['tracemalloc_diff']['current_diff_mb']:.2f} MB")
    
    # Pattern 2: Memory deallocation
    print("Pattern 2: Memory Deallocation")
    with tracker.track_memory_usage("memory_deallocation") as memory_info:
        del data  # Deallocate memory
        import gc
        gc.collect()  # Force garbage collection
        time.sleep(0.05)
    
    if 'tracemalloc_diff' in memory_info:
        print(f"  Memory deallocated: {memory_info['tracemalloc_diff']['current_diff_mb']:.2f} MB")
    
    # Pattern 3: No net change
    print("Pattern 3: No Net Memory Change")
    with tracker.track_memory_usage("no_net_change") as memory_info:
        temp_data = [i for i in range(100000)]
        del temp_data
        time.sleep(0.05)
    
    if 'tracemalloc_diff' in memory_info:
        print(f"  Net memory change: {memory_info['tracemalloc_diff']['current_diff_mb']:.2f} MB")
    
    print()


def demo_system_memory_info():
    """Demonstrate system memory information."""
    print("=== System Memory Information ===")
    
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    system_info = {
        'total_mb': vm.total / 1024 / 1024,
        'available_mb': vm.available / 1024 / 1024,
        'used_mb': vm.used / 1024 / 1024,
        'percent_used': vm.percent,
        'swap_total_mb': swap.total / 1024 / 1024,
        'swap_used_mb': swap.used / 1024 / 1024,
        'swap_percent': swap.percent
    }
    
    print(f"System Memory:")
    print(f"  Total: {system_info['total_mb']:.0f} MB")
    print(f"  Available: {system_info['available_mb']:.0f} MB")
    print(f"  Used: {system_info['used_mb']:.0f} MB ({system_info['percent_used']:.1f}%)")
    print(f"  Swap Total: {system_info['swap_total_mb']:.0f} MB")
    print(f"  Swap Used: {system_info['swap_used_mb']:.0f} MB ({system_info['swap_percent']:.1f}%)")
    print()


def demo_memory_efficiency():
    """Demonstrate memory efficiency calculations."""
    print("=== Memory Efficiency Demo ===")
    
    tracker = SimpleMemoryTracker()
    
    # Simulate OCR-like operations with different text lengths
    test_cases = [
        ("Short text", "Hello"),
        ("Medium text", "This is a medium length text for testing memory efficiency."),
        ("Long text", "This is a much longer text that contains many more characters and words to demonstrate how memory usage scales with text length and complexity." * 10)
    ]
    
    for text_name, text in test_cases:
        with tracker.track_memory_usage(f"ocr_{text_name.lower().replace(' ', '_')}") as memory_info:
            # Simulate OCR processing
            time.sleep(0.01)
        
        if 'tracemalloc_diff' in memory_info:
            memory_used = memory_info['tracemalloc_diff']['current_diff_mb']
            chars = len(text)
            words = len(text.split())
            
            memory_per_char = memory_used / chars if chars > 0 else 0
            memory_per_word = memory_used / words if words > 0 else 0
            
            print(f"{text_name}:")
            print(f"  Text length: {chars} chars, {words} words")
            print(f"  Memory used: {memory_used:.4f} MB")
            print(f"  Memory per char: {memory_per_char:.6f} MB")
            print(f"  Memory per word: {memory_per_word:.6f} MB")
    
    print()


def demo_memory_report():
    """Demonstrate memory report generation."""
    print("=== Memory Report Demo ===")
    
    tracker = SimpleMemoryTracker()
    
    # Generate a comprehensive memory report
    report = {
        'timestamp': time.time(),
        'system_memory': {
            'total_mb': psutil.virtual_memory().total / 1024 / 1024,
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'percent_used': psutil.virtual_memory().percent
        },
        'process_memory': tracker.get_psutil_memory(),
        'top_allocations': tracker.get_top_memory_allocations(limit=3)
    }
    
    print("Memory Report:")
    print(f"  System Memory: {report['system_memory']['total_mb']:.0f} MB total, {report['system_memory']['percent_used']:.1f}% used")
    print(f"  Process Memory: {report['process_memory']['rss']:.2f} MB RSS, {report['process_memory']['percent']:.2f}%")
    print(f"  Top Allocations:")
    for i, allocation in enumerate(report['top_allocations'], 1):
        print(f"    {i}. {allocation['filename']}: {allocation['size_mb']:.2f} MB")
    
    print()


def main():
    """Run all memory tracking demos."""
    print("Enhanced Memory Tracking Demo")
    print("============================")
    print()
    
    try:
        # Demo 1: Basic memory tracking
        demo_basic_memory_tracking()
        
        # Demo 2: Memory patterns
        demo_memory_patterns()
        
        # Demo 3: System memory information
        demo_system_memory_info()
        
        # Demo 4: Memory efficiency
        demo_memory_efficiency()
        
        # Demo 5: Memory report generation
        demo_memory_report()
        
        print("All memory tracking demos completed successfully!")
        print()
        print("Key Features Demonstrated:")
        print("- tracemalloc for precise memory allocation tracking")
        print("- psutil for system-level memory monitoring")
        print("- Context managers for operation-level tracking")
        print("- Memory efficiency calculations")
        print("- Comprehensive memory reporting")
        print()
        print("This enhanced memory tracking provides:")
        print("- More accurate memory measurements")
        print("- Detailed memory usage patterns")
        print("- Memory efficiency analysis")
        print("- Better resource optimization insights")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 