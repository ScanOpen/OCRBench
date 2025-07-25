"""
Memory Tracking Utilities

Provides precise memory usage tracking using tracemalloc and memory_profiler.
"""

import tracemalloc
import time
import psutil
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
import gc
import os

# Try to import memory_profiler, but make it optional
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    # Create a dummy decorator if memory_profiler is not available
    def memory_profile(func):
        return func


class MemoryTracker:
    """
    Comprehensive memory tracking utility using multiple methods.
    """
    
    def __init__(self, enable_tracemalloc: bool = True, enable_memory_profiler: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            enable_tracemalloc: Whether to use tracemalloc for tracking
            enable_memory_profiler: Whether to use memory_profiler for tracking
        """
        self.enable_tracemalloc = enable_tracemalloc
        self.enable_memory_profiler = enable_memory_profiler and MEMORY_PROFILER_AVAILABLE
        self.tracemalloc_started = False
        
        if enable_tracemalloc:
            self._start_tracemalloc()
    
    def _start_tracemalloc(self):
        """Start tracemalloc if not already started."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_started = True
    
    def _stop_tracemalloc(self):
        """Stop tracemalloc if we started it."""
        if self.tracemalloc_started and tracemalloc.is_tracing():
            tracemalloc.stop()
            self.tracemalloc_started = False
    
    def get_psutil_memory(self) -> Dict[str, float]:
        """
        Get memory usage using psutil.
        
        Returns:
            Dictionary with memory usage information in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent(),     # Memory usage percentage
            'available': psutil.virtual_memory().available / 1024 / 1024  # Available system memory in MB
        }
    
    def get_tracemalloc_memory(self) -> Dict[str, Any]:
        """
        Get memory usage using tracemalloc.
        
        Returns:
            Dictionary with tracemalloc memory information
        """
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
        """
        Get comprehensive memory snapshot using all available methods.
        
        Returns:
            Dictionary with memory information from all sources
        """
        snapshot = {
            'timestamp': time.time(),
            'psutil': self.get_psutil_memory()
        }
        
        if self.enable_tracemalloc:
            snapshot['tracemalloc'] = self.get_tracemalloc_memory()
        
        return snapshot
    
    @contextmanager
    def track_memory_usage(self, operation_name: str = "operation"):
        """
        Context manager for tracking memory usage during an operation.
        
        Args:
            operation_name: Name of the operation being tracked
            
        Yields:
            Dictionary with memory tracking information
        """
        # Force garbage collection before starting
        gc.collect()
        
        # Get initial memory state
        initial_snapshot = self.get_memory_snapshot()
        start_time = time.time()
        
        # Start tracemalloc if enabled
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
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
            if self.enable_tracemalloc and tracemalloc.is_tracing():
                tracemalloc.stop()
            
            # Force garbage collection after operation
            gc.collect()
            
            # Update the yield dictionary with results
            memory_diff.update({
                'duration': duration,
                'final_snapshot': final_snapshot
            })
    
    def _calculate_memory_difference(self, initial: Dict[str, Any], final: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate memory usage differences between snapshots.
        
        Args:
            initial: Initial memory snapshot
            final: Final memory snapshot
            
        Returns:
            Dictionary with memory differences
        """
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
    
    def get_top_memory_allocations(self, limit: int = 10) -> list:
        """
        Get top memory allocations using tracemalloc.
        
        Args:
            limit: Number of top allocations to return
            
        Returns:
            List of top memory allocations
        """
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
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()


class MemoryProfiler:
    """
    Memory profiler using memory_profiler decorator.
    """
    
    @staticmethod
    def profile_function(func, *args, **kwargs):
        """
        Profile a function using memory_profiler.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if MEMORY_PROFILER_AVAILABLE:
            return memory_profile(func)(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    @staticmethod
    def get_memory_usage(func, *args, **kwargs) -> Dict[str, Any]:
        """
        Get memory usage for a function call.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with memory usage information
        """
        # This is a simplified version - in practice, you'd use memory_profiler's
        # detailed profiling capabilities
        tracker = MemoryTracker()
        
        with tracker.track_memory_usage(f"function_{func.__name__}") as memory_info:
            result = func(*args, **kwargs)
            memory_info['result'] = result
        
        return memory_info


def create_memory_tracker(enable_tracemalloc: bool = True, 
                         enable_memory_profiler: bool = True) -> MemoryTracker:
    """
    Factory function to create a memory tracker.
    
    Args:
        enable_tracemalloc: Whether to enable tracemalloc
        enable_memory_profiler: Whether to enable memory_profiler
        
    Returns:
        MemoryTracker instance
    """
    return MemoryTracker(
        enable_tracemalloc=enable_tracemalloc,
        enable_memory_profiler=enable_memory_profiler
    )


def get_system_memory_info() -> Dict[str, Any]:
    """
    Get comprehensive system memory information.
    
    Returns:
        Dictionary with system memory information
    """
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total_mb': vm.total / 1024 / 1024,
        'available_mb': vm.available / 1024 / 1024,
        'used_mb': vm.used / 1024 / 1024,
        'percent_used': vm.percent,
        'swap_total_mb': swap.total / 1024 / 1024,
        'swap_used_mb': swap.used / 1024 / 1024,
        'swap_percent': swap.percent
    }


def format_memory_size(bytes_value: int) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        bytes_value: Memory size in bytes
        
    Returns:
        Formatted memory size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def is_memory_profiler_available() -> bool:
    """
    Check if memory_profiler is available.
    
    Returns:
        True if memory_profiler is available, False otherwise
    """
    return MEMORY_PROFILER_AVAILABLE 