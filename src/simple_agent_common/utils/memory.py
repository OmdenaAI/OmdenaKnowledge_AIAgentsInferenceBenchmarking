from typing import Dict
import psutil
import gc
import os

class MemoryManager:
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
        try:
            # Get current process
            self.process = psutil.Process()
            # Verify process is running and accessible
            if not self.process.is_running():
                raise psutil.NoSuchProcess(os.getpid())
        except psutil.Error as e:
            raise RuntimeError(f"Failed to initialize process monitoring: {e}")
        
    def start_tracking(self):
        """Start memory tracking with clean state"""
        gc.collect()  # Force garbage collection before starting
        self.start_memory = self.get_current_memory()
        self.peak_memory = self.start_memory
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get essential memory statistics"""
        try:
            current = self.get_current_memory()
            self.peak_memory = max(self.peak_memory, current)
            
            return {
                'current': current,
                'peak': self.peak_memory,
                'delta': current - self.start_memory
            }
        except psutil.Error as e:
            raise RuntimeError(f"Failed to get memory statistics: {e}")
    
    def get_current_memory(self) -> float:
        """Get current RSS memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except psutil.Error as e:
            raise RuntimeError(f"Failed to get current memory usage: {e}") 