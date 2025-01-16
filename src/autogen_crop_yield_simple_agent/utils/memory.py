from typing import Dict
import psutil

class MemoryManager:
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
    
    def start_tracking(self):
        """Start memory tracking"""
        self.start_memory = self.get_current_memory()
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        current = self.get_current_memory()
        self.peak_memory = max(self.peak_memory, current)
        return {
            'current': current,
            'peak': self.peak_memory,
            'delta': current - self.start_memory
        }
    
    @staticmethod
    def get_current_memory() -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 