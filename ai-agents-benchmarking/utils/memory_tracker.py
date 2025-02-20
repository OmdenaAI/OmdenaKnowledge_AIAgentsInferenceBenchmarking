import psutil
import os
import time
import threading

class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self._running = False

    def _track_memory(self):
        """Internal function to continuously track memory."""
        while self._running:
            current_memory = self.process.memory_info().rss / (1024 * 1024)  # In MB
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            time.sleep(0.1)

    def start(self):
        """Start memory tracking."""
        self.peak_memory = 0
        self._running = True
        self.tracker_thread = threading.Thread(target=self._track_memory)
        self.tracker_thread.start()

    def stop(self):
        """Stop memory tracking and return peak memory usage."""
        self._running = False
        self.tracker_thread.join()
        return self.peak_memory

if __name__ == "__main__":
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    # Simulate some memory usage
    a = [i for i in range(1000000)]
    time.sleep(1)

    peak_memory = mem_tracker.stop()
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")
