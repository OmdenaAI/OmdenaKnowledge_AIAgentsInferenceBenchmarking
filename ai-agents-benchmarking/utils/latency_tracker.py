# utils/latency_tracker.py

import time

class LatencyTracker:
    """Utility for tracking execution time."""
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and return elapsed time."""
        self.end_time = time.time()
        return round(self.end_time - self.start_time, 2)

    def reset(self):
        """Reset the timer."""
        self.start_time = 0
        self.end_time = 0
