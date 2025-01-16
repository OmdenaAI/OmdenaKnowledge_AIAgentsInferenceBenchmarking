import time
import logging
from typing import Optional

class RateLimiter:
    def __init__(self, max_calls=5, pause_time=20):
        self.max_calls = max_calls
        self.pause_time = pause_time
        self.calls = []
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        now = time.time()
        # Remove old calls
        self.calls = [t for t in self.calls if now - t < self.pause_time]
        
        # If at max calls, sleep until oldest call is more than pause_time ago
        if len(self.calls) >= self.max_calls:
            sleep_time = self.pause_time - (now - self.calls[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            self.calls = self.calls[1:]  # Remove oldest call
            
        self.calls.append(now)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass 