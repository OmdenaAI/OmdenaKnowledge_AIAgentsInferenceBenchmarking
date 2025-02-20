import time

class BenchmarkTimer:
    """A simple timer class for benchmarking AI agents."""
    
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def get_time(self):
        """Returns the elapsed time."""
        return self.elapsed_time
