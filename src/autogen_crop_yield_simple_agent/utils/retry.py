from functools import wraps
import time
from typing import Any, Callable
import logging

def retry_with_exponential_backoff(max_retries=5, base_delay=4, max_delay=20):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            delay = base_delay
            
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    
                    # Calculate next delay with exponential backoff
                    delay = min(delay * 2, max_delay)
                    
                    # Log warning and wait
                    logging.warning(f"Attempt {retry_count} failed with error: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator 