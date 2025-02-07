import time
import logging
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

class RateLimitError(Exception):
    """Custom exception for rate limit violations"""
    pass

@dataclass
class RateWindow:
    """Track rate limiting within a time window"""
    start_time: float
    calls: list[float]
    tokens_used: int = 0

    def is_expired(self, now: float, window_size: float) -> bool:
        """Check if window has expired"""
        return now - self.start_time >= window_size

    def cleanup_calls(self, now: float, window_size: float) -> None:
        """Remove calls outside the window"""
        self.calls = [t for t in self.calls if now - t < window_size]

class RateLimiter:
    """Rate limiter that handles both call frequency and token usage limits"""
    
    def __init__(self, max_calls: int = 5, pause_time: float = 20, token_limit: Optional[int] = None):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum number of calls allowed per window
            pause_time: Time window in seconds
            token_limit: Optional token limit per window
        """
        self.max_calls = max_calls
        self.pause_time = pause_time
        self.token_limit = token_limit
        self.window = RateWindow(start_time=time.time(), calls=[])
        self.logger = logging.getLogger(__name__)

    def _reset_window(self, now: float) -> None:
        """Reset the rate limiting window"""
        self.window = RateWindow(start_time=now, calls=[])
        self.logger.debug(f"Rate window reset at {datetime.fromtimestamp(now)}")

    def _handle_rate_limit(self, now: float) -> None:
        """Handle case when rate limit is reached"""
        sleep_time = self.pause_time - (now - self.window.start_time)
        if sleep_time > 0:
            self.logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        self._reset_window(time.time())

    def __enter__(self) -> 'RateLimiter':
        now = time.time()
        
        # Reset window if expired
        if self.window.is_expired(now, self.pause_time):
            self._reset_window(now)
        else:
            # Cleanup old calls
            self.window.cleanup_calls(now, self.pause_time)
        
        # Handle rate limiting
        if len(self.window.calls) >= self.max_calls:
            self._handle_rate_limit(now)
            
        self.window.calls.append(now)
        return self

    def check_tokens(self, tokens: int) -> None:
        """
        Check if adding tokens would exceed limit
        
        Args:
            tokens: Number of tokens to add
            
        Raises:
            RateLimitError: If token limit would be exceeded
        """
        if not self.token_limit:
            return
            
        if self.window.tokens_used + tokens > self.token_limit:
            raise RateLimitError(
                f"Token limit exceeded: {self.window.tokens_used + tokens} > {self.token_limit} "
                f"in window starting at {datetime.fromtimestamp(self.window.start_time)}"
            )
        self.window.tokens_used += tokens

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass 