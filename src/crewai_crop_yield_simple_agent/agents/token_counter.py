from typing import Protocol

class TokenCounter(Protocol):
    """Protocol for token counting functions"""
    def __call__(self, text: str) -> int:
        """Count tokens in text"""
        ... 