from abc import ABC, abstractmethod
from typing import List, Callable

class BaseLLMProvider(ABC):
    """Base interface for LLM providers."""
    
    @abstractmethod
    def infer(self, prompt: str) -> str:
        """Generate text response."""
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings."""
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Count tokens in text."""
        pass 