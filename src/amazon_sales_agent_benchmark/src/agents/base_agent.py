from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import logging

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, llm, templates: Dict, logger: logging.Logger):
        """Initialize base agent with common components."""
        self.llm = llm
        self.templates = templates
        self.logger = logger

        if not self.llm:
            raise ValueError("LLM is required")
        if not self.templates:
            raise ValueError("Templates are required")
        if not self.logger:
            raise ValueError("Logger is required")
        
        self.initialized = False
    
    def _clean_response(self, response: str) -> str:
        """Clean response by removing assistant markers and whitespace."""
        tokens = response.split('\n\n')
        if (len(tokens) > 1):
            if (tokens[0].startswith('\n<|user|>')):
                response = tokens[1]
            else:
                response = tokens[0]

        return response.replace('<|assistant|>', '').replace('"', '').strip() if response else ""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize agent with necessary components."""
        pass
        
    @abstractmethod
    def execute_query(self, query: str, category: str, context: Optional[str] = None) -> str:
        """Execute a query and return response."""
        pass
        
    @abstractmethod
    def aggregate_results(self, results: List[str], original_prompt: str) -> str:
        """Aggregate multiple responses into single response."""
        pass 