import logging
from typing import Dict, Any
from .base_agent import BaseAgent

SYSTEM_PROMPT = """You are a Professor of Biology. Return only a single numerical value with no explanation, units, or additional text.

Strict formatting rules:
- No explanation.
- No equations or calculations.
- No extra text.

"""
class BiologyAgent(BaseAgent):
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__("biology", SYSTEM_PROMPT, logger, config)