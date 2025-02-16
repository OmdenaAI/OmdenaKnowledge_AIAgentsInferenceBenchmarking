from typing import Dict, Any
from .base_agent import BaseAgent
import logging
from simple_agent_common.multiagent import MATH_PROMPT

class MathAgent(BaseAgent):
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__("math", MATH_PROMPT, logger, config)
