import logging
from typing import Dict, Any
from .base_agent import BaseAgent
from simple_agent_common.multiagent import BIOLOGY_PROMPT

class BiologyAgent(BaseAgent):
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__("biology", BIOLOGY_PROMPT, logger, config)