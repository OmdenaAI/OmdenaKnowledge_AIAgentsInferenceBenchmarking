from .base_agent import BaseAgent
import logging
from typing import Dict, Any, List
from simple_agent_common.multiagent import BIOLOGY_PROMPT

class BiologyAgent(BaseAgent):
    def __init__(self, name: str, logger: logging.Logger, llm_config: List[Dict[str, Any]], config: Dict[str, Any]):
        super().__init__(name, BIOLOGY_PROMPT, logger, llm_config, config) 