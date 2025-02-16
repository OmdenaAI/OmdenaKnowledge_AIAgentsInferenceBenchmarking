from .base_agent import BaseAgent
import logging
from typing import Dict, List, Any
from simple_agent_common.multiagent import PHYSICS_PROMPT

class PhysicsAgent(BaseAgent):
    def __init__(self, name: str, logger: logging.Logger, llm_config: List[Dict[str, Any]], config: Dict[str, Any]):
        super().__init__(name, PHYSICS_PROMPT, logger, llm_config, config) 