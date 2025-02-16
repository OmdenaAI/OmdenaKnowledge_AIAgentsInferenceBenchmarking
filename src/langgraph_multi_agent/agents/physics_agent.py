from typing import Dict, Any
from .base_agent import BaseAgent
import logging
from simple_agent_common.multiagent import PHYSICS_PROMPT

class PhysicsAgent(BaseAgent):
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__("physics", PHYSICS_PROMPT, logger, config) 