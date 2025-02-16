from typing import Dict, Any
from .base_agent import BaseAgent
import logging
from simple_agent_common.multiagent import CHEMISTRY_PROMPT

class ChemistryAgent(BaseAgent):
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__("chemistry", CHEMISTRY_PROMPT, logger, config)