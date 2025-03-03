from crewai import Agent
import re
from pydantic import Field, ConfigDict
import logging
from simple_agent_common.utils import extract_number

class BaseAgent(Agent):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        validate_assignment=False,
        validate_default=False,
        frozen=False,
        from_attributes=True,
        revalidate_instances='never'
    )
    name: str = Field(None, description="The agent name")
    logger: logging.Logger = Field(None, description="The logger for the agent")
    """Base class for CrewAI agents"""

    def __init__(self, name, role, goal, backstory, logger):

        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=False
        )
        self.name = name
        self.logger = logger
        
    def process_response(self, results):
        """Process the response from the LLM"""
        response = results["result"]
        return extract_number(response, self.logger)