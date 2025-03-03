from typing import Dict, Any
from crewai import Task
from simple_agent_common.multiagent import (
    MATH_PROMPT, PHYSICS_PROMPT, CHEMISTRY_PROMPT, 
    BIOLOGY_PROMPT, ROUTER_PROMPT
)
from .llm_task import LLMTask

class TaskFactory:
    """Factory for creating CrewAI tasks"""
    
    def __init__(self, llm, logger, config: Dict[str, Any]):
        self.llm = llm
        self.logger = logger
        self.config = config
        self.prompts = {
            "math": MATH_PROMPT,
            "biology": BIOLOGY_PROMPT,
            "chemistry": CHEMISTRY_PROMPT,
            "physics": PHYSICS_PROMPT,
            "router": ROUTER_PROMPT
        }

    def create_task(self, agent_name: str, agent) -> Task:
        """Create a single task for an agent"""
        return LLMTask(
            description=self.prompts[agent_name],
            expected_output=self._get_expected_output(agent_name),
            agent=agent,
            llm=self.llm,
            logger=self.logger,
            config=self.config,
            agent_keys=list(self.prompts.keys())
        )

    def create_tasks(self, agents: Dict[str, Any]) -> Dict[str, Task]:
        """Create all tasks for the given agents"""
        return {
            agent_name: self.create_task(agent_name, agent)
            for agent_name, agent in agents.items()
        }

    def _get_expected_output(self, agent_name: str) -> str:
        """Get the expected output description based on agent type"""
        return ("A well-structured and validated numerical response." 
                if agent_name != "router" 
                else "A expert category name.") 