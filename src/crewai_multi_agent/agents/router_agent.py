from typing import Dict, Any
from agents.base_agent import BaseAgent
import logging
class RouterAgent(BaseAgent):
    """Handles LLM-based routing to determine which agent should process the task."""

    def __init__(self, name: str, logger: logging.Logger):
        role = f"Routing Agent"
        goal = f"You are a query classifier. Classify a question into one of the following categories: math, physics, chemistry, or biology and return only the category name."
        backstory = f"An AI-powered question classifier trained on how to route questions to experts."

        super().__init__(name, role, goal, backstory, logger)

    def determine_agent(self, response: str, tasks_dict) -> str:
        result = tasks_dict["math"]

        if not response:
            self.logger.warning("No agent response received. Defaulting to math.")
        else: # Default fallback
            agent_type = response.strip().lower()

            if agent_type not in tasks_dict:
                self.logger.warning(f"Invalid classification: {agent_type}. Defaulting to math.")
            else:
                self.logger.info(f"Task assigned to: {agent_type}")
                result = tasks_dict[agent_type]

        return result    

    def process_response(self, results):
        """Process the response from the LLM"""
        response = results["result"]
        task_dict = results["tasks_dict"]

        response = self.determine_agent(response, task_dict)
        return response
