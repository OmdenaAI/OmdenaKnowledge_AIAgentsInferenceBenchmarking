from .base_agent import BaseAgent
import logging

class ExpertAgent(BaseAgent):
    """Expert Agent"""

    def __init__(self, name: str, logger: logging.Logger):
        role = f"{name.capitalize()} Professor"
        goal = f"Evalute {name} questions and provide a well-structured and validated numerical response."
        backstory = f"An AI-powered {name} expert trained in answering {name} questions."

        super().__init__(name, role, goal, backstory, logger)
