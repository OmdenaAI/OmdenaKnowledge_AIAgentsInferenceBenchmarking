from crewai import Crew
from simple_agent_common.utils import MemoryManager
from agents import ExpertAgent, RouterAgent
from simple_agent_common.multiagent import OrchestratorBase
import time
from langchain_groq import ChatGroq
import os
from tasks.task_factory import TaskFactory

class MultiAgentOrchestrator(OrchestratorBase):
    """CrewAI-compatible orchestrator that dynamically assigns agents based on LLM routing."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.memory_manager = MemoryManager()

        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=config["model"]["name"],
            temperature=config["model"]["temperature"],
            max_tokens=config["model"]["max_tokens"]
        )

        self.agents = {
            "router": RouterAgent("router", self.logger),
            "math": ExpertAgent("math", self.logger),
            "physics": ExpertAgent("physics", self.logger),
            "chemistry": ExpertAgent("chemistry", self.logger),
            "biology": ExpertAgent("biology", self.logger)
        }
        # Create tasks using factory
        task_factory = TaskFactory(self.llm, self.logger, self.config)
        self.tasks = task_factory.create_tasks(self.agents)

        # Define Crew with Dynamic Task Assignment
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks.values(),
            verbose=False
        )

    def run(self, question: str) -> dict:
        """
        Executes ONE question at a time by dynamically assigning the agent.
        """
        start_time = time.time()
        self.logger.info(f"Executing question: {question}")

        # Select the router agent
        router_agent = self.tasks["router"]
        self.crew.tasks = [router_agent]  

        # Execute CrewAI with the router to determine which agent to use
        routing_result = self.crew.kickoff(inputs={"input": question, "tasks_dict": self.tasks})
        
        # Get the assigned agent from routing result
        assigned_agent = routing_result["predicted_yield"]  # This should be the agent name (math, physics, etc.)

        # Execute with the assigned expert agent
        self.crew.tasks = [assigned_agent]
        crew_result = self.crew.kickoff(inputs={"input": question})

        return crew_result
