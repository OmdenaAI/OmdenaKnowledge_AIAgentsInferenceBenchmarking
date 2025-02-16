import autogen
from typing import Dict, Any, List
import logging
import time
import os
from agents import RouterAgent, MathAgent, PhysicsAgent, ChemistryAgent, BiologyAgent
from simple_agent_common.multiagent import OrchestratorBase

class MultiAgentOrchestrator(OrchestratorBase):
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        llm_config = self._get_groq_config(config)

        # Initialize router
        self.router = RouterAgent("router", logger, llm_config, config)
        
        # Initialize domain agents
        self.agents = {
            "math": MathAgent("math", logger, llm_config, config),
            "physics": PhysicsAgent("physics", logger, llm_config, config),
            "chemistry": ChemistryAgent("chemistry", logger, llm_config, config),
            "biology": BiologyAgent("biology", logger, llm_config, config)
        }
        
    def run(self, question: str) -> Dict[str, Any]:
        """Process question with memory and metrics tracking"""
        start_time = time.time()
        
        try:
            # Get agent type from router
            agent_type = self.router.determine_agent(question)
            
            # Execute with appropriate agent
            agent = self.agents[agent_type]
            result = agent.execute(question)
            
            return {
                **result,
                'latency': time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            raise

    def _get_groq_config(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get Groq configuration"""
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.logger.info("GROQ API key loaded successfully")
        
        return [{
            "model": config['model']['name'],
            "api_key": groq_key,
            "base_url": "https://api.groq.com/openai/v1",
            "temperature": config['model']['temperature'],
            "max_tokens": config['model']['max_tokens']
        }]