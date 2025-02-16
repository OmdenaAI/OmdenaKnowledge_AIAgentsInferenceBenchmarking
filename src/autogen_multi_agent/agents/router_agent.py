import autogen
from typing import Dict, Any, List
import time
import logging
from simple_agent_common.utils import TokenCounter
from simple_agent_common.multiagent import ROUTER_PROMPT

class RouterAgent:
    def __init__(self, name: str, logger: logging.Logger, llm_config: List[Dict[str, Any]], config: Dict[str, Any]):
        self.logger = logger
        self.llm_config = llm_config
        self.config = config
        self.token_counter = TokenCounter()
        
        # Create AutoGen assistant with proper configuration
        self.assistant = autogen.AssistantAgent(
            name=name,
            system_message=ROUTER_PROMPT,
            llm_config={"config_list": self.llm_config}
        )
        
        # Create user proxy for interactions with code execution disabled
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False  # Disable code execution
        )

    def determine_agent(self, question: str) -> str:
        """Uses Groq LLM to determine the correct agent."""
        try:
            # Use direct chat with assistant
            self.user_proxy.initiate_chat(
                self.assistant,
                message=question
            )
            
            # Get the last message using AutoGen's standard pattern
            response = self.assistant.last_message()["content"]
            if not response:
                raise ValueError("No response received")
            
            agent_type = response.strip().lower()
            if agent_type not in ["math", "physics", "chemistry", "biology"]:
                self.logger.warning(f"Invalid agent classification: {agent_type}. Defaulting to math.")
                agent_type = "math"
                
            return agent_type
            
        except Exception as e:
            self.logger.error(f"Error determining agent: {str(e)}")
            return "math"

    def determine_agent_with_metrics(self, question: str) -> Dict[str, str]:
        """Uses Groq LLM to determine the correct agent and returns metrics."""
        start_time = time.time()
        
        with self.rate_limiter:
            response = self.generate(question)
            end_time = time.time()
            
            agent_type = response.strip().lower()
            if agent_type not in ["math", "physics", "chemistry", "biology"]:
                self.logger.warning(f"Invalid agent classification: {agent_type}. Defaulting to math.")
                agent_type = "math"
            
            token_counts = self.token_counter.count_messages([question, response])
            
            return {
                "agent": agent_type,
                "llm_metrics": {
                    "calls": 1,
                    "total_tokens": token_counts['total_tokens'],
                    "total_latency": end_time - start_time
                }
            } 