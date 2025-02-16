import os
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from utils.token_counter import LangChainTokenCounter
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import time
from simple_agent_common.multiagent import ROUTER_PROMPT

class QueryRouter:
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        
        self.config = config
        self.logger = logger
        
        # Initialize tokenizer
        self.token_counter = LangChainTokenCounter()
        
        api_key = os.getenv("GROQ_API_KEY")
        # Initialize Groq model
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=self.config["model"]["name"],
            temperature=self.config["model"]["temperature"],
            max_tokens=self.config["model"]["max_tokens"]
        )
        self.max_retries = 3
        self.retry_count = 0

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.token_counter.count_tokens(text))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def determine_agent(self, question: str) -> str:
        """Uses Groq LLM to determine the correct agent."""
        start_time = time.time()
        system_prompt = (ROUTER_PROMPT)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        try:
            response = self.llm.invoke(messages)
            end_time = time.time()
            token_counts = self.token_counter.count_messages(messages + [response])
            agent_type = response.content.strip().lower()
            
            if agent_type not in ["math", "physics", "chemistry", "biology"]:
                self.logger.warning(f"Invalid agent classification: {agent_type}. Defaulting to math.")
                agent_type = "math"
                
            return {
                "agent": agent_type,
                "llm_metrics": {
                    "calls": 1,
                    "total_tokens": token_counts['total_tokens'],
                    "total_latency": end_time - start_time
                }
            }
        except Exception as e:
            self.logger.error(f"Error executing task: {e}")
            if self.retry_count >= self.max_retries:
                self.logger.error("Max retries reached")
            raise