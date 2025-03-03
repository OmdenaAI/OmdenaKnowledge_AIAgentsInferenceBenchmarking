from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
import groq
import logging
import os
import time
from simple_agent_common.utils import RateLimiter, TokenCounter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from simple_agent_common.utils import extract_number

class BaseAgent(ABC):
    def __init__(self, name: str, system_prompt: str, logger: logging.Logger, config: Dict[str, Any]):
        self.name = name
        self.system_prompt = system_prompt
        self.config = config
        self.logger = logger

        self.metrics = {
            'llm_calls': 0,
            'latencies': [],
            'prompt_tokens': [],
            'predictions': []
        }
        self.max_retries = 3
        self.retry_count = 0
        self.rate_limiter = RateLimiter(max_calls=4, pause_time=20)
        self.token_counter = TokenCounter()
        
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=self.config["model"]["name"],
            temperature=self.config["model"]["temperature"],
            max_tokens=self.config["model"]["max_tokens"]
        )

    def _get_retry_decorator(self):
        """Create retry decorator with config values"""
        return retry(
            stop=stop_after_attempt(self.config['model']['stop_after_attempt']),
            wait=wait_exponential(
                multiplier=self.config['model']['wait_multiplier'],
                min=self.config['model']['wait_min'],
                max=self.config['model']['wait_max']
            ),
            retry=retry_if_exception_type(groq.RateLimitError)
        )

    def _execute(self, prompt: str) -> Dict[str, Any]:
        """Execute the agent's task"""
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        try:
            # Count both system and user prompt tokens since both are used in the chat
            system_tokens = self.token_counter.count_tokens(self.system_prompt)
            user_tokens = self.token_counter.count_tokens(prompt)
            prompt_tokens = system_tokens + user_tokens
            
            # Track total prompt tokens
            self.metrics['prompt_tokens'].append(prompt_tokens)
            
            # Track API call (including retries)
            self.metrics['llm_calls'] += 1

            with self.rate_limiter:
                start_time = time.time()
                response = self.llm.invoke(messages)
                end_time = time.time()
            
            latency = end_time - start_time
            self.metrics['latencies'].append(latency)
            
            # Count response tokens
            response_tokens = self.token_counter.count_tokens(response.content)
            total_tokens = prompt_tokens + response_tokens
            predicted_value = extract_number(response.content, self.logger)
            
            return {
                'agent': self.name,
                'predicted_yield': predicted_value,
                'latency': latency,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'total_tokens': total_tokens,
                'retry_count': self.retry_count
            }
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            if self.retry_count >= self.max_retries:
                self.logger.error("Max retries reached")
            raise

    def execute(self, prompt: str) -> Dict[str, Any]:
        """Execute with retry decorator applied"""
        execute_with_retry = self._get_retry_decorator()(self._execute)
        return execute_with_retry(prompt)