from typing import Dict, Any, List, Optional
import autogen
from simple_agent_common.utils import RateLimiter, TokenCounter
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import groq
import logging
from simple_agent_common.utils import extract_number

class BaseAgent():
    def __init__(self, name: str, system_prompt: str, logger: logging.Logger, llm_config: List[Dict[str, Any]], 
                 config: Dict[str, Any]):

        self.name = name
        self.logger = logger

        self.llm_config = llm_config
        self.config = config
        self.rate_limiter = RateLimiter(
            max_calls=config["model"]["max_calls"],
            pause_time=config["model"]["pause_time"]
        )
        self.token_counter = TokenCounter()
        self.metrics = {
            'llm_calls': 0,
            'latencies': [],
            'prompt_tokens': [],
            'total_tokens': 0,
            'total_latency': 0
        }
        self.retry_count = 0
        self.max_retries = 3

        # Create AutoGen assistant with proper configuration
        self.assistant = autogen.AssistantAgent(
            name=name,
            system_message=system_prompt,
            llm_config={"config_list": self.llm_config}
        )
        
        # Create user proxy for interactions with code execution disabled
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False  # Disable code execution
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

    def execute(self, prompt: str) -> Dict[str, Any]:
        execute_with_retry = self._get_retry_decorator()(self._execute)
        return execute_with_retry(prompt)

    def _execute(self, prompt: str) -> Dict[str, Any]:
        """Execute with metrics tracking"""
        try:
            start_time = time.time()
            
            # Track prompt tokens
            prompt_tokens = self.token_counter.count_tokens(prompt)
            self.metrics['prompt_tokens'].append(prompt_tokens)
            
            # Use direct chat with assistant
            self.user_proxy.initiate_chat(
                self.assistant,
                message=prompt
            )
            
            # Get the last message using AutoGen's standard pattern
            response = self.assistant.last_message()["content"]

            if not response:
                raise ValueError("No response received")
            
            # Extract numerical answer
            predicted_value = extract_number(response, self.logger)
            
            # Update metrics
            end_time = time.time()
            latency = end_time - start_time
            self.metrics['latencies'].append(latency)
            self.metrics['llm_calls'] += 1
            self.metrics['total_latency'] += latency
            
            response_tokens = self.token_counter.count_tokens(response)
            total_tokens = prompt_tokens + response_tokens
            self.metrics['total_tokens'] += total_tokens
            
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
