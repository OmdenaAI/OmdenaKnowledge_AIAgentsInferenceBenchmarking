from typing import Dict, Any, List, Optional
import autogen
from simple_agent_common.utils import RateLimiter, TokenCounter
import time
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import logging
from openai import RateLimitError, OpenAI

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
        
    def _extract_number(self, question: str, response: str) -> float:
        """
        Extract numerical value from response with proper error handling
        
        Args:
            question: Original question for logging context
            response: Response from LLM to parse
            
        Returns:
            float: Extracted number or penalty value (1e6) if parsing fails
        """
        PENALTY_VALUE = 1e6
        value = PENALTY_VALUE
        response = response.strip()
        
        try:
            # First try: direct conversion after removing commas
            value = float(response.replace(',', ''))
            
        except (ValueError, IndexError, StopIteration):
            self.logger.info(f"Could not extract number for question: {question} from response: {response}")
            
            try:
                # Second try: find any numbers in the string
                numbers = re.findall(r'[-+]?\d*[,.]?\d+', response.replace(',', ''))
                if numbers:
                    value = float(numbers[-1])
                    self.logger.info(f"Extracted number with fallback for question: {question} from response: {response}")
                else:
                    self.logger.warning(f"No numbers found in response: {response}")
                    
            except Exception as e:
                self.logger.error(f"Could not extract number with fallback from response: '{response}' for question: '{question}'. Error: {str(e)}")
        
        return value

    def _get_retry_decorator(self):
        """Create retry decorator with config values"""
        return retry(
            stop=stop_after_attempt(self.config['model']['stop_after_attempt']),
            wait=wait_exponential(
                multiplier=self.config['model']['wait_multiplier'],
                min=self.config['model']['wait_min'],
                max=self.config['model']['wait_max']
            ),
            retry=retry_if_exception_type(RateLimitError)
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
            predicted_value = self._extract_number(prompt, response)
            
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
                'retry_count': 0
            }
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            if self.retry_count >= self.max_retries:
                self.logger.error("Max retries reached")
            raise
