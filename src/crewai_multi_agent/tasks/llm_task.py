from crewai import Task
from simple_agent_common.utils.rate_limiter import RateLimiter
import time
import groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from simple_agent_common.multiagent import MATH_PROMPT, PHYSICS_PROMPT, CHEMISTRY_PROMPT, BIOLOGY_PROMPT, ROUTER_PROMPT
from simple_agent_common.utils.token_counter import TokenCounter
from typing import Dict, Any, Optional, List
from pydantic import Field, ConfigDict
import logging


class LLMTask(Task):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        validate_assignment=False,
        validate_default=False,
        frozen=False,
        from_attributes=True,
        revalidate_instances='never'
    )
    llm: Any = Field(None, description="The LLM instance to be used for task execution")
    metrics: Dict[str, Any] = Field(default=None, description="Metrics for tracking task execution details")
    rate_limiter: RateLimiter = Field(default_factory=RateLimiter, description="Rate limiter for task execution")
    config: Dict[str, Any] = Field(default=None, description="Configuration for task execution")
    logger: Optional[logging.Logger] = Field(None, exclude=True)
    token_counter: TokenCounter = Field(default=None, description="Token counter for task execution")
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    context: Dict[str, Any] = Field(default=None, description="Inputs for task execution")

    def __init__(self, description, agent, llm, logger, config, **kwargs):
        super().__init__(description=description, agent=agent, verbose=False, **kwargs)
        self.llm = llm
        self.logger = logger
        self.config = config
        self.rate_limiter = RateLimiter(
            max_calls=self.config["model"]["max_calls"],
            pause_time=self.config["model"]["pause_time"]
        )
        self.metrics = {
            'llm_calls': 0,
            'latencies': [],
            'prompt_tokens': [],
            'total_tokens': 0,
            'total_latency': 0
        }

        self.context = {}
        self.token_counter = TokenCounter()
        self.retry_count = 0 
        self.max_retries = 3
        

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

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        execute_with_retry = self._get_retry_decorator()(self._execute)
        return execute_with_retry()
    
    def _execute(self) -> Dict[str, Any]:# Debug log
        """Executes LLM Call with Rate Limiting and Captures Metrics"""
        with self.rate_limiter:
            try:
                start_time = time.time()

                # Use question from context throw if not provided
                question = self.context.get("input", None)
                task_dict = self.context.get("tasks_dict", None)

                if not question:
                    raise ValueError("No question provided in context")

                # Track prompt tokens
                prompt_tokens = self.token_counter.count_tokens(self.description)
                self.metrics['prompt_tokens'].append(prompt_tokens)

                messages = [
                {"role": "system", "content": self.description},
                {"role": "user", "content": question}
                ]

                response = self.llm.invoke(messages)
                execution_time = time.time() - start_time

                response_content = response if response is str else response.content
                predicted_yield = self.agent.process_response({"result": response_content, "tasks_dict": task_dict})

                self.metrics['latencies'].append(execution_time)
                self.metrics['llm_calls'] += 1
                self.metrics['total_latency'] += execution_time
                
                response_tokens = self.token_counter.count_tokens(response.content)
                total_tokens = prompt_tokens + response_tokens
                self.metrics['total_tokens'] += total_tokens

                output = predicted_yield.agent.role if isinstance(predicted_yield, Task) else predicted_yield

                return {
                    "agent": self.agent.name,  
                    "response": response,
                    "predicted_yield": predicted_yield,
                    "latency": execution_time,
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
    
    # This overrides the default implementation because crewai only extracts description and output. SO DUMB
    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        super().interpolate_inputs(inputs)
        self.context = inputs
