from crewai import Agent
from simple_agent_common.data_classes import AgentMetricsConfig, CropDataset, CropPrediction
from typing import Any, Dict, Annotated, Optional, List, Callable, Protocol
from pydantic import Field, ConfigDict
import pandas as pd
import re
import logging
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from simple_agent_common.utils.token_counter import TokenCounter
from simple_agent_common.utils.rate_limiter import RateLimitError
import groq  # Import the module
from simple_agent_common.singleagent.prompts import YIELD_PREDICTION_PROMPT, YIELD_SYSTEM_PROMPT

class PredictionAgent(Agent):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        validate_assignment=False,
        validate_default=False,
        frozen=False,
        from_attributes=True,
        revalidate_instances='never',
        protected_namespaces=()
    )
    
    metrics: AgentMetricsConfig = Field(default_factory=AgentMetricsConfig)
    logger: Optional[logging.Logger] = Field(None, exclude=True)
    config: Optional[Dict[str, Any]] = Field(None, exclude=True)
    data: Dict = Field(default_factory=dict)
    model: str = Field(default="default")
    benchmark: Dict[str, Any] = Field(default_factory=dict)
    random_few_shot: bool = Field(default=False)
    num_few_shot: int = Field(default=5)
    token_counter: Optional[Callable[[str], int]] = None
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)

    def __init__(self, 
                 llm: Any, 
                 logger: logging.Logger, 
                 config: Dict[str, Any],
                 token_counter: Optional[TokenCounter] = None):
        super().__init__(
            llm=llm,
            role="Yield Prediction Specialist",
            goal="Predict crop yields based on environmental conditions",
            backstory="Expert in agricultural yield prediction using ML and historical data",
            verbose=False,
            tools=[],
            allow_delegation=False
        )
        
        if token_counter is None:
            raise ValueError("token_counter is required for PredictionAgent")
            
        self.model_post_init({
            "logger": logger,
            "config": config,
            "random_few_shot": bool(config['benchmark'].get('random_few_shot', False)),
            "num_few_shot": int(config['benchmark'].get('num_few_shot', 5)),
            "token_counter": token_counter,
            "retry_count": 0,
            "max_retries": 3
        })
        self.logger = logger
        self.config = config
        self.token_counter = token_counter
        self.retry_count = 0 
        self.max_retries = 3

    def track_tokens(self, prompt_text: str) -> int:           
        prompt_tokens = self.token_counter(prompt_text)
        self.metrics.llm_metrics.prompt_tokens.append(prompt_tokens)
        return prompt_tokens

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
    
    def _predict_yield(self, prompt: str, completion: str, dataset: CropDataset) -> CropPrediction:

        try:
            start_time = time.time()  # Start total prediction time
            
            features = self._extract_features(prompt)
            crop_data = dataset.df[
                (dataset.df['Crop'] == features['crop']) &
                ~(
                    (dataset.df['Precipitation (mm day-1)'] == features['precipitation']) &
                    (dataset.df['Specific Humidity at 2 Meters (g/kg)'] == features['specific_humidity']) &
                    (dataset.df['Relative Humidity at 2 Meters (%)'] == features['relative_humidity']) &
                    (dataset.df['Temperature at 2 Meters (C)'] == features['temperature'])
                )
            ]
            actual_yield = float(re.search(r"Yield is (\d+)", completion).group(1))
            
            # Get the summary yield stats for the featured crop
            yield_stats = dataset.summary["crop_distribution"][features["crop"]]["yield_stats"]

            # Prepare messages
            system_prompt = YIELD_SYSTEM_PROMPT
            user_prompt = self._build_prediction_context(features, crop_data, yield_stats)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Count total tokens BEFORE the API call
            full_prompt = system_prompt + user_prompt
            prompt_tokens = self.track_tokens(full_prompt)
            
            # Increment call count BEFORE the API call
            self.metrics.llm_metrics.call_count += 1
            
            llm_start_time = time.time()
            response = self.llm.invoke(messages)
            self.metrics.llm_metrics.latencies.append(time.time() - llm_start_time)
            
            try:
                predicted_yield = float(re.search(r"(\d+\.?\d*)", response.content).group(1))
            except:
                predicted_yield = -1
                self.logger.error(f"Failed to extract prediction from response: {response.content}")
            
            total_time = time.time() - start_time  # Calculate total prediction time
            self.logger.info(f"Predicted yield for {features['crop']}: {predicted_yield:.0f}, Actual yield: {actual_yield:.0f} " +
                             f"Total prediction time: {total_time:.2f}s, LLM API time: {self.metrics.llm_metrics.latencies[-1]:.2f}s")
            
            return CropPrediction(
                predicted_yield=predicted_yield,
                actual_yield=actual_yield,
                features=features,
                question=prompt,
                context=user_prompt
            )
        except groq.RateLimitError as e:  # Use full path
            self.logger.warning(f"Groq rate limit hit: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            if self.retry_count >= self.max_retries:
                self.logger.error("Max retries reached")
            raise

    def predict_yield(self, prompt: str, completion: str, dataset: CropDataset) -> CropPrediction:
        """Execute with retry decorator applied"""
        predict_yield_with_retry = self._get_retry_decorator()(self._predict_yield)
        return predict_yield_with_retry(prompt, completion, dataset)

    def _extract_features(self, prompt: str) -> dict:
        pattern = r"precipitation of ([\d.]+).*humidity of ([\d.]+).*humidity of ([\d.]+)%.*temperature of ([\d.]+)°C.*crop ([^.]+)\."
        match = re.search(pattern, prompt)
        return {
            'precipitation': float(match.group(1)),
            'specific_humidity': float(match.group(2)),
            'relative_humidity': float(match.group(3)),
            'temperature': float(match.group(4)),
            'crop': match.group(5).strip()
        }

    def _build_prediction_context(self, features: dict, crop_data: pd.DataFrame, yield_stats: str) -> str:
        if self.random_few_shot:
            # Get random indices directly
            selected_indices = random.sample(range(len(crop_data)), min(self.num_few_shot, len(crop_data)))
            selected_rows = crop_data.iloc[selected_indices]
            historical_records_type = "Random Historical Records"
            similarity_type = "random"
        else:
            # Calculate similarity scores
            similarity_scores = (
                abs(crop_data['Precipitation (mm day-1)'] - features['precipitation']) / crop_data['Precipitation (mm day-1)'] +
                abs(crop_data['Specific Humidity at 2 Meters (g/kg)'] - features['specific_humidity']) / crop_data['Specific Humidity at 2 Meters (g/kg)'] +
                abs(crop_data['Relative Humidity at 2 Meters (%)'] - features['relative_humidity']) / crop_data['Relative Humidity at 2 Meters (%)'] +
                abs(crop_data['Temperature at 2 Meters (C)'] - features['temperature']) / crop_data['Temperature at 2 Meters (C)']
            )
            selected_indices = similarity_scores.nsmallest(self.num_few_shot).index
            selected_rows = crop_data.loc[selected_indices]
            historical_records_type = "Most Similar Historical Records"
            similarity_type = "most similar"
            
        historical_examples = [
            f"* When precipitation={row['Precipitation (mm day-1)']:.2f}, "
            f"specific_humidity={row['Specific Humidity at 2 Meters (g/kg)']:.2f}, "
            f"relative_humidity={row['Relative Humidity at 2 Meters (%)']:.2f}, "
            f"temperature={row['Temperature at 2 Meters (C)']:.2f}, "
            f"yield was {row['Yield']:.0f}"
            for _, row in selected_rows.iterrows()
        ]

        return YIELD_PREDICTION_PROMPT.format(
            crop=features['crop'],
            precipitation=features['precipitation'],
            specific_humidity=features['specific_humidity'],
            relative_humidity=features['relative_humidity'],
            temperature=features['temperature'],
            historical_records_type=historical_records_type,
            historical_examples='\n'.join(historical_examples),
            yield_stats=yield_stats,
            mean_yield=yield_stats['mean'],
            similarity_type=similarity_type
        )