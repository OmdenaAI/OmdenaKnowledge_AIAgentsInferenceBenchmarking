from crewai import Agent
from simple_agent_common.data_classes import AgentMetricsConfig, CropDataset, CropPrediction
from typing import Any, Dict, Annotated, Optional, List, Callable, Protocol
from pydantic import Field, ConfigDict
import pandas as pd
import re
import logging
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential
from .token_counter import TokenCounter

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
    token_counter: Any = Field(None, exclude=True)

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
            "token_counter": token_counter
        })
        self.logger = logger
        self.config = config
        self.token_counter = token_counter

    def track_tokens(self, prompt_text: str) -> int:           
        prompt_tokens = self.token_counter(prompt_text)
        self.metrics.llm_metrics.prompt_tokens.append(prompt_tokens)
        return prompt_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def predict_yield(self, prompt: str, completion: str, dataset: CropDataset) -> CropPrediction:

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
            system_prompt = "You are an agricultural yield prediction expert. Return ONLY the predicted yield as a number, no other text."
            user_prompt = self._build_prediction_context(features, crop_data, yield_stats)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Count total tokens (system + user prompts)
            full_prompt = system_prompt + user_prompt
            prompt_tokens = self.track_tokens(full_prompt)
            
            # Increment call count BEFORE the API call to capture retries
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
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            if self.retry_count >= self.max_retries:
                self.logger.error("Max retries reached")
            raise



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
            # Get random indices directly - no need for similarity scores
            selected_indices = random.sample(range(len(crop_data)), min(self.num_few_shot, len(crop_data)))
            selected_rows = crop_data.iloc[selected_indices]
            
        else:
            # Only calculate similarity scores if we need them
            similarity_scores = (
                abs(crop_data['Precipitation (mm day-1)'] - features['precipitation']) / crop_data['Precipitation (mm day-1)'] +
                abs(crop_data['Specific Humidity at 2 Meters (g/kg)'] - features['specific_humidity']) / crop_data['Specific Humidity at 2 Meters (g/kg)'] +
                abs(crop_data['Relative Humidity at 2 Meters (%)'] - features['relative_humidity']) / crop_data['Relative Humidity at 2 Meters (%)'] +
                abs(crop_data['Temperature at 2 Meters (C)'] - features['temperature']) / crop_data['Temperature at 2 Meters (C)']
            )
            selected_indices = similarity_scores.nsmallest(self.num_few_shot).index
            selected_rows = crop_data.loc[selected_indices]
            
        few_shot_examples = [
            f"* When precipitation={row['Precipitation (mm day-1)']:.2f}, "
            f"specific_humidity={row['Specific Humidity at 2 Meters (g/kg)']:.2f}, "
            f"relative_humidity={row['Relative Humidity at 2 Meters (%)']:.2f}, "
            f"temperature={row['Temperature at 2 Meters (C)']:.2f}, "
            f"yield was {row['Yield']:.0f}"
            for _, row in selected_rows.iterrows()
        ]

        result = (
            f"Predict yield for {features['crop']} based on these conditions:\n\n"
            f"### Current Measurements:\n"
            f"- Precipitation: {features['precipitation']} mm/day\n"
            f"- Specific Humidity: {features['specific_humidity']} g/kg\n"
            f"- Relative Humidity: {features['relative_humidity']}%\n"
            f"- Temperature: {features['temperature']}°C\n\n"
            f"### Most Similar Historical Records:\n"
            f"{chr(10).join(few_shot_examples)}\n\n"
            f"### Yield Statistics:\n"
            f"Utilize the following yield stats from all {features['crop']} records for your prediction:\n"
            f"{yield_stats}.\n"
            f"- Your prediction must fall strictly within this range.\n"
            f"- Use the mean ({yield_stats['mean']:.2f}) as a central reference point.\n\n"
            f"### Instructions:\n"
            f"Base your prediction on the most similar historical records provided, ensuring it is constrained by the yield stats range. "
            f"Your output must be a single number representing the predicted yield, with no additional text."
        )

        return result