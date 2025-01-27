from typing import Dict, Any, List, Optional
import autogen
from .base_agent import BaseAgentConfig
from agents.token_counter import TokenCounter
import time
import re
import logging
import random
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

class PredictionAgent:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, 
                 token_counter: TokenCounter, llm_config: Dict[str, Any]):
        self.base_config = BaseAgentConfig(config, logger)
        
        # Create AutoGen agent with proper Groq configuration
        self.assistant = autogen.AssistantAgent(
            name="yield_prediction_expert",
            system_message="You are an agricultural yield prediction expert. Return ONLY the predicted yield as a number, no other text.",
            llm_config=llm_config
        )
        
        # Create user proxy for interactions
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            is_termination_msg=lambda x: isinstance(x.get('content', ''), str) and bool(re.search(r'^\d+\.?\d*$', x.get('content', '').strip()))
        )
        
        self.logger = logger
        self.token_counter = token_counter
        self.metrics = {
            'llm_calls': 0,
            'latencies': [],
            'prompt_tokens': [],
            'predictions': []
        }
        self.max_retries = 3
        self.retry_count = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def predict_yield(self, features: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Make a yield prediction using AutoGen agents with group chat"""
        self.logger.debug("predict_yield called with features: %s", features)
        try:
            # Count both system and user prompt tokens since both are used in the chat
            system_tokens = self.token_counter.count_tokens(self.assistant.system_message)
            user_tokens = self.token_counter.count_tokens(context)
            prompt_tokens = system_tokens + user_tokens
            
            # Track total prompt tokens
            self.metrics['prompt_tokens'].append(prompt_tokens)
            
            # Track API call (including retries)
            self.metrics['llm_calls'] += 1
            start_time = time.time()
            
            # Use direct chat with assistant
            self.user_proxy.initiate_chat(
                self.assistant,
                message=context
            )
            
            # Get the last message using AutoGen's standard pattern
            response = self.assistant.last_message()["content"]
            if not response:
                raise ValueError("No response received")
        
            # Track latency
            latency = time.time() - start_time
            self.metrics['latencies'].append(latency)
            
            # Count response tokens
            response_tokens = self.token_counter.count_tokens(response)
                
            # Track total tokens for this interaction
            total_tokens = prompt_tokens + response_tokens
            self.metrics['total_tokens'] = self.metrics.get('total_tokens', 0) + total_tokens
            
            predicted_yield = self._extract_predicated_yield(response)

            if predicted_yield < 0:
                raise ValueError("Failed to extract valid prediction")
            
            return {
                'predicted_yield': predicted_yield,
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
 
    def _extract_predicated_yield(self, response: str) -> float:
        result = -1
        """Extract numerical prediction from response"""
        try:
            result = float(re.search(r"(\d+\.?\d*)", response).group(1))
        except:
            self.logger.error(f"Failed to extract prediction from: {response}")
            
        return result

    def _build_prediction_context(self, features: Dict[str, Any], dataset: Dict[str, Any]) -> str:
        """Build prediction context from historical data"""
        try:
            df = dataset.df
            # Get num_few_shot from config
            num_few_shot = self.base_config.config['benchmark'].get('num_few_shot', 5)
            random_few_shot = self.base_config.config['benchmark'].get('random_few_shot', False)
            self.logger.debug(f"Using {num_few_shot} few-shot examples")
            
            # Filter dataset for the same crop
            crop_data = df[df['Crop'] == features['crop']].copy()
            
            # Exclude exact matches
            crop_data = crop_data[
                ~((crop_data['Precipitation (mm day-1)'] == features['precipitation']) &
                  (crop_data['Specific Humidity at 2 Meters (g/kg)'] == features['specific_humidity']) &
                  (crop_data['Relative Humidity at 2 Meters (%)'] == features['relative_humidity']) &
                  (crop_data['Temperature at 2 Meters (C)'] == features['temperature']))
            ]
            
            if crop_data.empty:
                self.logger.error(f"No historical data found for crop: {features['crop']}")
                raise
            
            # Choose similar historiucal rows
            if not random_few_shot:
                # Calculate similarity scores using CrewAI approach
                similarity_scores = (
                    abs(crop_data['Precipitation (mm day-1)'] - features['precipitation']) / crop_data['Precipitation (mm day-1)'] +
                    abs(crop_data['Specific Humidity at 2 Meters (g/kg)'] - features['specific_humidity']) / crop_data['Specific Humidity at 2 Meters (g/kg)'] +
                    abs(crop_data['Relative Humidity at 2 Meters (%)'] - features['relative_humidity']) / crop_data['Relative Humidity at 2 Meters (%)'] +
                    abs(crop_data['Temperature at 2 Meters (C)'] - features['temperature']) / crop_data['Temperature at 2 Meters (C)']
                )
            
                # Get most similar examples
                selected_indices = similarity_scores.nsmallest(num_few_shot).index
                selected_rows = crop_data.loc[selected_indices]
                historical_phrase = f"Most similar historical records for {features['crop']} (ordered by relevance):\n"

            else: # Choose random rows from crop data
                # Get random indices directly - no need for similarity scores
                selected_indices = random.sample(range(len(crop_data)), min(num_few_shot, len(crop_data)))
                selected_rows = crop_data.iloc[selected_indices]
                historical_phrase = f"Random historical records for {features['crop']}:\n"
            
            # Get the summary yield stats for the featured crop
            yield_stats = dataset.summary["crop_distribution"][features["crop"]]["yield_stats"]

                # Format few-shot examples
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
            
        except Exception as e:
            self.logger.error(f"Failed to build prediction context: {str(e)}")
            raise 