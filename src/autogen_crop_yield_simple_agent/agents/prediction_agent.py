from typing import Dict, Any, List, Optional
import autogen
from .base_agent import BaseAgentConfig
from simple_agent_common.utils import TokenCounter
import time
import re
import logging
import random
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import groq
from simple_agent_common.singleagent.prompts import YIELD_PREDICTION_PROMPT, YIELD_SYSTEM_PROMPT
import traceback

class PredictionAgent:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, 
                 token_counter: TokenCounter, llm_config: Dict[str, Any]):
        self.base_config = BaseAgentConfig(config, logger)
        self.config = config
        
        # Create AutoGen agent with proper Groq configuration
        self.assistant = autogen.AssistantAgent(
            name="yield_prediction_expert",
            system_message=YIELD_SYSTEM_PROMPT,
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
    
    def _predict_yield(self, features: Dict[str, Any], context: str) -> Dict[str, Any]:
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
        except groq.RateLimitError as e:  # Use full path
            self.logger.info(f"Groq rate limit hit: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            if self.retry_count >= self.max_retries:
                self.logger.error("Max retries reached")
            raise
 
    def predict_yield(self, features: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Execute with retry decorator applied"""
        predict_yield_with_retry = self._get_retry_decorator()(self._predict_yield)
        return predict_yield_with_retry(features, context)
    
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
                historical_records_type = "Most Similar Historical Records"
                similarity_type = "most similar"

            else: # Choose random rows from crop data
                # Get random indices directly - no need for similarity scores
                selected_indices = random.sample(range(len(crop_data)), min(num_few_shot, len(crop_data)))
                selected_rows = crop_data.iloc[selected_indices]
                historical_records_type = "Random Historical Records"
                similarity_type = "random"
            
            # Get the summary yield stats for the featured crop
            yield_stats = dataset.summary["crop_distribution"][features["crop"]]["yield_stats"]

                # Format few-shot examples
            historical_examples = [
                f"* When precipitation={row['Precipitation (mm day-1)']:.2f}, "
                f"specific_humidity={row['Specific Humidity at 2 Meters (g/kg)']:.2f}, "
                f"relative_humidity={row['Relative Humidity at 2 Meters (%)']:.2f}, "
                f"temperature={row['Temperature at 2 Meters (C)']:.2f}, "
                f"yield was {row['Yield']:.0f}"
                for _, row in selected_rows.iterrows()
            ]
            
            result = YIELD_PREDICTION_PROMPT.format(
                crop=features['crop'],
                precipitation=features['precipitation'],
                specific_humidity=features['specific_humidity'],
                relative_humidity=features['relative_humidity'],
                temperature=features['temperature'],
                historical_records_type=historical_records_type,
                historical_examples='\n'.join(historical_examples),
                yield_stats=yield_stats,
                mean_yield=yield_stats['mean'],
                similarity_type=similarity_type).strip()
       
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to build prediction context: {str(e)}")
            raise
