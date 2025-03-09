from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig
import logging
from langchain_groq import ChatGroq
from agents.langgraph_agent_template import LanggraphAgentTemplate, AgentState
import random
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import groq
from simple_agent_common.singleagent.prompts import YIELD_PREDICTION_PROMPT, YIELD_SYSTEM_PROMPT  
from simple_agent_common.data_classes import PredictionMetrics
from simple_agent_common.utils import RateLimiter, MemoryManager
import utils.token_counter as LangChainTokenCounter
import re
import time
import numpy as np


class YieldAgentState(AgentState):
    """Extended state for YieldAgent including predictions."""
    dataset: Dict[str, Any]
    questions: List[Dict[str, Any]]
    predictions: Dict[str, Any]

class PredictionAgent(LanggraphAgentTemplate):
    """Agent responsible for making yield predictions."""
    
    def __init__(self, llm: ChatGroq, memory_manager: MemoryManager, config: Dict[str, Any], logger: logging.Logger):
        """Initialize the YieldAgent.
        
        Args:
            llm: ChatGroq instance for predictions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config)
        self.llm = llm
        self.memory_manager = memory_manager
        self.config = config
               
        self.logger = logger
        self.token_counter = LangChainTokenCounter.TokenCounter()
        self.metrics = {
            'llm_calls': 0,
            'latencies': [],
            'prompt_tokens': [],
            'predictions': []
        }
        self.max_retries = 3
        self.retry_count = 0
        self.system_prompt = YIELD_SYSTEM_PROMPT

        # Setup rate limiter
        max_calls = self.config.get("model", {}).get("max_calls", 4)
        pause_time = self.config.get("model", {}).get("pause_time", 30)
        token_limit = self.config.get("model", {}).get("token_limit", 90000)

        self.rate_limiter = RateLimiter(
                max_calls=max_calls,  # More conservative
                pause_time=pause_time,  # Groq's window
                token_limit=token_limit  # Buffer below Groq's 100k limit
            )

    def process_step(self, state: YieldAgentState, config: Optional[RunnableConfig] = None) -> YieldAgentState:
        """Process all yield predictions.
        
        Args:
            state: Current agent state
            config: Optional runnable configuration
            
        Returns:
            YieldAgentState: Updated state with all predictions
        """

        start_time = time.time()

        # Lists to track predictions and metrics
        latencies = []
        llm_calls = 0
        total_prompt_tokens = 0
        total_tokens = 0
        prediction_metrics = PredictionMetrics()

        # Get initial memory state
        start_stats = self.memory_manager.get_memory_stats()
        
        with self.rate_limiter:
            try:
                # Process all questions
                for idx, question in enumerate(state['questions']):
                    features = question['features']
                    dataset = state['dataset']

                    context = self._build_prediction_context(features, dataset)
                    prediction_result = self.predict_yield(features, context)

                    # Update tracking
                    llm_calls += (prediction_result['retry_count'] + 1)
                    total_prompt_tokens += prediction_result['prompt_tokens']
                    total_tokens += prediction_result['total_tokens']
                        
                    prediction_metrics.predictions.append((prediction_result['predicted_yield'], features['Yield'], None))

                    latencies.append(prediction_result['latency'])
            
            except Exception as e:
                self.logger.error(f"Error in yield prediction: {str(e)}")
                raise

        # Get final memory stats
        end_stats = self.memory_manager.get_memory_stats()

        prediction_metrics.calculate_metrics()
        
        state['predictions'] = {
            'runtime': time.time() - start_time,
            'memory_delta': end_stats['delta'],
            'peak_memory': end_stats['peak'],
            'llm_calls': llm_calls,
            'avg_latency': np.mean(latencies) if latencies else 0.0,
            'total_prompt_tokens': total_prompt_tokens,
            'tokens_per_call': total_tokens / llm_calls if llm_calls > 0 else 0.0,
            'mae': prediction_metrics.mae,
            'mape': prediction_metrics.mape,
            'rmse': prediction_metrics.rmse}

        return state
    
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
            system_tokens = self.token_counter.count_tokens(self.system_prompt)
            user_tokens = self.token_counter.count_tokens(context)
            prompt_tokens = system_tokens + user_tokens
            
            # Track total prompt tokens
            self.metrics['prompt_tokens'].append(prompt_tokens)
            
            # Track API call (including retries)
            self.metrics['llm_calls'] += 1
            start_time = time.time()
            
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=context)]
            
            start_time = time.time()
            response = self.llm.invoke(messages)
            end_time = time.time()
            
            latency = end_time - start_time
            self.metrics['latencies'].append(latency)

            if not response or not response.content:
                raise ValueError("No response received")
        
            # Track latency
            latency = time.time() - start_time
            self.metrics['latencies'].append(latency)
            
            # Count response tokens
            response_tokens = self.token_counter.count_tokens(response.content)
                
            # Track total tokens for this interaction
            total_tokens = prompt_tokens + response_tokens
            self.metrics['total_tokens'] = self.metrics.get('total_tokens', 0) + total_tokens
            
            predicted_yield = self._extract_predicated_yield(response.content)

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
    
        # Implement context building logic
    def _build_prediction_context(self, features: Dict[str, Any], dataset: Dict[str, Any]) -> str:
        """Build prediction context from historical data"""
        try:
            df = dataset.df
            # Get num_few_shot from config
            num_few_shot = self.config['benchmark'].get('num_few_shot', 5)
            random_few_shot = self.config['benchmark'].get('random_few_shot', False)
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