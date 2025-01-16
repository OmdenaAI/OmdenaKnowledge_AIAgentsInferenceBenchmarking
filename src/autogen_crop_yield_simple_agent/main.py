import autogen
from agents import DataPreparationAgent, PredictionAgent
from agents.token_counter import TokenCounter
from utils.rate_limiter import RateLimiter
from utils.logging import setup_logging
from utils.config import load_config
from utils.metrics import MetricsHandler
from utils.env import load_env_vars
import logging
from typing import Dict, Any, List
from pathlib import Path
import os
import time
import numpy as np
import psutil


def setup_agents(config: Dict[str, Any], logger: logging.Logger, config_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Setup and configure all agents"""
    logger.debug("Setting up agents...")
    token_counter = TokenCounter()
    
    # Initialize data preparation agent
    data_agent = DataPreparationAgent(
        config=config,
        logger=logger
    )
    
    # Initialize prediction agent
    prediction_agent = PredictionAgent(
        config=config,
        logger=logger,
        token_counter=token_counter,
        llm_config={"config_list": config_list}
    )
    
    return {
        'data_agent': data_agent,
        'prediction_agent': prediction_agent
    }

def get_groq_config(config: Dict[str, Any], logger: logging.Logger) -> List[Dict[str, Any]]:
    """Get Groq configuration"""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    logger.info("GROQ API key loaded successfully")
    
    return [{
        "model": config['model']['name'],
        "api_key": groq_key,
        "base_url": "https://api.groq.com/openai/v1",
        "temperature": config['model']['temperature'],
        "max_tokens": config['model']['max_tokens']
    }]

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_benchmark(config: Dict[str, Any], logger: logging.Logger) -> List[Dict[str, Any]]:
    """Run the benchmark process with iterations"""
    logger.debug("Starting benchmark run...")
    iteration_metrics = []
    
    # Get number of iterations from config
    num_iterations = config['benchmark'].get('iterations', 3)
    logger.info(f"Running benchmark for {num_iterations} iterations")
    
    # Get Groq configuration and setup agents
    config_list = get_groq_config(config, logger)
    agents = setup_agents(config, logger, config_list)
    
    # Initialize handlers
    metrics_handler = MetricsHandler(config)
    rate_limiter = RateLimiter(max_calls=5, pause_time=20)
    
    # Load and prepare data once
    data_paths = Path(config['data']['paths']['crop_data'])
    if not data_paths.exists():
        raise FileNotFoundError(f"Data file not found: {data_paths}")
          
    # Run iterations
    for iteration in range(num_iterations):
        logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")
        start_time = time.time()
        start_memory = get_memory_usage()
        peak_memory = start_memory

        dataset = agents['data_agent'].prepare_data(data_paths)
        questions = agents['data_agent'].load_questions()
        logger.info(f"Loaded {len(questions)} questions via DataPreparationAgent")
        
        # Track metrics for this iteration
        iteration_result = {
            'iteration': iteration + 1,
            'runtime': 0.0,
            'memory_delta': 0.0,
            'peak_memory': 0.0,
            'llm_calls': 0,
            'avg_latency': 0.0,
            'total_prompt_tokens': 0,
            'tokens_per_call': 0,
            'mae': 0.0,
            'mape': 0.0,
            'rmse': 0.0
        }
        
        # Lists to track predictions and metrics
        predictions = []
        actuals = []
        latencies = []
        
        # Run predictions
        for question in questions:
            with rate_limiter:
                try:
                    # Track memory before prediction
                    current_memory = get_memory_usage()
                    peak_memory = max(peak_memory, current_memory)
                    
                    features = question['features']
                    context = agents['prediction_agent']._build_prediction_context(
                        features=features,
                        dataset=dataset
                    )
                    
                    prediction_result = agents['prediction_agent'].predict_yield(
                        features=features,
                        context=context
                    )
                    
                    # Update tracking
                    iteration_result['llm_calls'] += 1
                    iteration_result['total_prompt_tokens'] += prediction_result['prompt_tokens']
                    iteration_result['total_response_tokens'] = iteration_result.get('total_response_tokens', 0) + prediction_result['response_tokens']
                    
                    predictions.append(prediction_result['predicted_yield'])
                    actuals.append(question['actual_yield'])
                    latencies.append(prediction_result['latency'])
                    
                    # Track memory after prediction
                    current_memory = get_memory_usage()
                    peak_memory = max(peak_memory, current_memory)
                    
                except Exception as e:
                    logger.error(f"Prediction failed in iteration {iteration + 1}: {str(e)}")
                    continue
        
        # Calculate final metrics
        end_memory = get_memory_usage()
        runtime = time.time() - start_time
        
        iteration_result.update({
            'runtime': runtime,
            'memory_delta': end_memory - start_memory,
            'peak_memory': peak_memory,
            'avg_latency': np.mean(latencies) if latencies else 0.0,
            'tokens_per_call': iteration_result['total_prompt_tokens'] / iteration_result['llm_calls'] if iteration_result['llm_calls'] > 0 else 0.0,
            'mae': np.mean(np.abs(np.array(predictions) - np.array(actuals))) if predictions else 0.0,
            'mape': np.mean(np.abs((np.array(predictions) - np.array(actuals)) / np.array(actuals))) * 100 if predictions else 0.0,
            'rmse': np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2)) if predictions else 0.0
        })
        
        iteration_metrics.append(iteration_result)
        logger.info(f"Completed iteration {iteration + 1}")
    
    # Save metrics for all iterations
    metrics_handler.save_metrics(iteration_metrics, framework="autogen")
    return iteration_metrics

if __name__ == "__main__":
    config = load_config()
    load_env_vars(config)
    logger = setup_logging()
    
    results = run_benchmark(config, logger)