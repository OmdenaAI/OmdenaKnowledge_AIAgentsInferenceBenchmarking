from agents import DataPreparationAgent, PredictionAgent
import logging
from typing import Dict, Any, List
from pathlib import Path
import os
import time
import numpy as np
import argparse
from simple_agent_common.data_classes import BenchmarkMetrics, IterationMetrics, PredictionMetrics
from simple_agent_common.utils import RateLimiter, MemoryManager, load_env_vars, load_config, setup_logging, TokenCounter

# Set the environment variable to disable Docker
os.environ['AUTOGEN_USE_DOCKER'] = 'False'
FRAMEWORK = "autogen"

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
        "max_tokens": config['model']['max_tokens'],
        "cache_seed": None # Ensure cache is disabled
    }]

def run_benchmark(config: Dict[str, Any], logger: logging.Logger) -> List[Dict[str, Any]]:
    """Run the benchmark process with iterations"""
    logger.debug("Starting benchmark run...")

    benchmark = BenchmarkMetrics(framework=FRAMEWORK, config=config)
    
    # Get number of iterations from config
    num_iterations = config['benchmark'].get('iterations', 3)
    logger.info(f"Running benchmark for {num_iterations} iterations")
    
    # Get Groq configuration and setup agents
    config_list = get_groq_config(config, logger)
    agents = setup_agents(config, logger, config_list)
    
    # Setup rate limiter
    max_calls = config.get("model", {}).get("max_calls", 4)
    pause_time = config.get("model", {}).get("pause_time", 30)
    token_limit = config.get("model", {}).get("token_limit", 90000)

    rate_limiter = RateLimiter(
            max_calls=max_calls,  # More conservative
            pause_time=pause_time,  # Groq's window
            token_limit=token_limit  # Buffer below Groq's 100k limit
        )
    
    # Load and prepare data once
    data_paths = Path(config['data']['paths']['crop_data'])
    if not data_paths.exists():
        raise FileNotFoundError(f"Data file not found: {data_paths}")
          
    # Initialize memory tracking
    memory_manager = MemoryManager()
    memory_manager.start_tracking()
    
    # Run iterations
    for iteration in range(num_iterations):
        logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")
        start_time = time.time()
        
        # Get initial memory state
        start_stats = memory_manager.get_memory_stats()
        
        dataset = agents['data_agent'].prepare_data(data_paths)
        questions = agents['data_agent'].load_questions()
        logger.info(f"Loaded {len(questions)} questions via DataPreparationAgent")
        
        # Lists to track predictions and metrics
        latencies = []
        
        llm_calls = 0
        total_prompt_tokens = 0
        total_tokens = 0

        prediction_metrics = PredictionMetrics()

        # Run predictions
        for question in questions:
            with rate_limiter:
                try:
                    # Track memory around prediction
                    memory_stats = memory_manager.get_memory_stats()
                    
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
                    llm_calls += (prediction_result['retry_count'] + 1)
                    total_prompt_tokens += prediction_result['prompt_tokens']
                    total_tokens += prediction_result['total_tokens']
                    
                    prediction_metrics.predictions.append((prediction_result['predicted_yield'], features['Yield'], None))

                    latencies.append(prediction_result['latency'])
                    
                    # Update peak memory after prediction
                    memory_stats = memory_manager.get_memory_stats()
                    
                except Exception as e:
                    logger.error(f"Prediction failed: {str(e)}")
                    continue
        
        # Get final memory stats
        end_stats = memory_manager.get_memory_stats()

        prediction_metrics.calculate_metrics()
        
        iteration_result = IterationMetrics(
            iteration=iteration+1,
            runtime=time.time() - start_time,
            memory_delta=end_stats['delta'],
            peak_memory=end_stats['peak'],
            llm_calls=llm_calls,
            avg_latency=np.mean(latencies) if latencies else 0.0,
            total_prompt_tokens=total_prompt_tokens,
            tokens_per_call=total_tokens / llm_calls if llm_calls > 0 else 0.0,
            mae=prediction_metrics.mae,
            mape=prediction_metrics.mape,
            rmse=prediction_metrics.rmse
        )        
        benchmark.iterations.append(iteration_result)
        logger.info(f"Completed iteration {iteration + 1}")

        memory_manager.reset_tracking()
    
    # Save metrics for all iterations
    metrics_dir = Path(config['data']['paths']['metrics'])
    benchmark.save_metrics(metrics_dir,FRAMEWORK)
    
    return benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the benchmark process')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the yaml config file')
    args = parser.parse_args()
    
    config = load_config(config_location=args.config)
    load_env_vars(config)
    logger = setup_logging(framework_name=FRAMEWORK)
    
    results = run_benchmark(config, logger)