from crewai import Crew
from langchain_groq import ChatGroq
import yaml
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
import time
import psutil
import logging
from agents import PredictionAgent, DataPreparationAgent
from tasks import PredictionTask, DataPreparationTask, QuestionLoadingTask
from data_classes import BenchmarkMetrics, IterationMetrics
from typing import Dict, Any, Optional
from transformers import AutoTokenizer
from agents.token_counter import TokenCounter

def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level (default: INFO)
        log_format: Format for log messages
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("crop_yield_predictor")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger 

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_config():
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_llm(config):
    env_path = Path(os.path.expanduser(config['data']['paths']['env']))
    load_dotenv(dotenv_path=env_path, override=True)
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model=config['model']['name'],
        api_key=groq_key,
        temperature=config['model']['temperature'],
        max_tokens=config['model']['max_tokens'],
        max_retries=2
    )

def get_token_counter() -> TokenCounter:
    """Create a token counter for the model"""
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-hf",  # Open source version of Llama-2 tokenizer
        use_fast=True
    )
    return lambda text: len(tokenizer.encode(text))

def run_crew(config: dict, llm: ChatGroq, logger: logging.Logger) -> Dict[str, Any]:
    """Run a single crew iteration and return metrics"""
    # Create token counter
    token_counter = get_token_counter()
    
    # Create agents with token counter
    prep_agent = DataPreparationAgent(llm, logger, config)
    predict_agent = PredictionAgent(
        llm=llm, 
        logger=logger, 
        config=config,
        token_counter=token_counter
    )

    # Create tasks
    prep_task = DataPreparationTask(agent=prep_agent, config=config, logger=logger)
    
    # Make questions task dependent on prep task
    questions_task = QuestionLoadingTask(
        agent=prep_agent, 
        config=config, 
        logger=logger,
        input_tasks=[prep_task]  # Add dependency
    )
    
    # Make predict task dependent on both previous tasks
    predict_task = PredictionTask(
        agent=predict_agent,
        prep_task=prep_task,
        questions_task=questions_task,
        config=config,
        logger=logger,
        input_tasks=[prep_task, questions_task]  # Add dependencies
    )

    crew = Crew(
        agents=[prep_agent, predict_agent],
        tasks=[prep_task, questions_task, predict_task],
        verbose=True
    )

    crew.kickoff()
    
    # Return only the metrics we need
    return {
        'llm_calls': predict_agent.metrics.llm_metrics.call_count,
        'api_latency': predict_agent.metrics.llm_metrics.avg_latency,
        'total_prompt_tokens': predict_agent.metrics.llm_metrics.total_prompt_tokens,
        'tokens_per_call': predict_agent.metrics.llm_metrics.tokens_per_call,
        'mae': predict_task.metrics.prediction_metrics.mae,
        'mape': predict_task.metrics.prediction_metrics.mape,
        'rmse': predict_task.metrics.prediction_metrics.rmse,
        'dataset': prep_task.dataset
    }

def run_benchmark(config: dict, llm: ChatGroq, logger: logging.Logger) -> BenchmarkMetrics:
    benchmark = BenchmarkMetrics(config=config)
    iterations = config['benchmark']['iterations']
    
    for i in range(iterations):
        logger.info(f"\nStarting iteration {i+1}/{iterations}")
        
        # Capture metrics for this iteration
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # Run the crew and get predict_agent from results
        results = run_crew(config, llm, logger)
        
        # Store dataset stats only on first iteration
        if i == 0:
            benchmark.set_dataset_stats(results['dataset'])
        elif i % 4 == 0:
            # Help t0 minimize rate limiting
            time.sleep(60)
        
        # Calculate iteration metrics
        end_time = time.time()
        end_memory = get_memory_usage()
        
        iteration_metrics = IterationMetrics(
            iteration=i+1,
            runtime=end_time - start_time,
            memory_delta=end_memory - start_memory,
            peak_memory=get_memory_usage(),
            llm_calls=results['llm_calls'],
            avg_latency=results['api_latency'],
            total_prompt_tokens=results['total_prompt_tokens'],  # Get directly from results
            tokens_per_call=results['tokens_per_call'],         # Get directly from results
            mae=results['mae'],
            mape=results['mape'],
            rmse=results['rmse']
        )
        benchmark.iterations.append(iteration_metrics)

        # Sleep in between iterations due to rate limits
        time.sleep(60)

    # Save benchmark results
    metrics_dir = Path(config['data']['paths']['metrics'])
    benchmark.save_metrics(metrics_dir, 'crewai')
    
    return benchmark

if __name__ == "__main__":
    config = load_config()
    llm = get_llm(config)
    logger = setup_logging()
    benchmark_results = run_benchmark(config, llm, logger)
    
    # Print summary with clearer metrics
    print("\nBenchmark Summary:")
    print("=" * 50)
    print(f"Model Configuration:")
    print(f"- Name: {benchmark_results.model_name}")
    print(f"- Temperature: {benchmark_results.model_temperature}")
    print(f"- Max Tokens: {benchmark_results.model_max_tokens}")
    print(f"\nFew-Shot Configuration:")
    print(f"- Random Selection: {benchmark_results.random_few_shot}")
    print(f"- Number of Examples: {benchmark_results.num_few_shot}")
    print(f"\nPerformance Metrics:")
    print(f"- Total Iterations: {len(benchmark_results.iterations)}")
    print(f"- Total Runtime: {benchmark_results.avg_runtime:.2f} seconds")
    print(f"- Total LLM Calls: {benchmark_results.total_llm_calls}")
    print(f"- Average API Latency: {benchmark_results.avg_latency:.2f} seconds")
    print(f"- Average Tokens/Call: {benchmark_results.avg_token_count:.1f}")
    print(f"- Total Tokens: {benchmark_results.total_token_count}")
    print(f"- Average MAE: {benchmark_results.avg_mae:.2f}")
    print(f"- Average MAPE: {benchmark_results.avg_mape:.2f}%")
    print(f"- Average RMSE: {benchmark_results.avg_rmse:.2f}") 