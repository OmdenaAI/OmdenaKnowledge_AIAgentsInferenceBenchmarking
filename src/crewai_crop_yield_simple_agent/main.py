from crewai import Crew
from langchain_groq import ChatGroq
from pathlib import Path
import os
import time
import logging
import argparse
from agents import PredictionAgent, DataPreparationAgent
from tasks import PredictionTask, DataPreparationTask, QuestionLoadingTask
from simple_agent_common.data_classes import BenchmarkMetrics, IterationMetrics
from simple_agent_common.utils import MemoryManager, load_env_vars, load_config, setup_logging
from typing import Dict, Any, Optional, Callable
from simple_agent_common.utils.token_counter import TokenCounter

def get_llm(config):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model=config['model']['name'],
        api_key=groq_key,
        temperature=config['model']['temperature'],
        max_tokens=config['model']['max_tokens']
    )

def get_token_counter() -> Callable[[str], int]:
    """Create a token counter for the model"""
    counter = TokenCounter()
    return counter.count_tokens

def run_crew(config: dict, llm: ChatGroq, logger: logging.Logger, memory_manager: MemoryManager) -> Dict[str, Any]:
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
        memory_manager=memory_manager,
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
    
    # Initialize memory tracking
    memory_manager = MemoryManager()
    memory_manager.start_tracking()

    for i in range(iterations):
        logger.info(f"\nStarting iteration {i+1}/{iterations}")
        
        # Capture metrics for this iteration
        start_time = time.time()
        start_stats = memory_manager.get_memory_stats()
        
        # Run the crew and get predict_agent from results
        results = run_crew(config, llm, logger, memory_manager)
        
        # Store dataset stats only on first iteration
        if i == 0:
            benchmark.set_dataset_stats(results['dataset'])
        
        # Calculate iteration metrics
        end_time = time.time()
        
        # Get final memory stats
        end_stats = memory_manager.get_memory_stats()
        
        iteration_metrics = IterationMetrics(
            iteration=i+1,
            runtime=end_time - start_time,
            memory_delta=end_stats['delta'],
            peak_memory=end_stats['peak'],
            llm_calls=results['llm_calls'],
            avg_latency=results['api_latency'],
            total_prompt_tokens=results['total_prompt_tokens'],
            tokens_per_call=results['tokens_per_call'],
            mae=results['mae'],
            mape=results['mape'],
            rmse=results['rmse']
        )
        benchmark.iterations.append(iteration_metrics)

        memory_manager.reset_tracking()

    # Save benchmark results
    metrics_dir = Path(config['data']['paths']['metrics'])
    benchmark.save_metrics(metrics_dir, 'crewai')
    
    return benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the benchmark process')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the yaml config file')
    args = parser.parse_args()
    
    config = load_config(config_location=args.config)
    load_env_vars(config)
    llm = get_llm(config)
    logger = setup_logging(framework_name="crewai")
    benchmark_results = run_benchmark(config, llm, logger)