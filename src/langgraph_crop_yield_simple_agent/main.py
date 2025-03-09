from pathlib import Path
import os
import argparse
from orchestrator import run_workflow, build_graph
from simple_agent_common.utils import load_env_vars, load_config, setup_logging
from langchain_groq import ChatGroq
from simple_agent_common.data_classes import BenchmarkMetrics, IterationMetrics
from simple_agent_common.utils import MemoryManager, load_env_vars, load_config, setup_logging
from typing import Dict, Any, List
import logging

FRAMEWORK='langgraph'

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

def run_benchmark(config: Dict[str, Any], llm: ChatGroq, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Run the benchmark process with iterations"""
    logger.debug("Starting benchmark run...")

    benchmark = BenchmarkMetrics(framework=FRAMEWORK, config=config)

    # Get number of iterations from config
    num_iterations = config['benchmark'].get('iterations', 3)
    logger.info(f"Running benchmark for {num_iterations} iterations")
    
    # Initialize memory tracking
    memory_manager = MemoryManager()
    memory_manager.start_tracking()

    app = build_graph(config, logger, llm, memory_manager)
            
    # Run iterations
    for iteration in range(num_iterations):
        logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")
        
        run_result = run_workflow(app)
                
        iteration_result = IterationMetrics(
            iteration=iteration+1,
            runtime=run_result['predictions']['runtime'],
            memory_delta=run_result['predictions']['memory_delta'],
            peak_memory=run_result['predictions']['peak_memory'],
            llm_calls=run_result['predictions']['llm_calls'],
            avg_latency=run_result['predictions']['avg_latency'],
            total_prompt_tokens=run_result['predictions']['total_prompt_tokens'],
            tokens_per_call=run_result['predictions']['tokens_per_call'],
            mae=run_result['predictions']['mae'],
            mape=run_result['predictions']['mape'],
            rmse=run_result['predictions']['rmse']
        )

        benchmark.iterations.append(iteration_result)
        logger.info(f"Completed iteration {iteration + 1}")

        memory_manager.reset_tracking()
    
    # Save metrics for all iterations
    metrics_dir = Path(config['data']['paths']['metrics'])
    benchmark.save_metrics(metrics_dir, FRAMEWORK)
    
    return benchmark

def main():
    """Main entry point for the application."""
    try:
        parser = argparse.ArgumentParser(description='Run the benchmark process')
        parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the yaml config file')
        args = parser.parse_args()
        
        config = load_config(config_location=args.config)
        load_env_vars(config)
        logger = setup_logging(framework_name=FRAMEWORK)

        llm = get_llm(config)
        
        # Run workflow
        run_benchmark(config, llm, logger)  
        
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main()