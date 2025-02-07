import warnings
import random
# Disable specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
from agents.router import QueryRouter
from agents.math_agent import MathAgent
from agents.physics_agent import PhysicsAgent
from agents.chemistry_agent import ChemistryAgent
from agents.biology_agent import BiologyAgent
from orchestrator import MultiAgentOrchestrator
from data.dataset import Dataset
from simple_agent_common.data_classes import BenchmarkMetrics, IterationMetrics, PredictionMetrics
from simple_agent_common.utils import RateLimiter, MemoryManager, load_env_vars, load_config, setup_logging
import logging
import random
from typing import Dict
import numpy as np
from pathlib import Path
import time

def run_iteration(config: Dict, logger: logging.Logger, iteration: int, dataset: Dataset, memory_manager: MemoryManager) -> Dict:
    # Initialize components
    router = QueryRouter(logger, config)
    agents = {
        "math": MathAgent(logger, config),
        "physics": PhysicsAgent(logger, config),
        "chemistry": ChemistryAgent(logger, config),
        "biology": BiologyAgent(logger, config)
    }
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(router, agents)
    
    questions = dataset.get_questions()
    random.shuffle(questions)  # Shuffle the list of questions
    
    # Initialize metrics
    prediction_metrics = PredictionMetrics()
    
    # Process questions
    latencies = []
        
    llm_calls = 0
    total_prompt_tokens = 0
    total_tokens = 0

    prediction_metrics = PredictionMetrics()
    
    # Setup rate limiter
    max_calls = config.get("model", {}).get("max_calls", 4)
    pause_time = config.get("model", {}).get("pause_time", 30)
    token_limit = config.get("model", {}).get("token_limit", 90000)

    rate_limiter = RateLimiter(
            max_calls=max_calls,  # More conservative
            pause_time=pause_time,  # Groq's window
            token_limit=token_limit  # Buffer below Groq's 100k limit
        )

    # Get initial memory state
    start_stats = memory_manager.get_memory_stats()
    
    start_time = time.time()

    for question in questions:
        with rate_limiter:
            try:
                # Track memory around prediction
                memory_stats = memory_manager.get_memory_stats()

                # Execute through orchestrator
                result = orchestrator.run(question["question"])        

                # Update tracking
                llm_calls += (result['retry_count'] + 1)
                total_prompt_tokens += result['prompt_tokens']
                total_tokens += result['total_tokens']
                            
                agent = result['agent']
                prediction_metrics.predictions.append((result['predicted_yield'], question['answer'], agent))
                latencies.append(result['latency'])

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
            rmse=prediction_metrics.rmse,
            group_metrics=prediction_metrics.group_metrics
        )
           
    return iteration_result

def run_benchmark(config: Dict, logger: logging.Logger, iterations: int, dataset: Dataset) -> BenchmarkMetrics:
    iterations_data = []
    memory_manager = MemoryManager()
    benchmark = BenchmarkMetrics(config=config)
    
    memory_manager.start_tracking()

    for iteration in range(iterations):
        logger.info(f"Starting iteration {iteration + 1}/{iterations}")
        
        iteration_results = run_iteration(config, logger, iteration, dataset, memory_manager)
        
        benchmark.iterations.append(iteration_results)
        logger.info(f"Completed iteration {iteration + 1}")

    return benchmark

def main():
    config = load_config(required_paths = ['env', 'metrics', 'input_dir'], 
                         required_benchmark = ['iterations', 'total_questions'])
    load_env_vars(config)
    logger = setup_logging(framework_name="langgraph_multi_agent")
    iterations = config.get("benchmark", {}).get("iterations", 1)
    
    # Initialize dataset once for all iterations
    dataset = Dataset(
        data_dir=config["data"]["paths"]["input_dir"],
        total_questions=config["benchmark"]["total_questions"]
    )
    
    benchmark = run_benchmark(config, logger, iterations, dataset)

    # Save metrics for all iterations
    metrics_dir = Path(config['data']['paths']['metrics'])

    benchmark.save_metrics(metrics_dir, 'langgraph_multi_agent')

    # Save questions
    question_filename = f"{benchmark.model_name}_langgraph_multi_agent_{benchmark.timestamp.strftime('%Y%m%d_%H%M%S')}_questions.jsonl"
    dataset.save_questions(metrics_dir, question_filename)


if __name__ == "__main__":
    main() 