from typing import Dict, Any
import logging
from pathlib import Path
from ..data_classes import BenchmarkMetrics, IterationMetrics, PredictionMetrics
from ..utils import MemoryManager, RateLimiter, Dataset, extract_number
from .orchestrator_base import OrchestratorBase
import time
import numpy as np

class BenchmarkRunner:
    def __init__(self, benchmark_name: str, orchestrator: OrchestratorBase, config: Dict, logger: logging.Logger):
        self.benchmark_name = benchmark_name
        self.orchestrator = orchestrator
        self.config = config
        self.logger = logger
        self.iterations = self.config.get("benchmark", {}).get("iterations", 1)

    def run_iteration(self, iteration: int, dataset: Dataset, memory_manager: MemoryManager) -> IterationMetrics:
        """Run a single benchmark iteration with proper memory tracking"""
        start_time = time.time()
        peak_memory = 0
        
        questions = dataset.get_questions()
        prediction_metrics = PredictionMetrics()
        
        # Setup rate limiter
        max_calls = self.config.get("model", {}).get("max_calls", 4)
        pause_time = self.config.get("model", {}).get("pause_time", 30)
        token_limit = self.config.get("model", {}).get("token_limit", 90000)
        
        rate_limiter = RateLimiter(
            max_calls=max_calls,
            pause_time=pause_time,
            token_limit=token_limit
        )
        
        # Initialize metrics tracking
        llm_calls = 0
        total_prompt_tokens = 0
        total_tokens = 0
        latencies = []
        
        for question in questions:
            with rate_limiter:
                try:
                    result = self.orchestrator.run(question["question"])
                    
                    # Update tracking
                    llm_calls += (result['retry_count'] + 1)
                    total_prompt_tokens += result['prompt_tokens']
                    total_tokens += result['total_tokens']
                    
                    agent = result['agent']
                    predicted = result['predicted_yield']  # Already a Decimal
                    actual = question['answer'] # Direct conversion
                                       
                    prediction_metrics.predictions.append((predicted, actual, agent))
                    latencies.append(result['latency'])

                    # Track peak memory during processing
                    current_stats = memory_manager.get_memory_stats()
                    peak_memory = max(peak_memory, current_stats['current'])

                    self.logger.info(f"\tPrediction: {predicted}, Actual: {actual}")
                
                except Exception as e:
                    self.logger.error(f"Prediction failed: {str(e)}")
                    continue

        # Get final memory stats
        end_stats = memory_manager.get_memory_stats()
        prediction_metrics.calculate_metrics()

        return IterationMetrics(
            iteration=iteration+1,
            runtime=time.time() - start_time,
            memory_delta=end_stats['delta'],
            peak_memory=peak_memory,
            llm_calls=llm_calls,
            avg_latency=np.mean(latencies) if latencies else 0.0,
            total_prompt_tokens=total_prompt_tokens,
            tokens_per_call=total_tokens / llm_calls if llm_calls > 0 else 0.0,
            mae=prediction_metrics.mae,
            mape=prediction_metrics.mape,
            rmse=prediction_metrics.rmse,
            group_metrics=prediction_metrics.group_metrics
        )

    def run_benchmark(self, iterations: int, dataset: Dataset) -> BenchmarkMetrics:
        """Run complete benchmark with memory tracking"""
        benchmark = BenchmarkMetrics(config=self.config)
        memory_manager = MemoryManager()
        
        memory_manager.start_tracking()
        self.logger.info("Started memory tracking for benchmark")
        
        for iteration in range(iterations):
            self.logger.info(f"Starting iteration {iteration + 1}/{iterations}")
            
            # Reset tracking for clean iteration stats while maintaining overall peak
            if iteration > 0:  # Don't reset before first iteration
                memory_manager.reset_tracking()
                
            iteration_results = self.run_iteration(iteration, dataset, memory_manager)
            benchmark.iterations.append(iteration_results)
            self.logger.info(f"Completed iteration {iteration + 1}")

        # Get final memory stats for the entire benchmark
        final_stats = memory_manager.get_memory_stats()
        benchmark.memory_delta = final_stats['delta']
        benchmark.peak_memory = final_stats['overall_peak']  # Use overall peak
        
        self.logger.info(f"Benchmark memory delta: {benchmark.memory_delta:.2f}MB")
        self.logger.info(f"Benchmark peak memory: {benchmark.peak_memory:.2f}MB")
        
        return benchmark

    def run(self):       
        iterations = self.config.get("benchmark", {}).get("iterations", 1)

        dataset = Dataset(
            data_dir=self.config["data"]["paths"]["input_dir"],
            total_questions=self.config["benchmark"]["total_questions"]
        )
        
        benchmark = self.run_benchmark(iterations, dataset)
        
        metrics_dir = Path(self.config["data"]["paths"]["metrics"])
        benchmark.save_metrics(metrics_dir, self.benchmark_name)

        # Save questions
        question_filename = f"{benchmark.model_name}_{self.benchmark_name}_{benchmark.timestamp.strftime('%Y%m%d_%H%M%S')}_questions.jsonl"
        dataset.save_questions(metrics_dir, question_filename)