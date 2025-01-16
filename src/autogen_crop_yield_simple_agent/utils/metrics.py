from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class IterationMetrics(BaseModel):
    """Single iteration metrics"""
    iteration: int
    runtime: float
    memory_delta: float
    peak_memory: float
    llm_calls: int
    avg_latency: float
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    mae: float
    mape: float
    rmse: float

class MetricsHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_path = Path(config['data']['paths']['metrics'])
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.model_name = config['model']['name']
        self.iterations: List[IterationMetrics] = []
        self.timestamp = datetime.now()

    def save_metrics(self, results: List[Dict[str, Any]], framework: str) -> None:
        """Save metrics and create visualizations"""
        # Calculate metrics for each iteration
        for i, result in enumerate(results):
            metrics = self._calculate_iteration_metrics(i, result)
            self.iterations.append(metrics)

        # Save JSON metrics
        self._save_json_metrics(framework)
        
        # Create visualizations
        self._plot_performance_metrics(framework)
        self._plot_runtime_metrics(framework)

    def _plot_performance_metrics(self, framework: str) -> None:
        """Create performance visualization"""
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'{framework} Crop Yield Performance', fontsize=16, y=0.95)
        
        # Create iteration range
        iterations = range(len(self.iterations))
        mae_values = [m.mae for m in self.iterations]
        mape_values = [m.mape for m in self.iterations]
        rmse_values = [m.rmse for m in self.iterations]
        
        # Plot MAE
        ax1 = plt.subplot(2, 2, 1)
        sns.lineplot(x=iterations, y=mae_values, ax=ax1, marker='o')
        ax1.set_title('Mean Absolute Error by Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MAE')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Plot MAPE
        ax2 = plt.subplot(2, 2, 2)
        sns.lineplot(x=iterations, y=mape_values, ax=ax2, marker='o')
        ax2.set_title('Mean Absolute Percentage Error by Iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('MAPE (%)')
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Plot RMSE
        ax3 = plt.subplot(2, 1, 2)
        sns.lineplot(x=iterations, y=rmse_values, ax=ax3, marker='o')
        ax3.set_title('Root Mean Square Error by Iteration')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('RMSE')
        ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Add model info
        plt.figtext(0.02, 0.02, 
                    f'Model: {self.model_name}\n'
                    f'Temperature: {self.config["model"]["temperature"]}\n'
                    f'Few-Shot Examples: {self.config["benchmark"]["num_few_shot"]}', 
                    fontsize=8, ha='left')
        
        # Save plot
        plot_filename = f"{self.model_name}_{framework}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_performance.png"
        plt.savefig(self.metrics_path / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_runtime_metrics(self, framework: str) -> None:
        """Create runtime performance visualization"""
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{framework} Runtime Performance Metrics', fontsize=16, y=0.95)
        
        # Create iteration range
        iterations = range(len(self.iterations))
        
        # Plot 1: API Latency
        latencies = [m.avg_latency for m in self.iterations]
        sns.lineplot(x=iterations, y=latencies, ax=ax1, marker='o')
        ax1.set_title('API Latency by Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('API Latency (seconds)')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Plot 2: Overall Runtime
        runtimes = [m.runtime for m in self.iterations]
        sns.lineplot(x=iterations, y=runtimes, ax=ax2, marker='o')
        ax2.set_title('Total Runtime by Iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Plot 3: Memory Usage
        memory = [m.memory_delta for m in self.iterations]
        sns.lineplot(x=iterations, y=memory, ax=ax3, marker='o')
        ax3.set_title('Memory Usage Delta by Iteration')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Memory Delta (MB)')
        ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Plot 4: Token Usage
        tokens = [m.total_tokens for m in self.iterations]
        sns.lineplot(x=iterations, y=tokens, ax=ax4, marker='o')
        ax4.set_title('Total Tokens by Iteration')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Total Tokens')
        ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Add model info
        plt.figtext(0.02, 0.02, 
                    f'Model: {self.model_name}\n'
                    f'Temperature: {self.config["model"]["temperature"]}\n'
                    f'Few-Shot Examples: {self.config["benchmark"]["num_few_shot"]}', 
                    fontsize=8, ha='left')
        
        # Save plot
        runtime_plot = f"{self.model_name}_{framework}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_runtime.png"
        plt.savefig(self.metrics_path / runtime_plot, dpi=300, bbox_inches='tight')
        plt.close() 

    def _calculate_iteration_metrics(self, iteration: int, result: Dict[str, Any]) -> IterationMetrics:
        """Calculate metrics for a single iteration"""
        return IterationMetrics(
            iteration=result['iteration'],
            runtime=result['runtime'],
            memory_delta=result['memory_delta'],
            peak_memory=result['peak_memory'],
            llm_calls=result['llm_calls'],
            avg_latency=result['avg_latency'],
            prompt_tokens=result['total_prompt_tokens'],
            response_tokens=0,  # We'll need to add this to the main.py tracking
            total_tokens=result['total_prompt_tokens'],  # Sum of prompt and response tokens
            mae=result['mae'],
            mape=result['mape'],
            rmse=result['rmse']
        )

    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all iterations"""
        if not self.iterations:
            return {}
        
        return {
            'model_info': {
                'name': self.model_name,
                'temperature': self.config['model']['temperature'],
                'max_tokens': self.config['model']['max_tokens'],
                'num_few_shot': self.config['benchmark']['num_few_shot'],
                'random_few_shot': self.config['benchmark']['random_few_shot']
            },
            'performance_metrics': {
                'mae': np.mean([m.mae for m in self.iterations]),
                'mape': np.mean([m.mape for m in self.iterations]),
                'rmse': np.mean([m.rmse for m in self.iterations])
            },
            'runtime_metrics': {
                'total_runtime': sum(m.runtime for m in self.iterations),
                'avg_runtime': np.mean([m.runtime for m in self.iterations]),
                'total_llm_calls': sum(m.llm_calls for m in self.iterations),
                'avg_latency': np.mean([m.avg_latency for m in self.iterations])
            },
            'token_metrics': {
                'total_prompt_tokens': sum(m.prompt_tokens for m in self.iterations),
                'total_response_tokens': sum(m.response_tokens for m in self.iterations),
                'total_tokens': sum(m.total_tokens for m in self.iterations),
                'avg_tokens_per_call': np.mean([m.total_tokens for m in self.iterations])
            },
            'memory_metrics': {
                'peak_memory_mb': max(m.peak_memory for m in self.iterations),
                'avg_memory_delta_mb': np.mean([m.memory_delta for m in self.iterations])
            }
        }

    def _print_benchmark_summary(self, metrics: Dict[str, Any]) -> None:
    # Print summary with clearer metrics
        print("\nBenchmark Summary:")
        print("=" * 50)
        print(f"Model Configuration:")
        print(f"- Name: {metrics['model_info']['name']}")
        print(f"- Temperature: {metrics['model_info']['temperature']}")
        print(f"- Max Tokens: {metrics['model_info']['max_tokens']}")
        print(f"\nFew-Shot Configuration:")
        print(f"- Random Selection: {metrics['model_info']['random_few_shot']}")
        print(f"- Number of Examples: {metrics['model_info']['num_few_shot']}")
        print(f"\nPerformance Metrics:")
        print(f"- Total Iterations: {self.iterations}")
        print(f"- Total Runtime: {metrics['runtime_metrics']['avg_runtime']:.2f} seconds")
        print(f"- Total LLM Calls: {metrics['runtime_metrics']['total_llm_calls']}")
        print(f"- Average API Latency: {metrics['runtime_metrics']['avg_latency']:.2f} seconds")
        print(f"- Average Tokens/Call: {metrics['token_metrics']['avg_tokens_per_call']:.1f}")
        print(f"- Total Tokens: {metrics['token_metrics']['total_tokens']}")
        print(f"- Average MAE: {metrics['performance_metrics']['mae']:.2f}")
        print(f"- Average MAPE: {metrics['performance_metrics']['mape']:.2f}%")
        print(f"- Average RMSE: {metrics['performance_metrics']['rmse']:.2f}") 

    def _save_json_metrics(self, framework: str) -> None:
        """Save metrics to JSON file"""
        metrics = self._calculate_aggregate_metrics()

        self._print_benchmark_summary(metrics)
        
        # Add raw iteration data
        metrics['iterations'] = [m.model_dump() for m in self.iterations]
        metrics['timestamp'] = self.timestamp.isoformat()
        metrics['framework'] = framework
        
        # Save to file
        filename = f"{self.model_name}_{framework}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(self.metrics_path / filename, 'w') as f:
            json.dump(metrics, f, indent=2) 