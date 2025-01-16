from pydantic import BaseModel, Field, ConfigDict
from typing import List, Tuple, Dict, TYPE_CHECKING, Any
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

if TYPE_CHECKING:
    from .crop_dataset import CropDataset  # Import for type checking

class BaseMetrics(BaseModel):
    """Base metrics tracking"""
    call_count: int = Field(default=0, description="Number of API calls")
    latencies: List[float] = Field(default_factory=list, description="API latencies")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class LLMMetrics(BaseMetrics):
    """Track LLM API calls and latency"""
    prompt_tokens: List[int] = Field(default_factory=list)
    latencies: List[float] = Field(default_factory=list)
    
    @property
    def avg_prompt_tokens(self) -> float:
        """Average tokens per API call"""
        return np.mean(self.prompt_tokens) if self.prompt_tokens else 0.0
    
    @property
    def total_prompt_tokens(self) -> int:
        """Total tokens across all API calls"""
        return sum(self.prompt_tokens)
    
    @property
    def tokens_per_call(self) -> float:
        """Average tokens per API call including retries"""
        return self.total_prompt_tokens / self.call_count if self.call_count > 0 else 0.0
        
    @property
    def avg_latency(self) -> float:
        """Average latency across all API calls"""
        return np.mean(self.latencies) if self.latencies else 0.0

class PredictionMetrics(BaseMetrics):
    """Track prediction accuracy"""
    predictions: List[Tuple[float, float]] = Field(default_factory=list, description="Predicted vs actual values")
    mae: float = Field(default=0.0, description="Mean Absolute Error")
    mape: float = Field(default=0.0, description="Mean Absolute Percentage Error")
    rmse: float = Field(default=0.0, description="Root Mean Square Error")

    def calculate_metrics(self) -> None:
        """Calculate all prediction metrics"""
        if not self.predictions:
            return

        predicted = np.array([p[0] for p in self.predictions])
        actual = np.array([p[1] for p in self.predictions])
        
        # Calculate metrics
        self.mae = np.mean(np.abs(predicted - actual))
        self.mape = np.mean(np.abs((predicted - actual) / actual)) * 100
        self.rmse = np.sqrt(np.mean((predicted - actual) ** 2))

    def plot_performance(self, metrics_dir: str, framework: str) -> None:
        # Set seaborn style globally
        sns.set_style("whitegrid")  # Use seaborn's whitegrid style
        
        # Create figure and axis
        plt.figure(figsize=(10, 6))
        
        # Get actual and predicted values
        actual = [p[1] for p in self.predictions]
        predicted = [p[0] for p in self.predictions]
        
        # Create scatter plot
        plt.scatter(actual, predicted, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Yield')
        plt.ylabel('Predicted Yield')
        plt.title(f'Actual vs Predicted Yield ({framework})\nMAE: {self.mae:.2f}, RMSE: {self.rmse:.2f}')
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(metrics_dir, f'{framework}_performance.png'))
        plt.close()

class IterationMetrics(BaseModel):
    """Metrics for a single iteration"""
    iteration: int = Field(..., description="Iteration number")
    runtime: float = Field(..., description="Total runtime in seconds")
    memory_delta: float = Field(..., description="Memory usage delta in MB")
    peak_memory: float = Field(..., description="Peak memory usage in MB")
    llm_calls: int = Field(..., description="Total LLM API calls")
    avg_latency: float = Field(..., description="Average LLM latency")
    total_prompt_tokens: int = Field(..., description="Total prompt tokens in iteration")
    tokens_per_call: float = Field(..., description="Average tokens per API call")
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    rmse: float = Field(..., description="Root Mean Square Error")

class BenchmarkMetrics(BaseModel):
    """Track benchmark metrics across iterations"""
    model_name: str
    model_temperature: float
    model_max_tokens: int
    random_few_shot: bool
    num_few_shot: int
    iterations: List[IterationMetrics] = Field(default_factory=list)
    dataset_stats: Dict[str, Dict] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            model_name=config['model']['name'],
            model_temperature=config['model']['temperature'],
            model_max_tokens=config['model']['max_tokens'],
            random_few_shot=config['benchmark']['random_few_shot'],
            num_few_shot=config['benchmark']['num_few_shot'],
            iterations=[],
            dataset_stats={},
            timestamp=datetime.now()
        )

    def _print_benchmark_summary(self) -> None:
        # Print summary with metrics
        print("\nBenchmark Summary:")
        print("=" * 50)
        print(f"Model Configuration:")
        print(f"- Name: {self.model_name}")
        print(f"- Temperature: {self.model_temperature}")
        print(f"- Max Tokens: {self.model_max_tokens}")
        print(f"\nFew-Shot Configuration:")
        print(f"- Random Selection: {self.random_few_shot}")
        print(f"- Number of Examples: {self.num_few_shot}")
        print(f"\nPerformance Metrics:")
        print(f"- Total Iterations: {len(self.iterations)}")
        print(f"- Total Runtime: {self.avg_runtime:.2f} seconds")
        print(f"- Total LLM Calls: {self.total_llm_calls}")
        print(f"- Average API Latency: {self.avg_latency:.2f} seconds")
        print(f"- Average Tokens/Call: {self.avg_token_count:.1f}")
        print(f"- Total Tokens: {self.total_token_count}")
        print(f"- Average MAE: {self.avg_mae:.2f}")
        print(f"- Average MAPE: {self.avg_mape:.2f}%")
        print(f"- Average RMSE: {self.avg_rmse:.2f}") 

    def set_dataset_stats(self, dataset: 'CropDataset') -> None:
        """Store dataset statistics"""
        self.dataset_stats = dataset.summary['crop_distribution']

    def save_metrics(self, metrics_dir: Path, framework: str) -> None:
        """Save metrics and create visualizations"""
        metrics_dir.mkdir(exist_ok=True)
        
        # Save JSON metrics
        filename = f"{self.model_name}_{framework}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{len(self.iterations)}.json"
        with open(metrics_dir / filename, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
            
        # Create and save performance plots
        self.plot_performance(metrics_dir, framework)
        self.plot_runtime_metrics(metrics_dir, framework)

    def plot_performance(self, metrics_dir: Path, framework: str) -> None:
        """Create performance visualization"""
        sns.set_theme(style="whitegrid")  # Modern seaborn styling
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'{framework} Crop Yield Performance', fontsize=16, y=0.95)
        
        # Create iteration range starting at 0
        iterations = range(0, len(self.iterations))  # Start at 0
        mae_values = [m.mae for m in self.iterations]
        mape_values = [m.mape for m in self.iterations]
        rmse_values = [m.rmse for m in self.iterations]
        
        # Plot MAE
        ax1 = plt.subplot(2, 2, 1)
        sns.lineplot(x=iterations, y=mae_values, ax=ax1, marker='o')
        ax1.set_title('Mean Absolute Error by Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MAE')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Force whole numbers
        
        # Plot MAPE
        ax2 = plt.subplot(2, 2, 2)
        sns.lineplot(x=iterations, y=mape_values, ax=ax2, marker='o')
        ax2.set_title('Mean Absolute Percentage Error by Iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('MAPE (%)')
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Force whole numbers
        
        # Plot RMSE
        ax3 = plt.subplot(2, 1, 2)
        sns.lineplot(x=iterations, y=rmse_values, ax=ax3, marker='o')
        ax3.set_title('Root Mean Square Error by Iteration')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('RMSE')
        ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Force whole numbers
        
        # Add model info
        plt.figtext(0.02, 0.02, f'Model: {self.model_name}\nTemperature: {self.model_temperature}\nFew-Shot Examples: {self.num_few_shot}', 
                   fontsize=8, ha='left')
        
        # Adjust layout and save
        plt.tight_layout()
        plot_filename = f"{self.model_name}_{framework}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{len(self.iterations)}_performance.png"
        plt.savefig(metrics_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_runtime_metrics(self, metrics_dir: Path, framework: str) -> None:
        """Create runtime performance visualization"""
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{framework} Runtime Performance Metrics', fontsize=16, y=0.95)
        
        # Create iteration range
        iterations = range(1, len(self.iterations) + 1)
        
        # Plot 1: API Latency
        api_latencies = [m.avg_latency for m in self.iterations]
        sns.lineplot(x=iterations, y=api_latencies, ax=ax1, marker='o')
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
        
        # Plot 3: Memory Delta
        memory_deltas = [m.memory_delta for m in self.iterations]
        sns.lineplot(x=iterations, y=memory_deltas, ax=ax3, marker='o')
        ax3.set_title('Memory Usage Delta by Iteration')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Memory Delta (MB)')
        ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Plot 4: Tokens per Iteration
        tokens = [m.tokens_per_call for m in self.iterations]
        sns.lineplot(x=iterations, y=tokens, ax=ax4, marker='o')
        ax4.set_title('Tokens per Call by Iteration')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Tokens per Call')
        ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Add model info
        plt.figtext(0.02, 0.02, 
                    f'Model: {self.model_name}\n'
                    f'Temperature: {self.model_temperature}\n'
                    f'Few-Shot Examples: {self.num_few_shot}', 
                    fontsize=8, ha='left')
        
        # Adjust layout and save
        plt.tight_layout()
        runtime_plot = f"{self.model_name}_{framework}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{len(self.iterations)}_runtime.png"
        plt.savefig(metrics_dir / runtime_plot, dpi=300, bbox_inches='tight')
        plt.close()

    # Properties for aggregate metrics
    @property
    def avg_runtime(self) -> float:
        return np.mean([m.runtime for m in self.iterations]) if self.iterations else 0.0

    @property
    def total_llm_calls(self) -> int:
        return sum(m.llm_calls for m in self.iterations)

    @property
    def avg_latency(self) -> float:
        return np.mean([m.avg_latency for m in self.iterations]) if self.iterations else 0.0

    @property
    def avg_token_count(self) -> float:
        """Average tokens per iteration"""
        return np.mean([m.tokens_per_call for m in self.iterations]) if self.iterations else 0.0

    @property
    def total_token_count(self) -> int:
        """Total tokens across all iterations"""
        return sum(m.total_prompt_tokens for m in self.iterations)

    @property
    def avg_mae(self) -> float:
        return np.mean([m.mae for m in self.iterations]) if self.iterations else 0.0

    @property
    def avg_mape(self) -> float:
        return np.mean([m.mape for m in self.iterations]) if self.iterations else 0.0

    @property
    def avg_rmse(self) -> float:
        return np.mean([m.rmse for m in self.iterations]) if self.iterations else 0.0 