from pydantic import BaseModel, Field, ConfigDict
from typing import List, Tuple, Dict, TYPE_CHECKING, Any, Optional
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
    predictions: List[Tuple[float, float, str]] = Field(default_factory=list, description="Predicted vs actual values with group label")
    mae: float = Field(default=0.0, description="Mean Absolute Error")
    mape: float = Field(default=0.0, description="Mean Absolute Percentage Error")
    rmse: float = Field(default=0.0, description="Root Mean Square Error")
    group_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Metrics grouped by group label")

    def _calculate_metric_set(self, predicted: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        """Calculate a set of metrics for the given predictions and actual values"""
        return {
            'mae': np.mean(np.abs(predicted - actual)),
            'mape': np.mean(np.abs((predicted - actual) / actual)) * 100,
            'rmse': np.sqrt(np.mean((predicted - actual) ** 2))
        }

    def calculate_metrics(self) -> None:
        """Calculate all prediction metrics, both overall and grouped by the third parameter if present"""
        if not self.predictions:
            return

        # Calculate overall metrics
        predicted = np.array([p[0] for p in self.predictions])
        actual = np.array([p[1] for p in self.predictions])
        
        # Calculate overall metrics
        metrics = self._calculate_metric_set(predicted, actual)
        self.mae = metrics['mae']
        self.mape = metrics['mape']
        self.rmse = metrics['rmse']

        # Group predictions
        groups: Dict[str, List[Tuple[float, float]]] = {}
        for pred, act, group in self.predictions:
            if group not in groups:
                groups[group] = []
            groups[group].append((pred, act))

        # Calculate metrics for each group
        self.group_metrics = {}
        for group, group_predictions in groups.items():
            group_predicted = np.array([p[0] for p in group_predictions])
            group_actual = np.array([p[1] for p in group_predictions])
            self.group_metrics[group] = self._calculate_metric_set(group_predicted, group_actual)

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
    group_metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Optional group-wise metrics. Only populated when predictions contain group labels."
    )

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
    memory_delta: float = Field(default=0.0, description="Total memory delta across entire benchmark in MB")
    peak_memory: float = Field(default=0.0, description="Peak memory usage across entire benchmark in MB")
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            model_name=config['model']['name'],
            model_temperature=config['model']['temperature'],
            model_max_tokens=config['model']['max_tokens'],
            random_few_shot=config.get('benchmark', {}).get('random_few_shot', False),
            num_few_shot=config.get('benchmark', {}).get('num_few_shot', 0),
            iterations=[],
            dataset_stats={},
            timestamp=datetime.now(),
            memory_delta=0.0,
            peak_memory=0.0
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
        print(f"- Average MAE: {self.avg_mae:.2e}")
        print(f"- Average MAPE: {self.avg_mape:.2e}%")
        print(f"- Average RMSE: {self.avg_rmse:.2e}")

    def set_dataset_stats(self, dataset: 'CropDataset') -> None:
        """Store dataset statistics"""
        self.dataset_stats = dataset.summary['crop_distribution']

    def save_metrics(self, metrics_dir: Path, framework: str) -> None:

        # Print a summary of the metrics
        self._print_benchmark_summary()

        """Save metrics and create visualizations"""
        metrics_dir.mkdir(exist_ok=True)
        
        # Save JSON metrics
        filename = f"{self.model_name}_{framework}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{len(self.iterations)}.json"
        with open(metrics_dir / filename, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
            
        # Create and save performance plots
        self.plot_performance(metrics_dir, framework)
        self.plot_runtime_metrics(metrics_dir, framework)
        self.plot_agent_performance(metrics_dir, framework)

    def plot_performance(self, metrics_dir: Path, framework: str) -> None:
        """Create performance visualization"""
        sns.set_theme(style="whitegrid")  # Modern seaborn styling
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'{framework} Inference Performance', fontsize=16, y=1.02)
        
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
        
        # Force y-axis to show actual runtime values
        min_runtime = min(runtimes)
        max_runtime = max(runtimes)
        ax2.set_ylim(bottom=min_runtime * 0.95, top=max_runtime * 1.05)  # 5% padding
        
        # Add value annotations for each point
        for i, runtime in zip(iterations, runtimes):
            ax2.annotate(f'{runtime:.1f}s', 
                        (i, runtime),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
        
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

    def plot_agent_performance(self, metrics_dir: Path, framework: str) -> None:
        """Create group-wise performance visualization across iterations."""
        if not any(iteration.group_metrics for iteration in self.iterations):
            return
            
        # Get all unique groups across iterations
        all_groups = set()
        for iteration in self.iterations:
            all_groups.update(iteration.group_metrics.keys())
            
        # Setup the plot style
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        fig.suptitle(f'{framework} Agent-wise Performance', fontsize=16, y=1.02)
        
        def plot_metric_values(ax, metric_name, title):
            """Helper to plot metric values with opacity for overlapping lines"""
            # Collect all values for each group
            group_values = {}
            for group in sorted(all_groups):
                values = [
                    iteration.group_metrics.get(group, {}).get(metric_name, None) 
                    for iteration in self.iterations
                ]
                valid_points = [(i, v) for i, v in enumerate(values) if v is not None]
                if valid_points:
                    group_values[group] = valid_points

            # Find overlapping value sequences
            value_sequences = {}
            for group, points in group_values.items():
                value_seq = tuple(v for _, v in points)
                if value_seq in value_sequences:
                    value_sequences[value_seq].append(group)
                else:
                    value_sequences[value_seq] = [group]

            # Plot with appropriate opacity and labels
            for value_seq, groups in value_sequences.items():
                points = group_values[groups[0]]  # Use first group's points (they're the same)
                x_vals, y_vals = zip(*points)
                
                # Use more opacity if values overlap
                alpha = 0.3 if len(groups) > 1 else 1.0
                
                if len(groups) > 1:
                    # For overlapping groups, create a single line with all groups in label
                    label = f"{', '.join(groups)} ({len(groups)} agents)"
                    ax.plot(x_vals, y_vals, marker='o', alpha=alpha, label=label)
                else:
                    # For single group, just plot normally
                    ax.plot(x_vals, y_vals, marker='o', alpha=alpha, label=groups[0])

            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.legend(title='Agents', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Plot each metric
        plot_metric_values(ax1, 'mae', 'Mean Absolute Error')
        plot_metric_values(ax2, 'mape', 'MAPE (%)')
        plot_metric_values(ax3, 'rmse', 'RMSE')
        
        # Add model info
        plt.figtext(0.02, 0.02, 
                   f'Model: {self.model_name}\n'
                   f'Temperature: {self.model_temperature}\n'
                   f'Few-Shot Examples: {self.num_few_shot}', 
                   fontsize=8, ha='left')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{self.model_name}_{framework}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{len(self.iterations)}_agent_performance.png"
        plt.savefig(metrics_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close() 