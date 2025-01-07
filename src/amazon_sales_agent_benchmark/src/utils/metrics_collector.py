from typing import Dict, List
import logging
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from bert_score import score
from datetime import datetime

logger = logging.getLogger("amazon_query_processor")

class MetricsCollector:
    """Collects and manages benchmark metrics."""

    def __init__(self, output_dir: str):
        """Initialize metrics collector."""
        try:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.metrics = {}
            self.current_metrics = {}
            logger.info(f"Metrics collector initialized with output dir: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {str(e)}")
            raise

    def start_iteration(self, agent_name: str, iteration: int):
        """Start timing a new iteration."""
        if agent_name not in self.metrics:
            self.metrics[agent_name] = []
        
        self.current_metrics = {
            "iteration": iteration,
            "start_time": time.time(),
            "api_calls": 0,
            "retries": 0,
            "bert_scores": [],
            "api_latencies": []  # Add list to store individual API latencies
        }

    def end_iteration(self):
        """End timing current iteration."""
        self.current_metrics["latency"] = time.time() - self.current_metrics["start_time"]

    def add_bert_score(self, prediction: str, reference: str):
        """Calculate and add BERT score."""
        try:
            P, R, F1 = score([prediction], [reference], lang="en")
            self.current_metrics["bert_scores"].append(F1.item())
        except Exception as e:
            logger.error(f"Failed to calculate BERT score: {str(e)}")
            self.current_metrics["bert_scores"].append(0.0)

    def increment_api_calls(self):
        """Increment API call counter."""
        self.current_metrics["api_calls"] += 1

    def increment_retries(self):
        """Increment retry counter."""
        self.current_metrics["retries"] += 1

    def add_api_latency(self, latency: float):
        """Add individual API call latency."""
        self.current_metrics["api_latencies"].append(latency)

    def save_iteration(self, agent_name: str):
        """Save metrics for current iteration."""
        self.current_metrics["avg_bert_score"] = (
            sum(self.current_metrics["bert_scores"]) / len(self.current_metrics["bert_scores"]) 
            if self.current_metrics["bert_scores"] else 0
        )
        self.current_metrics["avg_api_latency"] = (
            sum(self.current_metrics["api_latencies"]) / len(self.current_metrics["api_latencies"])
            if self.current_metrics["api_latencies"] else 0
        )
        self.metrics[agent_name].append(self.current_metrics)

    def save_metrics(self):
        """Save all metrics to file with timestamp and iteration count."""
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get total iterations from first agent's metrics
            first_agent = next(iter(self.metrics))
            total_iterations = len(self.metrics[first_agent])
            
            # Create filename with timestamp and iterations
            filename = f"metrics_{timestamp}_iter{total_iterations}.json"
            metrics_file = self.output_dir / filename
            
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
                
            logger.info(f"Metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
            raise

    def generate_plots(self):
        """Generate visualization plots."""
        try:
            plt.style.use('ggplot')
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Framework Comparison Metrics', fontsize=16)

            # Define colors for each framework
            framework_colors = {
                'langchain': '#2ecc71',    # Green
                'langgraph': '#3498db',    # Blue
                'autogen': '#e74c3c',      # Red
                'crewai': '#f1c40f'        # Yellow
            }

            # Calculate averages for each framework
            framework_averages = {}
            for framework, iterations in self.metrics.items():
                framework_averages[framework] = {
                    "bert_score": sum(iter["avg_bert_score"] for iter in iterations) / len(iterations),
                    "latency": sum(iter["latency"] for iter in iterations) / len(iterations),
                    "api_calls": sum(iter["api_calls"] for iter in iterations) / len(iterations),
                    "api_latency": sum(iter["avg_api_latency"] for iter in iterations) / len(iterations)
                }

            # Create plots
            frameworks = list(framework_averages.keys())
            metrics = [
                ("bert_score", "Average BERT Score", 0, 0),
                ("latency", "Total Iteration Time (s)", 0, 1),
                ("api_calls", "Average API Calls", 1, 0),
                ("api_latency", "Average API Latency (s)", 1, 1)
            ]

            for metric, title, row, col in metrics:
                values = [framework_averages[f][metric] for f in frameworks]
                colors = [framework_colors.get(f, '#95a5a6') for f in frameworks]
                
                bars = axes[row][col].bar(frameworks, values, color=colors)
                
                # Add value labels on top of bars with conditional formatting
                for bar in bars:
                    height = bar.get_height()
                    # Use 3 decimal places for API latency, 2 for others
                    format_str = '{:.2f}' if metric == 'api_calls' else '{:.3f}'
                    axes[row][col].text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        format_str.format(height),
                        ha='center', 
                        va='bottom'
                    )
                
                axes[row][col].set_title(title)
                axes[row][col].set_xticklabels(frameworks, rotation=45)

            plt.tight_layout()
            
            # Create timestamp and filename matching save_metrics format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            first_agent = next(iter(self.metrics))
            total_iterations = len(self.metrics[first_agent])
            filename = f"metrics_{timestamp}_iter{total_iterations}.png"
            
            plot_file = self.output_dir / filename
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Plots saved to {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {str(e)}")
            raise 