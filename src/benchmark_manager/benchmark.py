import os
from datetime import datetime, date
from benchmark_data import BenchmarkData
from typing import List, Dict, Optional
import json
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Force non-interactive backend
from pathlib import Path

class Benchmark:
    def __init__(self, benchmark_results: List[BenchmarkData], output_dir: str, logger):
        self.benchmark_results = benchmark_results
        self.output_dir = output_dir
        self.logger = logger
        self.framework_scores = {}

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Ensuring output directory exists: {self.output_dir}")

    def _find_files(self, input_dir: str, date_to_process: Optional[str] = None) -> List[str]:
        """Find JSON files in directory matching date pattern."""
        date_pattern = self._format_date_for_matching(date_to_process)
        self.logger.info(f"Searching for files matching date pattern: {date_pattern}")
        
        pattern = f"*_{date_pattern}_*.json"
        files = list(Path(input_dir).glob(pattern))
        
        if not files:
            self.logger.warning(f"No files found matching date pattern: {date_pattern}")
            return []
            
        self.logger.debug(f"Found {len(files)} matching files")
        return [str(f) for f in files]

    def _format_date_for_matching(self, date_str: Optional[str] = None) -> str:
        """Convert date string to YYYYMMDD format or use today's date."""
        if not date_str:
            return date.today().strftime("%Y%m%d")
            
        try:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            return parsed_date.strftime("%Y%m%d")
        except ValueError:
            self.logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")

    def _generate_timestamp(self) -> str:
        """Generate timestamp for file naming."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def process(self):
        """Process benchmarks and save all results."""
        self.logger.info("Starting benchmark processing")
        
        # Dictionary to store total scores for CSV and leaderboard
        total_scores = {benchmark.name: {} for benchmark in self.benchmark_results}
        
        # Dictionary to store all metrics for visualization
        all_metrics = {
            'total_score': {benchmark.name: {} for benchmark in self.benchmark_results},
            'quality_percentage': {benchmark.name: {} for benchmark in self.benchmark_results},
            'speed_percentage': {benchmark.name: {} for benchmark in self.benchmark_results},
            'resource_percentage': {benchmark.name: {} for benchmark in self.benchmark_results}
        }
        
        # Process each benchmark
        for benchmark in self.benchmark_results:
            self.logger.info(f"Processing benchmark: {benchmark.name}")
            files = self._find_files(benchmark.input_dir, benchmark.date_to_process)
            
            frameworks_in_benchmark = set()
            
            for file in files:
                self.logger.debug(f"Processing file: {file}")
                with open(file, "r") as f:
                    data = json.load(f)
                    
                framework = data["framework"]
                if framework in frameworks_in_benchmark:
                    error_msg = f"Duplicate framework '{framework}' found in benchmark '{benchmark.name}'"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                frameworks_in_benchmark.add(framework)
                
                # Store total score for CSV and leaderboard
                total_scores[benchmark.name][framework] = data["benchmark_score"]["total_score"]
                
                # Store all metrics for visualization
                for metric in all_metrics:
                    all_metrics[metric][benchmark.name][framework] = data["benchmark_score"][metric]
        
        if not total_scores:
            self.logger.error("No benchmark data was processed")
            raise ValueError("No benchmark data was processed")
        
        # Generate outputs
        timestamp = self._generate_timestamp()
        
        # Save CSV
        csv_path = os.path.join(self.output_dir, f"benchmark_score_results_{timestamp}.csv")
        self.logger.info(f"Saving CSV to: {csv_path}")
        pd.DataFrame(total_scores).to_csv(csv_path)
        
        # Save visualization
        viz_path = os.path.join(self.output_dir, f"benchmark_scoring_results_{timestamp}.png")
        self.logger.info(f"Saving visualization to: {viz_path}")
        
        plt.ioff()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Add suptitle with proper spacing
        plt.suptitle('AI Framework Benchmark Comparison Results', 
                    fontsize=16, 
                    fontweight='bold',
                    y=1.02)  # Moved up to prevent overlap
        
        # Transpose DataFrames to group by benchmark and color by framework
        pd.DataFrame(all_metrics['quality_percentage']).T.plot(kind='bar', ax=ax1, title='Quality Score')
        pd.DataFrame(all_metrics['speed_percentage']).T.plot(kind='bar', ax=ax2, title='Speed Score')
        pd.DataFrame(all_metrics['resource_percentage']).T.plot(kind='bar', ax=ax3, title='Resource Score')
        pd.DataFrame(all_metrics['total_score']).T.plot(kind='bar', ax=ax4, title='Total Score')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Framework', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(top=0.85, right=0.85, hspace=0.3)
        
        plt.savefig(viz_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save leaderboard as a clean DataFrame image
        leaderboard_path = os.path.join(self.output_dir, f"benchmark_results_leaderboard_{timestamp}.png")
        self.logger.info(f"Saving leaderboard to: {leaderboard_path}")
        
        # Create styled DataFrame
        df = pd.DataFrame(total_scores)
        styled_df = df.style.format("{:.2f}").set_caption("AI Framework Benchmark Results")
        
        # Save to PNG without displaying
        plt.ioff()
        fig, ax = plt.subplots(figsize=(8, len(df) * 0.5 + 2))  # Added height for title
        ax.axis('off')
        
        # Add title
        plt.title('Framework Benchmark Results', pad=20, fontsize=12, fontweight='bold')
        
        # Create and style table
        table = ax.table(
            cellText=df.round(2).values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 0.95]  # Adjusted to make room for title
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Color header
        for k, cell in table._cells.items():
            if k[0] == 0:  # Header
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', weight='bold')
            elif k[1] == -1:  # Index
                cell.set_text_props(weight='bold')
        
        plt.savefig(leaderboard_path, bbox_inches='tight', dpi=300)
        plt.close()
