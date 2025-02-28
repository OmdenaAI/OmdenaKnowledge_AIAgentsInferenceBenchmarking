import pandas as pd
import os
import logging
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

def consolidate_framework_results(results_dir="data/results"):
    """
    Consolidate results for each framework into separate CSV files.

    Args:
        results_dir (str): Directory containing individual framework results CSVs.
    """
    framework_data = defaultdict(list)

    # Collect data for each framework
    for filename in os.listdir(results_dir):
        if filename.startswith("benchmark_results_") and filename.endswith(".csv"):
            parts = filename.split("_")
            if len(parts) >= 3:
                framework_name = parts[2].replace(".csv", "")
                filepath = os.path.join(results_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    df["Framework"] = framework_name # ✅ Add the framework column.
                    framework_data[framework_name].append(df)
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")

    # Consolidate and save each framework's data
    for framework_name, dataframes in framework_data.items():
        if dataframes:
            consolidated_df = pd.concat(dataframes, ignore_index=True)
            consolidated_filename = os.path.join(results_dir, f"benchmark_results_{framework_name}.csv")
            consolidated_df.to_csv(consolidated_filename, index=False)
            logger.info(f"Consolidated {framework_name} results saved to {consolidated_filename}")
        else:
            logger.warning(f"No results found for {framework_name}.")

def consolidate_and_generate_leaderboard(results_dir="data/results", leaderboard_file="results/public_leaderboard.csv"):
    """Generate the public leaderboard after consolidating all frameworks."""
    all_framework_results = []
    for filename in os.listdir(results_dir):
        if filename.startswith("benchmark_results_") and filename.endswith(".csv") and len(filename.split("_"))==3:
            filepath = os.path.join(results_dir, filename)
            try:
              df = pd.read_csv(filepath)
              all_framework_results.append(df)
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

    if all_framework_results:
        consolidated_all_framework = pd.concat(all_framework_results, ignore_index=True)
        leaderboard_df = create_leaderboard(consolidated_all_framework)
        os.makedirs(os.path.dirname(leaderboard_file), exist_ok=True)
        leaderboard_df.to_csv(leaderboard_file, index=False)
        logger.info(f"Leaderboard saved to {leaderboard_file}")
    else:
        logger.warning(f"No results found for leaderboard generation.")

#Helper
def create_leaderboard(df):
    """Create a leaderboard DataFrame from benchmark results."""
    # Group by framework and calculate averages
    try:
        leaderboard = df.groupby('Framework').agg(
            {'Accuracy': 'mean', 'Execution Time': 'mean'}
        ).reset_index().sort_values(by="Accuracy", ascending=False)
        
        #Rename columns
        leaderboard.rename(columns={'Accuracy': 'Average Accuracy', 'Execution Time': 'Average Time'}, inplace=True)
        return leaderboard

    except Exception as e:
        logger.error(f"Error creating leaderboard: {e}")
        return pd.DataFrame(columns=["Framework", "Average Accuracy", "Average Time"])
