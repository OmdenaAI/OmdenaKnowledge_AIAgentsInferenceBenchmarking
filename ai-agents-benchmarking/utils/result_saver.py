# utils/result_saver.py

import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def save_results_to_csv(results, framework_name, directory="data/results"):
    """
    Save benchmarking results to a timestamped CSV file.

    Args:
        results (list): List of tuples (Task, Accuracy, Execution Time, Tokens, Memory Usage).
        framework_name (str): The framework name, for example `CrewAI`
        directory (str): Directory to save results.
    """
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{framework_name}_{timestamp}.csv"

    file_path = os.path.join(directory, filename)

    # ✅ Updated columns to match data structure
    df = pd.DataFrame(results, columns=["Task", "Accuracy", "Time", "Tokens", "Memory"]) # updated this line

    # Add framework information to the data
    df['Framework'] = framework_name

    # Save using UTF-8 encoding to avoid encoding errors
    df.to_csv(file_path, index=False, encoding='utf-8')

    logger.info(f"✅ Results saved to {file_path}")
    return file_path


def load_results_from_csv(filepath):
    """
    Load benchmark results from CSV with encoding fallback.

    Args:
        filepath (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if os.path.exists(filepath):
        try:
            # Try reading with UTF-8
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to ISO-8859-1 if UTF-8 fails
            df = pd.read_csv(filepath, encoding='ISO-8859-1', errors='replace')

        logger.info(f"📊 Loaded results from {filepath}")
        return df
    else:
        logger.warning(f"❌ File not found: {filepath}")
        return None


def plot_benchmark_results(df, title, save_path=None):
    """
    Plot Accuracy, Execution Time, Tokens, and Memory Usage for each task.

    Args:
        df (pd.DataFrame): DataFrame with benchmark results.
        title (str): Title of the plot.
        save_path (str): If provided, saves the plot as PNG.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Skipping plot generation.")
        return

    tasks = df["Task"].astype(str)
    accuracy = df["Accuracy"]
    exec_time = df["Time"] # updated this line
    tokens = df["Tokens"]
    memory_usage = df["Memory"] # updated this line

    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    plt.bar(tasks, accuracy, color='skyblue')
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Tasks")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_accuracy.png")
    plt.close()

    # Plot Execution Time
    plt.figure(figsize=(12, 5))
    plt.bar(tasks, exec_time, color='salmon')
    plt.title(f"{title} - Execution Time")
    plt.xlabel("Tasks")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_time.png")
    plt.close()

    # Plot Token Usage
    plt.figure(figsize=(12, 5))
    plt.bar(tasks, tokens, color='lightgreen')
    plt.title(f"{title} - Token Usage")
    plt.xlabel("Tasks")
    plt.ylabel("Tokens")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_tokens.png")
    plt.close()

    # Plot Memory Usage
    plt.figure(figsize=(12, 5))
    plt.bar(tasks, memory_usage, color='purple')
    plt.title(f"{title} - Memory Usage")
    plt.xlabel("Tasks")
    plt.ylabel("Memory (MB)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_memory.png")
    plt.close()

    logger.info(f"📊 Plots generated")
