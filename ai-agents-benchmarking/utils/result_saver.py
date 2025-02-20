# utils/result_saver.py

import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

def save_results_to_csv(results, filename=None, directory="results"):
    """
    Save benchmarking results to a timestamped CSV file.

    Args:
        results (list): List of tuples (Task, Accuracy, Execution Time, Tokens, Memory Usage).
        filename (str): Optional filename. If None, generates timestamped filename.
        directory (str): Directory to save results.
    """
    os.makedirs(directory, exist_ok=True)
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.csv"

    file_path = os.path.join(directory, filename)
    
    # ✅ Updated columns to match data structure
    df = pd.DataFrame(results, columns=["Task", "Accuracy", "Execution Time", "Tokens", "Memory Usage"])
    
    # Save using UTF-8 encoding to avoid encoding errors
    df.to_csv(file_path, index=False, encoding='utf-8')

    print(f"✅ Results saved to {file_path}")
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

        print(f"📊 Loaded results from {filepath}")
        return df
    else:
        print(f"❌ File not found: {filepath}")
        return None


def plot_benchmark_results(df, title="Benchmark Results", save_path=None):
    """
    Plot Accuracy, Execution Time, Tokens, and Memory Usage for each task.

    Args:
        df (pd.DataFrame): DataFrame with benchmark results.
        title (str): Title of the plot.
        save_path (str): If provided, saves the plot as PNG.
    """
    tasks = df["Task"]
    accuracy = df["Accuracy"]
    exec_time = df["Execution Time"]
    tokens = df["Tokens"]
    memory_usage = df["Memory Usage"]

    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    plt.bar(tasks, accuracy, color='skyblue')
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Tasks")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_accuracy.png")
    plt.show()

    # Plot Execution Time
    plt.figure(figsize=(12, 5))
    plt.bar(tasks, exec_time, color='salmon')
    plt.title(f"{title} - Execution Time")
    plt.xlabel("Tasks")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_time.png")
    plt.show()

    # Plot Token Usage
    plt.figure(figsize=(12, 5))
    plt.bar(tasks, tokens, color='lightgreen')
    plt.title(f"{title} - Token Usage")
    plt.xlabel("Tasks")
    plt.ylabel("Tokens")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_tokens.png")
    plt.show()

    # Plot Memory Usage
    plt.figure(figsize=(12, 5))
    plt.bar(tasks, memory_usage, color='purple')
    plt.title(f"{title} - Memory Usage")
    plt.xlabel("Tasks")
    plt.ylabel("Memory (MB)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_memory.png")
    plt.show()


def compare_benchmarks(file_paths):
    """
    Compare multiple benchmark runs.

    Args:
        file_paths (list): List of CSV file paths to compare.
    """
    combined_df = pd.DataFrame()

    for file in file_paths:
        df = load_results_from_csv(file)
        if df is not None:
            df["Run"] = os.path.basename(file).replace(".csv", "")
            combined_df = pd.concat([combined_df, df])

    if combined_df.empty:
        print("❌ No data to compare.")
        return

    # Plot Accuracy Comparison
    plt.figure(figsize=(14, 6))
    for run in combined_df["Run"].unique():
        run_df = combined_df[combined_df["Run"] == run]
        plt.plot(run_df["Task"], run_df["Accuracy"], marker='o', label=run)

    plt.title("Benchmark Accuracy Comparison")
    plt.xlabel("Tasks")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Execution Time Comparison
    plt.figure(figsize=(14, 6))
    for run in combined_df["Run"].unique():
        run_df = combined_df[combined_df["Run"] == run]
        plt.plot(run_df["Task"], run_df["Execution Time"], marker='o', label=run)

    plt.title("Benchmark Execution Time Comparison")
    plt.xlabel("Tasks")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
