import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define correct relative path (go one level up)
base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
results_folder = os.path.join(base_folder, "results")  # Folder to save images
os.makedirs(results_folder, exist_ok=True)  # Ensure the results folder exists

# Define all benchmark folders and filenames
benchmarks = {
    "CrewAI": os.path.join(base_folder, "crewai", "crewai_benchmark_results.csv"),
    "LangGraph": os.path.join(base_folder, "langgraph", "langgraph_benchmark_results.csv"),
    "Autogen": os.path.join(base_folder, "autogen", "autogen_benchmark_results.csv"),
}

# Dictionary to store data
latency_data = {}
token_data = {}
response_rating_data = {}
peak_memory_data = {}
memory_delta_data = {}


# Read and store data for each benchmark
for name, file_path in benchmarks.items():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Read CSV, ignoring the last row
    df = pd.read_csv(file_path)
    df = df[:-1]  # Exclude last row

    # Extract keyword, latency, and price columns
    latency_data[name] = pd.to_numeric(df['Latency (seconds)'], errors='coerce')
    token_data[name] = pd.to_numeric(df['Total Tokens'], errors='coerce')
    response_rating_data[name] = pd.to_numeric(df['Response Ratings'], errors='coerce')
    peak_memory_data[name] = pd.to_numeric(df['Peak Memory'], errors='coerce')
    memory_delta_data[name] = pd.to_numeric(df['Memory Delta'], errors='coerce')

Keyword = df['Keyword']

# Set bar width
bar_width = 0.25
x = np.arange(len(Keyword))  # X-axis positions

# Colors for each benchmark
colors = {"CrewAI": "skyblue", "LangGraph": "lightgreen", "Autogen": "lightcoral"}

# **Plot Latency Comparison**
plt.figure(figsize=(10, 6))
for i, (name, latencies) in enumerate(latency_data.items()):
    plt.bar(x + i * bar_width, latencies, width=bar_width, label=name, color=colors[name])

plt.xticks(x + bar_width, Keyword, rotation=45)
plt.xlabel("Keyword")
plt.ylabel("Latency (seconds)")
plt.title("Latency Comparison Across Benchmarks")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "latency_comparison.png"))
plt.close()

# **Plot Token Comparison**
plt.figure(figsize=(10, 6))
for i, (name, tokens) in enumerate(token_data.items()):
    plt.bar(x + i * bar_width, tokens, width=bar_width, label=name, color=colors[name])

plt.xticks(x + bar_width, Keyword, rotation=45)
plt.xlabel("Keyword")
plt.ylabel("Total Tokens")
plt.title("Token Comparison Across Benchmarks")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "token_comparison.png"))
plt.close()

# **Plot Response Rating Comparison**
plt.figure(figsize=(10, 6))
for i, (name, ratings) in enumerate(response_rating_data.items()):
    plt.bar(x + i * bar_width, ratings, width=bar_width, label=name, color=colors[name])

plt.xticks(x + bar_width, Keyword, rotation=45)
plt.xlabel("Keyword")
plt.ylabel("Response Rating")
plt.title("Response Rating Comparison Across Benchmarks")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "response_rating_comparison.png"))
plt.close()

# **Plot Peak Memory Comparison**
plt.figure(figsize=(10, 6))
for i, (name, memory) in enumerate(peak_memory_data.items()):
    plt.bar(x + i * bar_width, memory, width=bar_width, label=name, color=colors[name])

plt.xticks(x + bar_width, Keyword, rotation=45)
plt.xlabel("Keyword")
plt.ylabel("Peak Memory (MB)")
plt.title("Peak Memory Usage Across Benchmarks")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "peak_memory_comparison.png"))
plt.close()

# **Plot Memory Delta Comparison**
plt.figure(figsize=(10, 6))
for i, (name, delta) in enumerate(memory_delta_data.items()):
    plt.bar(x + i * bar_width, delta, width=bar_width, label=name, color=colors[name])

plt.xticks(x + bar_width, Keyword, rotation=45)
plt.xlabel("Keyword")
plt.ylabel("Memory Delta (MB)")
plt.title("Memory Delta Comparison Across Benchmarks")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "memory_delta_comparison.png"))
plt.close()


print("All comparison graphs and summary saved in 'results' folder.")
