# Script to generate graphs and leaderboard
# scripts/run_benchmark.py

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from frameworks.crewai_benchmark import benchmark_crewai
from frameworks.langchain_benchmark import benchmark_langchain
from frameworks.langgraph_benchmark import benchmark_langgraph
from frameworks.swarm_benchmark import benchmark_swarm
from frameworks.autogen_benchmark import benchmark_autogen

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_all_benchmarks():
    """Runs all AI agent framework benchmarks with error handling."""
    benchmarks = {
        "CrewAI": benchmark_crewai,
        "LangChain": benchmark_langchain,
        "LangGraph": benchmark_langgraph,
        "Swarm": benchmark_swarm,
        "AutoGen": benchmark_autogen
    }
    
    for name, benchmark_func in benchmarks.items():
        try:
            logging.info(f"Starting benchmark: {name}")
            benchmark_func()
            logging.info(f"Completed benchmark: {name}\n")
        except Exception as e:
            logging.error(f"Error in {name} benchmark: {e}")

def visualize_results(result_file="results/public_leaderboard.csv", show_plots=False):
    """Generates visualization from benchmark results."""
    logging.info(f"Visualizing results from {result_file}")
    try:
        df = pd.read_csv(result_file)
        
        if df.empty:
            logging.warning("No results found in leaderboard.")
            return
        
        # Plot inference time comparison
        plt.figure(figsize=(10, 5))
        df.groupby("Framework")["Average Time"].mean().plot(kind="bar", color="skyblue", edgecolor="black")
        plt.xlabel("Framework")
        plt.ylabel("Avg Inference Time (seconds)")
        plt.title("AI Agent Inference Time Comparison")
        plt.xticks(rotation=45)
        plt.grid(axis="y")
        plt.savefig(f"{result_file.replace('.csv', '')}_time.png")
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 5))
        df.groupby("Framework")["Average Accuracy"].mean().plot(kind="bar", color="lightcoral", edgecolor="black")
        plt.xlabel("Framework")
        plt.ylabel("Avg Accuracy (%)")
        plt.title("AI Agent Accuracy Comparison")
        plt.xticks(rotation=45)
        plt.grid(axis="y")
        plt.savefig(f"{result_file.replace('.csv', '')}_accuracy.png")
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Plot tokens comparison
        plt.figure(figsize=(10, 5))
        df.groupby("Framework")["Average Tokens"].mean().plot(kind="bar", color="lightgreen", edgecolor="black")
        plt.xlabel("Framework")
        plt.ylabel("Avg Tokens")
        plt.title("AI Agent Tokens Comparison")
        plt.xticks(rotation=45)
        plt.grid(axis="y")
        plt.savefig(f"{result_file.replace('.csv', '')}_tokens.png")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Plot memory comparison
        plt.figure(figsize=(10, 5))
        df.groupby("Framework")["Average Memory"].mean().plot(kind="bar", color="orange", edgecolor="black")
        plt.xlabel("Framework")
        plt.ylabel("Avg Memory (MB)")
        plt.title("AI Agent Memory Comparison")
        plt.xticks(rotation=45)
        plt.grid(axis="y")
        plt.savefig(f"{result_file.replace('.csv', '')}_memory.png")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        logging.error(f"Error visualizing results: {e}")

def main():
    """Main function to execute benchmark based on command-line arguments."""
    parser = argparse.ArgumentParser(description="Run AI agent benchmarking")
    parser.add_argument("--framework", type=str, choices=["CrewAI", "LangChain", "LangGraph", "Swarm", "AutoGen", "all"], default="all", help="Specify framework to benchmark or run all")
    parser.add_argument("--visualize", action="store_true", help="Visualize benchmark results")
    parser.add_argument("--show_plots", action="store_true", help="Show plots instead of saving them") # added this line
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_results(show_plots=args.show_plots) # updated this line
    elif args.framework == "all":
        run_all_benchmarks()
    else:
        try:
            logging.info(f"Running benchmark for {args.framework}")
            globals()[f"benchmark_{args.framework.lower()}"]()
        except KeyError:
            logging.error("Invalid framework specified.")
        except Exception as e:
            logging.error(f"Error running benchmark: {e}")

if __name__ == "__main__":
    main()
