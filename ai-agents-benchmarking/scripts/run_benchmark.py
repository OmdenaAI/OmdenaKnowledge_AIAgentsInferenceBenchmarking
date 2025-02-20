# Script to execute benchmarks
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

# Mapping of framework names to their benchmark functions
logger = logging.getLogger("BenchmarkRunner")

# Mapping of framework names to their benchmark functions
BENCHMARKS = {
    "CrewAI": benchmark_crewai,
    "LangChain": benchmark_langchain,
    "LangGraph": benchmark_langgraph,
    "Swarm": benchmark_swarm,
    "AutoGen": benchmark_autogen
}

def run_all_benchmarks():
    """Runs all AI agent framework benchmarks based on config settings."""
    frameworks = config.get("frameworks", [])
    
    for framework in frameworks:
        name, enabled = framework["name"], framework["enabled"]
        if enabled and name in BENCHMARKS:
            logger.info(f"Running {name} Benchmark...")
            try:
                BENCHMARKS[name]()
                logger.info(f"Completed {name} Benchmark.")
            except Exception as e:
                logger.error(f"Error running {name} Benchmark: {e}")
        else:
            logger.info(f"Skipping {name} Benchmark (disabled in config).")


def visualize_results(result_file="results/public_leaderboard.csv"):
    """Generates visualization from benchmark results."""
    try:
        df = pd.read_csv(result_file)
        if df.empty:
            logging.warning("No results found in leaderboard.")
            return
        
        # Plot inference time comparison
        plt.figure(figsize=(10, 5))
        df.groupby("framework")["inference_time"].mean().plot(kind="bar", color="skyblue", edgecolor="black")
        plt.xlabel("Framework")
        plt.ylabel("Avg Inference Time (seconds)")
        plt.title("AI Agent Inference Time Comparison")
        plt.xticks(rotation=45)
        plt.grid(axis="y")
        plt.show()
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 5))
        df.groupby("framework")["accuracy"].mean().plot(kind="bar", color="lightcoral", edgecolor="black")
        plt.xlabel("Framework")
        plt.ylabel("Avg Accuracy (%)")
        plt.title("AI Agent Accuracy Comparison")
        plt.xticks(rotation=45)
        plt.grid(axis="y")
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing results: {e}")

def main():
    """Main function to execute benchmark based on command-line arguments."""
    parser = argparse.ArgumentParser(description="Run AI agent benchmarking")
    parser.add_argument("--framework", type=str, choices=["CrewAI", "LangChain", "LangGraph", "Swarm", "AutoGen", "all"], default="all", help="Specify framework to benchmark or run all")
    parser.add_argument("--visualize", action="store_true", help="Visualize benchmark results")
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_results()
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
