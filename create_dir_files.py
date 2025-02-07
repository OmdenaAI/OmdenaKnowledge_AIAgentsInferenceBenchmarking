import os

def create_project_structure(base_dir):
    directories = [
        "benchmarks",
        "frameworks",
        "config",
        "data/test_cases",
        "data/results",
        "docs",
        "scripts",
        "tests",
        "results"
    ]
    
    files = {
        "benchmarks/simple_tasks.py": "# Benchmarking simple AI agent tasks\n",
        "benchmarks/complex_tasks.py": "# Benchmarking complex AI agent tasks\n",
        "frameworks/crewai_benchmark.py": "# Benchmark CrewAI framework\n",
        "frameworks/langchain_benchmark.py": "# Benchmark LangChain framework\n",
        "frameworks/langgraph_benchmark.py": "# Benchmark LangGraph framework\n",
        "frameworks/swarm_benchmark.py": "# Benchmark Swarm framework\n",
        "frameworks/autogen_benchmark.py": "# Benchmark AutoGen framework\n",
        "config/settings.yaml": "# Configuration settings for benchmarking\n",
        "scripts/run_benchmark.py": "# Script to execute benchmarks\n",
        "scripts/visualize_results.py": "# Script to generate graphs and leaderboard\n",
        "tests/test_benchmarks.py": "# Unit tests for benchmarking\n",
        "results/public_leaderboard.csv": "framework, task, inference_time, accuracy\n",
        "requirements.txt": "# List of dependencies\n",
        "README.md": "# AI Agents Benchmarking Project\n",
        ".gitignore": "# Ignore unnecessary files\n",
        "main.py": "# Entry point for running the benchmarking suite\n"
    }
    
    # Create directories
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    
    # Create files with initial content
    for file, content in files.items():
        file_path = os.path.join(base_dir, file)
        with open(file_path, "w") as f:
            f.write(content)
    
    print(f"Project structure created at: {base_dir}")

if __name__ == "__main__":
    base_directory = "ai-agents-benchmarking"  # Change if needed
    create_project_structure(base_directory)
