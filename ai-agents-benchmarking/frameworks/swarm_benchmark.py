# frameworks/swarm_benchmark.py

import logging
import time
import os
from utils.latency_tracker import LatencyTracker
from utils.result_saver import save_results_to_csv, plot_benchmark_results, load_results_from_csv
from utils.token_tracker import TokenTracker
from utils.memory_tracker import MemoryTracker
from utils.accuracy_calculator import calculate_accuracy
from utils.config_loader import get_llm_config
from benchmarks.simple_tasks import SIMPLE_TASKS
from benchmarks.complex_tasks import COMPLEX_TASKS
# ==============================
# ✅ FRAMEWORK NAME
# ==============================
def get_swarm_framework_name():
    return "Swarm"
# ==============================
# ✅ GLOBAL LOGGING CONFIGURATION
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress noisy libraries
for noisy_logger in ['LiteLLM', 'httpx', 'opentelemetry.trace', 'autogen.import_utils', 'urllib3']:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
# ==============================
# ✅ IMPORT Swarm FRAMEWORK
# ==============================
try:
    from swarm import Swarm, Agent
    from openai import OpenAI
except ImportError:
    raise ImportError("Swarm is not installed. Please install it using: pip install git+https://github.com/openai/swarm.git")

# ==============================
# ✅ SWARM AGENT SETUP
# ==============================
def create_swarm_agents(model_name=None):
    """Initialize Swarm agents for benchmarking."""
    llm_config = get_llm_config(model_name)
    agents = {}

    # Create agents for simple tasks
    for task_name, task_data in SIMPLE_TASKS.items():
        agents[task_name] = Agent(
            name=f"Agent_{task_name}",
            instructions=task_data["prompt"],
            functions=[task_data]  # Added task_func to the agent's functions
        )

    # Create agents for complex tasks
    for task_name, task_data in COMPLEX_TASKS.items():
        agents[task_name] = Agent(
            name=f"Agent_{task_name}",
            instructions=task_data["prompt"],
            functions=[task_data]  # Added task_func to the agent's functions
        )

    return agents

# ==============================
# ✅ EXECUTE TASKS WITH SWARM
# ==============================
def execute_task_with_swarm(task_func, model_name=None):
    """Executes a given task using Swarm."""
    tracker = LatencyTracker()
    memory_tracker = MemoryTracker()
    token_tracker = TokenTracker(model_name or "gpt-4")

    tracker.start()
    memory_tracker.start()

    try:
        llm_config = get_llm_config(model_name)
        task_data = task_func()
        prompt = task_data["prompt"]
        expected_answer = task_data.get('expected_answer', '')
        task_type = task_data.get('task_type', 'task')
        
        # Count tokens for the prompt
        token_count = token_tracker.count_tokens(prompt)

        client = OpenAI(api_key=llm_config["api_key"])
        exec_time = tracker.start()
        response = client.chat.completions.create(
            model=llm_config["model_name"],
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content
        exec_time = tracker.stop()
        peak_memory = memory_tracker.stop()

        # Calculate tokens for the result
        response_tokens = token_tracker.count_tokens(str(result))
        total_tokens = token_tracker.get_total_tokens()

        # Calculate accuracy
        accuracy = calculate_accuracy(result, expected_answer)

        # Output with resource usage
        print(f"\nLLM Output: {result}")
        print(f"Task: {task_type} | Accuracy: {accuracy}% | Time: {exec_time}s")
        print(f"Total Tokens Used: {total_tokens}")
        print(f"Peak Memory Usage: {peak_memory:.2f} MB\n")

        return accuracy, exec_time, total_tokens, peak_memory

    except Exception as e:
        exec_time = tracker.stop()
        peak_memory = memory_tracker.stop()
        print(f"❌ Error: {e} | Time: {exec_time}s | Memory: {peak_memory:.2f} MB")
        return 0.0, exec_time, 0, peak_memory

# ==============================
# ✅ BENCHMARK ALL TASKS
# ==============================
def benchmark_swarm():
    """Runs benchmarking tasks on Swarm."""
    print("🚀 Running Swarm Benchmarks...\n")
    
    agents = create_swarm_agents()
    total_tracker = LatencyTracker()
    total_tracker.start()

    all_results = []
        
    # Benchmark Simple Tasks
    for task_name, task_func in SIMPLE_TASKS.items():
        accuracy, exec_time, total_tokens, peak_memory = execute_task_with_swarm(task_func)
        all_results.append((task_name, accuracy, exec_time, total_tokens, peak_memory))
        print(f"Task: {task_name} | Accuracy: {accuracy}% | Time: {exec_time:.2f}s | Tokens: {total_tokens} | Memory: {peak_memory:.2f} MB\n")

    # Benchmark Complex Tasks
    for task_name, task_func in COMPLEX_TASKS.items():
        accuracy, exec_time, total_tokens, peak_memory = execute_task_with_swarm(task_func)
        all_results.append((task_name, accuracy, exec_time, total_tokens, peak_memory))
        print(f"Task: {task_name} | Accuracy: {accuracy}% | Time: {exec_time:.2f}s | Tokens: {total_tokens} | Memory: {peak_memory:.2f} MB\n")

    total_time = total_tracker.stop()

    # Save and Plot Results
    framework_name = get_swarm_framework_name()
    csv_path = save_results_to_csv(all_results, framework_name)
    df = load_results_from_csv(csv_path)
    plot_benchmark_results(df, title=f"{framework_name} Benchmark",save_path=csv_path.replace(".csv", ""))

    print(f"⏰ Total Benchmark Time: {total_time:.2f}s")

# ==============================
# ✅ MAIN FUNCTION
# ==============================
if __name__ == "__main__":
    benchmark_swarm()
