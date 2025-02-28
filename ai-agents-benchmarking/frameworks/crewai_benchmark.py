from benchmarks.simple_tasks import SIMPLE_TASKS
from benchmarks.complex_tasks import COMPLEX_TASKS
from typing import Dict, Any
import logging
import time
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from utils.latency_tracker import LatencyTracker
from utils.result_saver import save_results_to_csv, plot_benchmark_results, load_results_from_csv
from utils.token_tracker import TokenTracker
from utils.memory_tracker import MemoryTracker
from utils.accuracy_calculator import calculate_accuracy  # ✅ Accuracy in utils
from utils.config_loader import get_llm_config  # ✅ Config loader in utils
import os,re
from datetime import datetime


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
# ✅ FRAMEWORK NAME
# ==============================
def get_crewai_framework_name():
    return "CrewAI"

# ==============================
# ✅ IMPORT CrewAI FRAMEWORK
# ==============================
try:
    from crewai import Crew, Agent, Task, LLM, Process
except ImportError:
    raise ImportError("CrewAI is not installed. Please install it using: pip install crewai")

# ==============================
# ✅ CREWAI AGENT SETUP
# ==============================
def get_crewai_agent(model_name=None):
    """Create and return a CrewAI agent using a unified endpoint."""
    llm_config = get_llm_config(model_name)

    llm = LLM(
        model=f"ollama/{llm_config['model_name']}", # Always add "ollama/" prefix for crewai local call.
        base_url=llm_config['base_url'],
        api_key=llm_config['api_key'],
        temperature=llm_config['temperature']
    )

    return Agent(
        name="AI Benchmark Agent",
        role="AI Researcher",
        goal="Execute AI tasks with high accuracy",
        backstory="AI agent specialized in executing various AI tasks.",
        llm=llm,
        allow_delegation=False
    )

# ==============================
# ✅ EXECUTE TASKS WITH CREWAI
# ==============================
def execute_task_with_crewai(task_func, model_name=None):
    tracker = LatencyTracker()
    memory_tracker = MemoryTracker()
    tracker.start()
    memory_tracker.start()

    try:
        task_data = task_func()
        prompt = task_data["prompt"]
        expected_answer = task_data.get('expected_answer', '')
        task_type = task_data.get('task_type', 'task')

        # Initialize CrewAI agent
        agent = get_crewai_agent(model_name)

        # Create the task
        task = Task(
            name=f"CrewAI {task_type} Task",
            description=prompt,
            expected_output="Provide the correct answer directly.",
            agent=agent
        )

        # Initialize CrewAI
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential
            # process=Process.hierarchical
        )

        # Execute the task
        result = crew.kickoff()
        exec_time = tracker.stop()
        peak_memory = memory_tracker.stop()

        # Access token usage metrics from crew object
        total_tokens = 0
        try:
            total_tokens = crew.usage_metrics.total_tokens
        except Exception as e:
          logger.error(f"Error getting token usage: {e}")
          
        # ✅ Use refactored accuracy function
        accuracy = calculate_accuracy(result, expected_answer)

        # ✅ Output with resource usage
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
def benchmark_crewai():
    """Runs benchmarking tasks on CrewAI and saves results."""
    print("🚀 Running CrewAI Benchmarks...\n")

    total_tracker = LatencyTracker()
    total_tracker.start()

    all_results = []

    # Benchmark Simple Tasks
    for task_name, task_func in SIMPLE_TASKS.items():
        accuracy, exec_time, total_tokens, peak_memory = execute_task_with_crewai(task_func)
        all_results.append((task_name, accuracy, exec_time, total_tokens, peak_memory))
        print(f"Task: {task_name} | Accuracy: {accuracy}% | Time: {exec_time:.2f}s | Tokens: {total_tokens} | Memory: {peak_memory:.2f} MB\n")

    # Benchmark Complex Tasks
    for task_name, task_func in COMPLEX_TASKS.items():
        accuracy, exec_time, total_tokens, peak_memory = execute_task_with_crewai(task_func)
        all_results.append((task_name, accuracy, exec_time, total_tokens, peak_memory))
        print(f"Task: {task_name} | Accuracy: {accuracy}% | Time: {exec_time:.2f}s | Tokens: {total_tokens} | Memory: {peak_memory:.2f} MB\n")

    total_time = total_tracker.stop()

    # Save and Plot Results
    framework_name = get_crewai_framework_name()
    csv_path = save_results_to_csv(all_results, framework_name)
    df = load_results_from_csv(csv_path)
    plot_benchmark_results(df, title=f"{framework_name} Benchmark", save_path=csv_path)

    print(f"⏰ Total Benchmark Time: {total_time:.2f}s")

# ==============================
# ✅ MAIN FUNCTION
# ==============================
if __name__ == "__main__":
    benchmark_crewai()
