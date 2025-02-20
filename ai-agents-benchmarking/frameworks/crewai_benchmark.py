from benchmarks.simple_tasks import SIMPLE_TASKS
from benchmarks.complex_tasks import COMPLEX_TASKS
from typing import Dict, Any
import logging
import yaml
import time
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from utils.latency_tracker import LatencyTracker
from utils.result_saver import save_results_to_csv, plot_benchmark_results, compare_benchmarks, load_results_from_csv
from utils.token_tracker import TokenTracker
from utils.memory_tracker import MemoryTracker
from utils.accuracy_calculator import calculate_accuracy  # ✅ NEW IMPORT
import os

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
# ✅ IMPORT CrewAI FRAMEWORK
# ==============================
try:
    from crewai import Crew, Agent, Task, LLM, Process
except ImportError:
    raise ImportError("CrewAI is not installed. Please install it using: pip install crewai")

# ==============================
# ✅ LOAD CONFIGURATION
# ==============================
def get_llm_config(model_name=None):
    """Load LLM configuration from settings.yaml using a unified endpoint."""
    try:
        config_path = os.path.join("config", "settings.yaml")
        with open(config_path, 'r') as f:
            settings = yaml.safe_load(f)

        llm_settings = settings.get('llm_execution', {})

        config = {
            'model_name': model_name or llm_settings.get('model_name', 'default_model'),
            'base_url': llm_settings.get('api_base', ''),
            'api_key': llm_settings.get('api_key', ''),
            'temperature': 0.7  # Default temperature
        }

        logger.debug(f"Loaded LLM Config: {config}")
        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"⚠️ settings.yaml not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"⚠️ Error parsing YAML file: {e}")
    except Exception as e:
        raise Exception(f"⚠️ Unexpected error loading LLM config: {e}")

# ==============================
# ✅ CREWAI AGENT SETUP
# ==============================
def get_crewai_agent(model_name=None):
    """Create and return a CrewAI agent using a unified endpoint."""
    llm_config = get_llm_config(model_name)

    llm = LLM(
        model=llm_config['model_name'],
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
    token_tracker = TokenTracker(model_name or "gpt-4")

    tracker.start()
    memory_tracker.start()

    try:
        task_data = task_func()
        prompt = task_data["prompt"]
        expected_answer = task_data.get('expected_answer', '')
        task_type = task_data.get('task_type', 'task')

        # Count tokens for the prompt
        token_count = token_tracker.count_tokens(prompt)

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
        )

        # Execute the task
        result = crew.kickoff()
        exec_time = tracker.stop()
        peak_memory = memory_tracker.stop()

        # Calculate tokens for the result
        response_tokens = token_tracker.count_tokens(str(result))
        total_tokens = token_tracker.get_total_tokens()

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
    csv_path = save_results_to_csv(all_results)
    df = load_results_from_csv(csv_path)
    plot_benchmark_results(df, title="CrewAI Benchmark", save_path=csv_path.replace(".csv", ""))

    print(f"⏰ Total Benchmark Time: {total_time:.2f}s")

# ==============================
# ✅ MAIN FUNCTION
# ==============================
if __name__ == "__main__":
    benchmark_crewai()
