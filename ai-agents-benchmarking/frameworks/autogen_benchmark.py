# frameworks/autogen_benchmark.py

from benchmarks.simple_tasks import SIMPLE_TASKS
from benchmarks.complex_tasks import COMPLEX_TASKS
import logging
from autogen import AssistantAgent, UserProxyAgent
from utils.latency_tracker import LatencyTracker
from utils.result_saver import save_results_to_csv, plot_benchmark_results, load_results_from_csv
from utils.token_tracker import TokenTracker
from utils.memory_tracker import MemoryTracker
from utils.accuracy_calculator import calculate_accuracy
from utils.config_loader import get_llm_config

# ==============================
# Global Logging Configuration
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
def get_autogen_framework_name():
    return "AutoGen"

# ==============================
# AutoGen Agent Setup
# ==============================
def get_autogen_agents(model_name=None):
    """Create and return AutoGen assistant and user proxy agents using direct config."""
    llm_config = get_llm_config(model_name)

    # Directly use the loaded configuration
    assistant = AssistantAgent(
        name="assistant",
        llm_config={
            "model": llm_config['model_name'],
            "base_url": llm_config['base_url'],
            "api_key": llm_config['api_key'],
            "temperature": llm_config['temperature'],
            "api_type": "ollama",
        }
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "coding", "use_docker": False},
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
    )

    return assistant, user_proxy

# ==============================
# Execute Tasks with AutoGen
# ==============================
def execute_task_with_autogen(task_func, model_name=None):
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

        # Initialize AutoGen agents
        assistant, user_proxy = get_autogen_agents(model_name)

        # Initiate chat between user proxy and assistant
        user_proxy.initiate_chat(assistant, message=prompt)

        # Extract the last message (the final result)
        result = user_proxy.last_message()["content"]
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
# Benchmark All Tasks
# ==============================
def benchmark_autogen():
    """Runs benchmarking tasks on AutoGen and saves results."""
    print("🚀 Running AutoGen Benchmarks...\n")

    total_tracker = LatencyTracker()
    total_tracker.start()

    all_results = []

    # Benchmark Simple Tasks
    for task_name, task_func in SIMPLE_TASKS.items():
        accuracy, exec_time, total_tokens, peak_memory = execute_task_with_autogen(task_func)
        all_results.append((task_name, accuracy, exec_time, total_tokens, peak_memory))
        print(f"Task: {task_name} | Accuracy: {accuracy}% | Time: {exec_time:.2f}s | Tokens: {total_tokens} | Memory: {peak_memory:.2f} MB\n")

    # Benchmark Complex Tasks
    for task_name, task_func in COMPLEX_TASKS.items():
        accuracy, exec_time, total_tokens, peak_memory = execute_task_with_autogen(task_func)
        all_results.append((task_name, accuracy, exec_time, total_tokens, peak_memory))
        print(f"Task: {task_name} | Accuracy: {accuracy}% | Time: {exec_time:.2f}s | Tokens: {total_tokens} | Memory: {peak_memory:.2f} MB\n")

    total_time = total_tracker.stop()

    # Save and Plot Results
    framework_name = get_autogen_framework_name()
    csv_path = save_results_to_csv(all_results, framework_name)
    df = load_results_from_csv(csv_path)
    plot_benchmark_results(df, title=f"{framework_name} Benchmark",save_path=csv_path.replace(".csv", ""))

    print(f"⏰ Total Benchmark Time: {total_time:.2f}s")

# ==============================
# Main Function
# ==============================
if __name__ == "__main__":
    benchmark_autogen()
