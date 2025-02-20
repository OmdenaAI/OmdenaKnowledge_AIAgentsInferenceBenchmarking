# frameworks/swarm_benchmark.py

from benchmarks.simple_tasks import SIMPLE_TASKS
from benchmarks.complex_tasks import COMPLEX_TASKS
from openai import OpenAI
import yaml
import os

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Import Swarm framework (Ensure it's installed via pip)
try:
    from swarm import Swarm, Agent
except ImportError:
    raise ImportError("Swarm is not installed. Please install it using: pip install git+https://github.com/openai/swarm.git")

def create_swarm_agents():
    """Initialize Swarm agents for benchmarking."""
    agents = {}

    # Create agents for simple tasks
    for task_name, task_func in SIMPLE_TASKS.items():
        agents[task_name] = Agent(
            name=f"Agent_{task_name}",
            instructions=f"Execute the simple task: {task_name}",
            functions=[task_func]
        )

    # Create agents for complex tasks
    for task_name, task_func in COMPLEX_TASKS.items():
        agents[task_name] = Agent(
            name=f"Agent_{task_name}",
            instructions=f"Execute the complex task: {task_name}",
            functions=[task_func]
        )

    return agents


def execute_task_with_swarm(agent):
    """Executes a given task using Swarm."""
    config = load_config()
    client = OpenAI(api_key=config["llm_execution"]["api_key"])
    response = client.chat.completions.create(
        model=config["llm_execution"].get("local_model_name", "gpt-4"),
        messages=[{"role": "user", "content": "Start the task."}]
    )
    return response.choices[0].message.content


def benchmark_swarm():
    """Runs benchmarking tasks on Swarm."""
    print("Running Swarm Benchmarks...")
    agents = create_swarm_agents()

    # Run simple tasks
    for task_name in SIMPLE_TASKS.keys():
        run_benchmark("Swarm", lambda: execute_task_with_swarm(agents[task_name]), task_name)

    # Run complex tasks
    for task_name in COMPLEX_TASKS.keys():
        run_benchmark("Swarm", lambda: execute_task_with_swarm(agents[task_name]), task_name)

if __name__ == "__main__":
    benchmark_swarm()
