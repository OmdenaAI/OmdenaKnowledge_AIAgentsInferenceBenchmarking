# frameworks/autogen_benchmark.py

from benchmarks.simple_tasks import SIMPLE_TASKS
from benchmarks.complex_tasks import COMPLEX_TASKS

# Import AutoGen framework (Ensure it's installed via pip)
try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat
except ImportError:
    raise ImportError("AutoGen is not installed. Please install it using: pip install autogen-agentchat autogen-ext[openai]")

def create_autogen_agents():
    """Initialize AutoGen agents for benchmarking."""
    assistant = AssistantAgent("assistant")
    user_proxy = UserProxyAgent("user_proxy")
    return assistant, user_proxy

def execute_task_with_autogen(task_func):
    """Executes a given task using AutoGen."""
    assistant, user_proxy = create_autogen_agents()
    task_prompt = task_func()
    group_chat = GroupChat(
        agents=[assistant, user_proxy],
        messages=[],
        max_round=5
    )
    response = group_chat.run(task_prompt)
    return response

def benchmark_autogen():
    """Runs benchmarking tasks on AutoGen."""
    print("Running AutoGen Benchmarks...")

    # Run simple tasks
    for task_name, task_func in SIMPLE_TASKS.items():
        run_benchmark("AutoGen", lambda: execute_task_with_autogen(task_func), task_name)

    # Run complex tasks
    for task_name, task_func in COMPLEX_TASKS.items():
        run_benchmark("AutoGen", lambda: execute_task_with_autogen(task_func), task_name)

if __name__ == "__main__":
    benchmark_autogen()
