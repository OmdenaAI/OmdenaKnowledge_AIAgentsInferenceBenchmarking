# frameworks/langgraph_benchmark.py

from benchmarks.simple_tasks import SIMPLE_TASKS
from benchmarks.complex_tasks import COMPLEX_TASKS
import logging
import time
from utils.latency_tracker import LatencyTracker
from utils.result_saver import save_results_to_csv, plot_benchmark_results, load_results_from_csv
from utils.token_tracker import TokenTracker
from utils.memory_tracker import MemoryTracker
from utils.accuracy_calculator import calculate_accuracy
from utils.config_loader import get_llm_config
import os

# ==============================
# ✅ FRAMEWORK NAME
# ==============================
def get_langgraph_framework_name():
    return "LangGraph"

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
# ✅ IMPORT LangGraph FRAMEWORK
# ==============================
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.runnables import RunnablePassthrough
    from langgraph.graph import StateGraph, END
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.llms import Ollama
    from typing import TypedDict, List, Dict, Union
except ImportError:
    raise ImportError("LangGraph is not installed. Please install it using: pip install langgraph langchain-core langchain-community")

# ==============================
# ✅ LangGraph LLM SETUP
# ==============================
def get_langgraph_model(model_name=None):
    """Initialize LangGraph model for benchmarking, using either OpenAI or Ollama based on configuration."""
    llm_config = get_llm_config(model_name)
    api_type = llm_config["api_type"]

    if api_type == "openai":
        return ChatOpenAI(model=llm_config['model_name'],
                          temperature=llm_config['temperature'],
                          openai_api_key=llm_config['api_key'])
    else:
        return Ollama(model=llm_config['model_name'],
                       base_url=llm_config['base_url'],
                       temperature=llm_config['temperature'])

# ==============================
# ✅ LangGraph STATE
# ==============================
class AgentState(TypedDict):
    messages: List[str]

# ==============================
# ✅ LangGraph AGENT
# ==============================
# def agent(state: AgentState) -> Dict:
#     """LangGraph agent."""
#     last_message = state["messages"][-1]
#     return {"messages": [last_message]}

# ==============================
# ✅ EXECUTE TASKS WITH LangGraph
# ==============================
def execute_task_with_langgraph(task_func, model_name=None):
    """Executes a given task using LangGraph, configured for either OpenAI (cloud) or Ollama (local)."""
    tracker = LatencyTracker()
    memory_tracker = MemoryTracker()
    llm_config = get_llm_config(model_name)
    token_tracker = TokenTracker(llm_config.get("model_name", "gpt-4"), llm_config.get("api_type", "openai"))

    tracker.start()
    memory_tracker.start()
    try:
        llm = get_langgraph_model(model_name)
        task_data = task_func()
        prompt = task_data["prompt"]
        expected_answer = task_data.get('expected_answer', '')
        task_type = task_data.get('task_type', 'task')

        # Count tokens for the prompt
        token_count = token_tracker.count_tokens(prompt)

        # LangGraph setup
        prompt_template = ChatPromptTemplate.from_messages([("human", "{messages}")])
        agent_runnable = prompt_template | llm
        def agent(state):
            last_message = state["messages"][-1]
            response: Union[str, AIMessage] = agent_runnable.invoke({"messages": last_message})
            if isinstance(response, str):
                return {"messages": [response]}
            else:
                return {"messages": [response.content]}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)
        graph = workflow.compile()

        # Execute LangGraph
        exec_time = tracker.start()
        result = graph.invoke({"messages": [prompt]})["messages"][0] # changed this line
        exec_time = tracker.stop()
        peak_memory = memory_tracker.stop()

        # Calculate tokens for the result
        response_tokens = token_tracker.count_tokens(str(result))
        total_tokens = token_tracker.get_total_tokens()

        accuracy = calculate_accuracy(result, expected_answer)

        print(f"\nLLM Output: {result}")
        print(f"Task: {task_type} | Accuracy: {accuracy}% | Time: {exec_time}s")
        print(f"Total Tokens Used: {total_tokens}")
        print(f"Peak Memory Usage: {peak_memory:.2f} MB\n")

        return accuracy, exec_time, total_tokens, peak_memory

    except Exception as e:
        exec_time = tracker.stop()
        peak_memory = memory_tracker.stop()
        logger.exception(f"❌ Error during task execution: {e} | Time: {exec_time}s | Memory: {peak_memory:.2f} MB")
        return 0.0, exec_time, 0, peak_memory

# ==============================
# ✅ BENCHMARK ALL TASKS
# ==============================
def benchmark_langgraph():
    """Runs benchmarking tasks on LangGraph."""
    print("🚀 Running LangGraph Benchmarks...\n")

    total_tracker = LatencyTracker()
    total_tracker.start()

    all_results = []

    # Benchmark Simple Tasks
    for task_name, task_func in SIMPLE_TASKS.items():
        accuracy, exec_time, total_tokens, peak_memory = execute_task_with_langgraph(task_func)
        all_results.append((task_name, accuracy, exec_time, total_tokens, peak_memory))
        print(f"Task: {task_name} | Accuracy: {accuracy}% | Time: {exec_time:.2f}s | Tokens: {total_tokens} | Memory: {peak_memory:.2f} MB\n")

    # Benchmark Complex Tasks
    for task_name, task_func in COMPLEX_TASKS.items():
        accuracy, exec_time, total_tokens, peak_memory = execute_task_with_langgraph(task_func)
        all_results.append((task_name, accuracy, exec_time, total_tokens, peak_memory))
        print(f"Task: {task_name} | Accuracy: {accuracy}% | Time: {exec_time:.2f}s | Tokens: {total_tokens} | Memory: {peak_memory:.2f} MB\n")

    total_time = total_tracker.stop()

    # Save and Plot Results
    framework_name = get_langgraph_framework_name()
    csv_path = save_results_to_csv(all_results, framework_name)
    df = load_results_from_csv(csv_path)
    plot_benchmark_results(df, title=f"{framework_name} Benchmark",save_path=csv_path.replace(".csv", ""))

    print(f"⏰ Total Benchmark Time: {total_time:.2f}s")

# ==============================
# ✅ MAIN FUNCTION
# ==============================
if __name__ == "__main__":
    benchmark_langgraph()
