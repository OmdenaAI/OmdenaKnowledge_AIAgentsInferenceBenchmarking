# frameworks/langgraph_benchmark.py

from benchmarks.simple_tasks import SIMPLE_TASKS
from benchmarks.complex_tasks import COMPLEX_TASKS

# Import LangGraph framework (Ensure it's installed via pip)
try:
    from langgraph.graph import StateGraph
    from langchain_community.llms import OpenAI
    from langchain.prompts import PromptTemplate
except ImportError:
    raise ImportError("LangGraph or LangChain is not installed. Please install them using: pip install langgraph langchain openai")

# Define state structure
class BenchmarkState:
    def __init__(self):
        self.results = {}

def create_langgraph_model():
    """Initialize LangGraph model for benchmarking."""
    return OpenAI(model_name="gpt-4", temperature=0.5)

def execute_task_with_langgraph(task_func, state):
    """Executes a given task using LangGraph."""
    llm = create_langgraph_model()
    task_prompt = task_func()
    prompt_template = PromptTemplate(input_variables=["query"], template="{query}")
    response = llm(prompt_template.format(query=task_prompt))
    # Return dictionary instead of modifying state
    return {"results": {task_func.__name__: response}}

def benchmark_langgraph():
    """Runs benchmarking tasks on LangGraph."""
    print("Running LangGraph Benchmarks...")
    
    graph = StateGraph()

    # Add simple tasks as nodes
    for task_name, task_func in SIMPLE_TASKS.items():
        graph.add_node(task_name, lambda x: execute_task_with_langgraph(task_func, x))
    
    # Define execution order
    graph.set_entry_point(list(SIMPLE_TASKS.keys())[0])
    
    compiled_graph = graph.compile()
    
    # Execute with initial empty state
    final_state = compiled_graph.invoke({})
    return final_state.get("results", {})
    
if __name__ == "__main__":
    benchmark_langgraph()