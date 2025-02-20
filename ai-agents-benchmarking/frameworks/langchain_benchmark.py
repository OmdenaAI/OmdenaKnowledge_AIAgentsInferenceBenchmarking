# frameworks/langchain_benchmark.py

from benchmarks.simple_tasks import SIMPLE_TASKS
from benchmarks.complex_tasks import COMPLEX_TASKS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from utils.config_loader import get_llm_config  # ✅ Config loader in utils

# Import LangChain framework
try:
    from langchain_community.llms import OpenAI, Ollama  # Updated import path for both OpenAI and Ollama
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
except ImportError:
    raise ImportError("LangChain is not installed. Please install it using: pip install langchain-community openai langchain-ollama")

def create_langchain_model():
    """Initialize LangChain model for benchmarking, using either OpenAI or Ollama based on configuration."""
    mode = config["llm_execution"]["mode"]
    
    if mode == "cloud":
        return OpenAI(model=config["llm_execution"].get("cloud_model", "gpt-4"), 
                      temperature=0.5, 
                      openai_api_key=config["llm_execution"].get("api_key", ""))
    elif mode == "local":
        return Ollama(model=config["llm_execution"].get("local_model_name", "llama3.2"))
    else:
        raise ValueError("Invalid LLM execution mode. Choose 'local' or 'cloud'.")

def execute_task_with_langchain(task_func):
    """Executes a given task using LangChain, configured for either OpenAI (cloud) or Ollama (local)."""
    llm = create_langchain_model()
    task_prompt = task_func()
    prompt_template = PromptTemplate(input_variables=["query"], template="{query}")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run({"query": task_prompt})
    return result

def benchmark_langchain():
    """Runs benchmarking tasks on LangChain."""
    print("Running LangChain Benchmarks...")
    
    # Run simple tasks
    for task_name, task_func in SIMPLE_TASKS.items():
        run_benchmark("LangChain", lambda: execute_task_with_langchain(task_func), task_name)
    
    # Run complex tasks
    for task_name, task_func in COMPLEX_TASKS.items():
        run_benchmark("LangChain", lambda: execute_task_with_langchain(task_func), task_name)

if __name__ == "__main__":
    benchmark_langchain()
