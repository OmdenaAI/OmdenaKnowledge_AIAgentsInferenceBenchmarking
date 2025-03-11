import time
import sys
from crewai import Crew, Agent, Task
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq
from groq import Groq
import tracemalloc
import tiktoken
import yaml 
from agents import writer_agent
from tasks import writer_task
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from save_to_csv import save_results_to_csv
from rate_paragraph import rate_paragraph


def text_generation(test_queries):
    results = []
    total_keywords = len(test_queries)

    # Start memory tracking
    tracemalloc.start()
    total_start_time = time.time()
    total_tokens_used = 0  # Track total tokens

    for query in test_queries:
        tracemalloc.reset_peak()
        snapshot_before = tracemalloc.take_snapshot()

        agent = writer_agent(prompts, llm) 
        task = writer_task(query, prompts, llm) 

        crew = Crew(
            agents=[agent], 
            tasks=[task], 
            verbose=False
        )

        start_time = time.time()
        response = crew.kickoff()  # Get full CrewAI response
        end_time = time.time()

        latency = end_time - start_time
        print(response)

        # Measure memory usage
        snapshot_after = tracemalloc.take_snapshot()
        peak_memory = tracemalloc.get_traced_memory()[1] / 1024**2  # Convert to MB
        memory_diff = sum(stat.size_diff for stat in snapshot_after.compare_to(snapshot_before, 'lineno')) / 1024**2

        # Extract generated text
        generated_text = response.tasks_output[0] if response.tasks_output else "No output"

        # Manually count tokens
        input_tokens = len(enc.encode(query))  # Tokens for input query
        output_tokens = len(enc.encode(str(generated_text)))  # Tokens for generated paragraph
        total_tokens = input_tokens + output_tokens  # Total tokens for this request

        total_tokens_used += total_tokens  # Accumulate total token count

        results.append((query, latency, generated_text, peak_memory, memory_diff, input_tokens, output_tokens, total_tokens))
    
    # Stop memory tracking
    tracemalloc.stop()

    rated_results = []
    for keyword, latency, paragraph, peak_memory, memory_diff, input_tokens, output_tokens, total_tokens in results:
        rating = rate_paragraph(str(paragraph), client, prompts, llm_config)   
        rated_results.append((keyword, latency, paragraph, rating, peak_memory, memory_diff, input_tokens, output_tokens, total_tokens))
        print(f"Keyword: {keyword} | Latency: {latency:.4f} sec | Rating: {rating}/10 | Tokens Used: {total_tokens} | Peak Memory: {peak_memory:.4f} MB | Memory Delta: {memory_diff:.4f} MB")

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    throughput = total_keywords / total_time_taken if total_time_taken > 0 else 0

    print(f"\nTotal Keywords Processed: {total_keywords}")
    print(f"Total Time Taken: {total_time_taken:.4f} seconds")
    print(f"Throughput: {throughput:.4f} keywords per second")
    print(f"Total Tokens Used in Benchmark: {total_tokens_used}")

    # Save results to CSV
    save_results_to_csv(results, total_keywords, total_time_taken, throughput, total_tokens, csv_save, framework="crewai")


DOTENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(DOTENV_PATH)

# Load config.yaml from the project root
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

llm_config = config["llm"]
encoder_name = config["tiktoken_encoder"]
prompts = config["prompts"]
csv_save = config["csv"]
benchmark_keywords = config["benchmark"]

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# LLM Setup
llm = ChatGroq(
    model=llm_config["crewai_agent_model"],
    api_key=os.environ['GROQ_API_KEY'],
    temperature=llm_config["temperature"],
    max_tokens=llm_config["max_tokens"],
    )

enc = tiktoken.get_encoding(encoder_name) 

# Run Benchmarking
if __name__ == "__main__":
    test_queries = benchmark_keywords["test_queries"]
    text_generation(test_queries)
