import autogen
import time
import tracemalloc
import tiktoken
from groq import Groq
from config.config_loader import get_config
from utils.common_functions import save_results_to_csv, rate_paragraph
from agents import writer_agent


# Define the user proxy
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    system_message="A human admin controlling the text generation process.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False  # Disable Docker
    },
)

def text_generation(test_queries):
    results = []
    total_keywords = len(test_queries)
    total_tokens_used = 0

    tracemalloc.start()
    total_start_time = time.time()
    
    for query in test_queries:
        tracemalloc.reset_peak()
        snapshot_before = tracemalloc.take_snapshot()

        start_time = time.time()
        response = user_proxy.initiate_chat(
            writer_agent(config_list, prompts),
            message=(prompts["task"]["description"].format(keyword=query)
                    + "\n\nExpected Output:\n"
                    + prompts["task"]["expected_output"]),
            max_turns=2
        )
        end_time = time.time()
        latency = end_time - start_time
        generated_text = response.chat_history[-1]["content"] if response.chat_history else "No output"

        snapshot_after = tracemalloc.take_snapshot()
        peak_memory = tracemalloc.get_traced_memory()[1] / 1024**2  # Convert to MB
        memory_diff = sum(stat.size_diff for stat in snapshot_after.compare_to(snapshot_before, 'lineno')) / 1024**2

        input_tokens = len(enc.encode(query))
        output_tokens = len(enc.encode(str(generated_text)))
        total_tokens = input_tokens + output_tokens
        total_tokens_used += total_tokens

        results.append((query, latency, generated_text, None, peak_memory, memory_diff, input_tokens, output_tokens, total_tokens))
    
    tracemalloc.stop()

    rated_results = []
    for keyword, latency, paragraph,_, peak_memory, memory_diff, input_tokens, output_tokens, total_tokens in results:
        rating = rate_paragraph(paragraph, client, prompts, llm_config)
        rated_results.append((keyword, latency, paragraph, rating, peak_memory, memory_diff, input_tokens, output_tokens, total_tokens))

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    throughput = total_keywords / total_time_taken if total_time_taken > 0 else 0

    print(f"\nTotal Keywords Processed: {total_keywords}")
    print(f"Total Time Taken: {total_time_taken:.4f} seconds")
    print(f"Throughput: {throughput:.4f} keywords per second")
    print(f"Total Tokens Used in Benchmark: {total_tokens_used}")

    save_results_to_csv(rated_results, total_keywords, total_time_taken, throughput, total_tokens_used, csv_save, framework="autogen")


config = get_config()
llm_config = config["llm"]
encoder_name = config["tiktoken_encoder"]
prompts = config["prompts"]
csv_save = config["csv"]
benchmark_keywords = config["benchmark"]

client = Groq(api_key=config["api_key"])

config_list = [{
    "model": llm_config["langgraph_agent_model"],
    "api_key": config["api_key"],
    "api_type": llm_config["api_type"],
    "temperature": llm_config["temperature"],
    "max_tokens": llm_config["max_tokens"]
}]

enc = tiktoken.get_encoding(encoder_name) 

if __name__ == "__main__":
    test_queries = benchmark_keywords["test_queries"]
    text_generation(test_queries)
