import time
from crewai import Crew, Agent, Task
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq
from groq import Groq
import tracemalloc
import tiktoken

load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])
# LLM Setup
llm = ChatGroq(
        model='groq/llama-3.3-70b-versatile',
        api_key=os.environ['GROQ_API_KEY'],
        temperature=0.1,
        max_tokens=1000,
    )
enc = tiktoken.get_encoding("cl100k_base")  

def rate_paragraph(paragraph: str) -> str:
    """Sends a paragraph to Groq LLM and gets a rating from 1 to 10."""
    messages = [
        {"role": "system", "content": "You are an AI assistant that rates paragraphs on a scale of 1-10 based on:"
                                      " 1. Clarity and coherence"
                                      " 2. Relevance to the topic"
                                      " 3. Engagement and fluency."
                                      " Respond with only a single number between 1 and 10."},
        {"role": "user", "content": paragraph},
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages, 
            model="llama-3.2-90b-vision-preview",
        )
        rating = chat_completion.choices[0].message.content.strip()

        return rating if rating.isdigit() else "Invalid rating received"

    except Exception as e:
        return f"Error: {e}"

    
def writer_agent():

    return Agent(
        role="Content Writer",
        goal="Generate a short descriptive paragraph based on a single input keyword.",
        backstory="A skilled AI writer trained to craft engaging and meaningful content.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

def writer_task(single_keyowrd):
    return Task(
        description=f"The agent will generate a short, descriptive paragraph based on {single_keyowrd}. It should ensure clarity, coherence, and relevance while maintaining an engaging tone.",
        expected_output="""
            A well-structured, concise, and descriptive paragraph (50-100 words).
            The paragraph should be grammatically correct and contextually relevant to the keyword.
            The style should be engaging and adaptable to different contexts if required.""",
        agent=writer_agent(),
    )

def save_results_to_csv(results, total_keywords, total_time_taken, throughput, total_tokens, filename="benchmark_results.csv"):
    df = pd.DataFrame(results, columns=[
    "Keyword", "Latency (seconds)", "Generated Paragraph", "Response Ratings", 
    "Peak Memory", "Memory Delta", "Input Tokens", "Output Tokens", "Total Tokens"
])
    
    # Create a summary row
    summary_data = {
        "Keyword": "SUMMARY",
        "Latency (seconds)": f"Total Keywords Processed: {total_keywords}",
        "Generated Paragraph": f"Total Time Taken: {total_time_taken:.4f} sec",
        "Response Ratings": f"Throughput: {throughput:.4f} keywords/sec",
        "Tokens Used": f"Total Tokens Used: {total_tokens}",
        "Peak Memory": "",
        "Memory Delta": ""
    }
    
    # Append summary row **after results**
    df = pd.concat([df, pd.DataFrame([summary_data])], ignore_index=True)

    df.to_csv(filename, index=False)

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

        crew = Crew(
            agents=[writer_agent()],
            tasks=[writer_task(query)],
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
        print(f"Keyword: {query} | Latency: {latency:.4f} sec | Tokens: {total_tokens} | Peak Memory: {peak_memory:.4f} MB | Memory Delta: {memory_diff:.4f} MB")
    
    # Stop memory tracking
    tracemalloc.stop()

    rated_results = []
    for keyword, latency, paragraph, peak_memory, memory_diff, input_tokens, output_tokens, total_tokens in results:
        rating = rate_paragraph(str(paragraph))  # Get rating from LLM
        rated_results.append((keyword, latency, paragraph, rating, peak_memory, memory_diff, input_tokens, output_tokens, total_tokens))
        print(f"Keyword: {keyword} | Rating: {rating}/10 | Tokens Used: {total_tokens} | Peak Memory: {peak_memory:.4f} MB | Memory Delta: {memory_diff:.4f} MB")

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    throughput = total_keywords / total_time_taken if total_time_taken > 0 else 0

    print(f"\nTotal Keywords Processed: {total_keywords}")
    print(f"Total Time Taken: {total_time_taken:.4f} seconds")
    print(f"Throughput: {throughput:.4f} keywords per second")
    print(f"Total Tokens Used in Benchmark: {total_tokens_used}")

    # Save results to CSV
    save_results_to_csv(
        rated_results, 
        total_keywords, 
        total_time_taken, 
        throughput, 
        total_tokens_used, 
        filename="crewai_benchmark_results.csv"
    )


test_queries = [
    'Rainforest','Desert', 'Robotics', 'Blockchain', 'Tennis',
'Golf', 'Coffee', 'Friendship', 'Venice', 'Tokyo', 
'Galaxy', 'DNA', 'Painting', 'Sculpture'
]

# Run Benchmarking
if __name__ == "__main__":
    text_generation(test_queries)

