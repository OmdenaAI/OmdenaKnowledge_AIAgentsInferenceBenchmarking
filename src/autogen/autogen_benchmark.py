import autogen
import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Configure LLM
config_list = [{
    "model": "llama-3.3-70b-versatile",
    "api_key": os.environ.get("GROQ_API_KEY"),
    "api_type": "groq"
}]

# Initialize Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

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

# Define the agent
writer_agent = autogen.AssistantAgent(
    name="ContentWriter",
    system_message=(
        "Role: Content Writer\n"
        "Goal: Generate a short descriptive paragraph based on a single input keyword.\n"
        "Backstory: A skilled AI writer trained to craft engaging and meaningful content."
    ),
    llm_config={"config_list": config_list},
)

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

# Function to generate text and measure performance
def text_generation(test_queries):
    results = []
    total_keywords = len(test_queries)

    total_start_time = time.time()
    
    for query in test_queries:
        start_time = time.time()
        
        response = user_proxy.initiate_chat(
            writer_agent,
            message=(f"The agent will generate a short, descriptive paragraph based on '{query}'. "
                     "It should ensure clarity, coherence, and relevance while maintaining an engaging tone.\n\n"
                     "Expected Output:\n"
                     "- A well-structured, concise, and descriptive paragraph (50-100 words).\n"
                     "- The paragraph should be grammatically correct and contextually relevant to the keyword.\n"
                     "- The style should be engaging and adaptable to different contexts if required."),
            max_turns=2
        )
        
        end_time = time.time()
        latency = end_time - start_time
        generated_text = str(response)
        
        print(f"Keyword: {query} | Latency: {latency:.4f} sec")
        
        # Append raw results
        results.append((query, latency, generated_text))

    # Rating each paragraph
    rated_results = []
    for keyword, latency, paragraph in results:
        rating = rate_paragraph(paragraph)
        rated_results.append((keyword, latency, paragraph, rating))
        print(f"Keyword: {keyword} | Rating: {rating}/10")

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    throughput = total_keywords / total_time_taken if total_time_taken > 0 else 0

    print(f"\nTotal Keywords Processed: {total_keywords}")
    print(f"Total Time Taken: {total_time_taken:.4f} seconds")
    print(f"Throughput: {throughput:.4f} keywords per second")

    # Save results to CSV
    save_results_to_csv(rated_results)

def save_results_to_csv(results, filename="autogen_benchmark_results.csv"):
    df = pd.DataFrame(results, columns=["Keyword", "Latency (seconds)", "Generated Paragraph", "Response Rating"])
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Results saved to {filename}")

# Test queries
test_queries = [
    'Rainforest', 'Desert', 'Robotics', 'Blockchain', 'Tennis',
    'Golf', 'Coffee', 'Friendship', 'Venice', 'Tokyo',
    'Galaxy', 'DNA', 'Painting', 'Sculpture'
]

# Run benchmarking
if __name__ == "__main__":
    text_generation(test_queries)
