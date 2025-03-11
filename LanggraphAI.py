import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,END
# from pi.graph import StateGraph,END
from typing import TypedDict
from groq import Groq
# Load environment variables
load_dotenv()
# Initialize Groq client for rating
client = Groq(api_key=os.environ['GROQ_API_KEY'])
# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=1000,
)
class AgentState(TypedDict):
    keyword: str
    response: str
def generate_paragraph(state: AgentState) -> AgentState:
    keyword = state["keyword"]
    prompt = f"""
    Role: Content Writer
    Goal: Generate a short descriptive paragraph based on a single input keyword.
    Backstory: A skilled AI writer trained to craft engaging and meaningful content.
    Task Description: Generate a short, descriptive paragraph based on the keyword '{keyword}'.
    It should ensure clarity, coherence, and relevance while maintaining an engaging tone.
    Expected Output:
    - A well-structured, concise, and descriptive paragraph (50-100 words).
    - The paragraph should be grammatically correct and contextually relevant to the keyword.
    - The style should be engaging and adaptable to different contexts if required.
    """
    
    response = llm.invoke(prompt)
    return {"keyword": keyword, "response": response.content}
def rate_paragraphs(paragraphs):
    rated_results = []
    for keyword, latency, paragraph in paragraphs:
        messages = [
            {"role": "system", "content": "You are an AI assistant that rates paragraphs on a scale of 1-10 based on:\n1. Clarity and coherence\n2. Relevance to the topic\n3. Engagement and fluency.\nRespond with only a single number between 1 and 10."},
            {"role": "user", "content": paragraph},
        ]
        try:
            chat_completion = client.chat.completions.create(
                messages=messages, 
                model="llama-3.2-90b-vision-preview",
            )
            rating = chat_completion.choices[0].message.content.strip()
            rating = rating if rating.isdigit() else "Invalid rating received"
        except Exception as e:
            rating = f"Error: {e}"
        rated_results.append((keyword, latency, paragraph, rating))
        print(f"Keyword: {keyword} | Rating: {rating}/10")
    return rated_results
# Define workflow
workflow = StateGraph(AgentState)
workflow.add_node("writer", generate_paragraph)
workflow.set_entry_point("writer")
workflow.set_finish_point("writer")
app = workflow.compile()
def benchmark(test_queries):
    results = []
    total_keywords = len(test_queries)
    total_start_time = time.time()
    
    for query in test_queries:
        start_time = time.time()
        result = app.invoke({"keyword": query})
        end_time = time.time()
        
        latency = end_time - start_time
        paragraph = result['response']
        results.append((query, latency, paragraph))
        print(f"Keyword: {query} | Latency: {latency:.4f} sec")
    
    rated_results = rate_paragraphs(results)
    
    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    throughput = total_keywords / total_time_taken if total_time_taken > 0 else 0
    
    print(f"\nTotal Keywords Processed: {total_keywords}")
    print(f"Total Time Taken: {total_time_taken:.4f} seconds")
    print(f"Throughput: {throughput:.4f} keywords per second")
    
    df = pd.DataFrame(rated_results, columns=["Keyword", "Latency (seconds)", "Generated Paragraph", "Response Rating"])
    df.to_csv("Labenchmark_results.csv", index=False, encoding="utf-8")
    print("Results saved to benchmark_results.csv")
# Define test queries
test_queries = ['Rainforest', 'Desert', 'Robotics', 'Blockchain', 'Tennis',
                'Golf', 'Coffee', 'Friendship', 'Venice', 'Tokyo', 
                'Galaxy', 'DNA', 'Painting', 'Sculpture']
if __name__ == "__main__":
    benchmark(test_queries)