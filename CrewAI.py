import time
from crewai import Crew, Agent, Task
import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# import matplotlib.pyplot as plt
import pandas as pd
from langchain_groq import ChatGroq
from groq import Groq
load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])
# LLM Setup
llm = ChatGroq(
        model='groq/llama-3.3-70b-versatile',
        api_key=os.environ['GROQ_API_KEY'],
        temperature=0.0,
        max_tokens=1000,
    )
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
def save_results_to_csv(results, filename="Crbenchmark_results.csv"):
    df = pd.DataFrame(results, columns=["Keyword", "Latency (seconds)", "Generated Paragraph","response ratings"])
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Results saved to {filename}")
def text_generation(test_queries):
    results = []
    total_keywords = len(test_queries)
    total_start_time = time.time()
    for query in test_queries:
        crew = Crew(
            agents=[writer_agent()],
            tasks=[writer_task(query)],
            verbose=False
        ) 
        start_time = time.time()
        response = crew.kickoff().tasks_output[0]   # Extract result properly
        end_time = time.time()
        latency = end_time - start_time
        generated_text = str(response)  # Convert to string
        results.append((query, latency, generated_text))
        print(f"Keyword: {query} | Latency: {latency:.4f} sec")
    
    rated_results = []
    for keyword, latency, paragraph in results:
        rating = rate_paragraph(paragraph)  # Get rating from LLM
        rated_results.append((keyword, latency, paragraph, rating))
        print(f"Keyword: {keyword} | Rating: {rating}/10")
    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    throughput = total_keywords / total_time_taken if total_time_taken > 0 else 0
    print(f"\nTotal Keywords Processed: {total_keywords}")
    print(f"Total Time Taken: {total_time_taken:.4f} seconds")
    print(f"Throughput: {throughput:.4f} keywords per second")
    save_results_to_csv(rated_results)
test_queries = [
    'Rainforest','Desert', 'Robotics', 'Blockchain', 'Tennis',
'Golf', 'Coffee', 'Friendship', 'Venice', 'Tokyo', 
'Galaxy', 'DNA', 'Painting', 'Sculpture'
]
# Run Benchmarking
if __name__ == "__main__":
    text_generation(test_queries)