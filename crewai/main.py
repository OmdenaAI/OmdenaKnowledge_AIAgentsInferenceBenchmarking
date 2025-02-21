import time
import csv
import numpy as np
from crewai import Crew, Agent, Task
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# LLM Setup
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ['GROQ_API_KEY'],
    model_name="groq/llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1000,
)

# Define Help Desk Agent
def helpdesk_agent():
    return Agent(
        role="Help Desk Assistant",
        goal="Provide accurate and concise answers to user queries.",
        backstory="An AI trained in troubleshooting and customer support.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

# Define Help Desk Task
def helpdesk_task(user_question):
    return Task(
        description=f"Answer the user's help desk query: \n\n**Question:** {user_question}",
        expected_output="A clear and complete answer to the user's question.",
        agent=helpdesk_agent(),
    )

# Fallback phrases for Escalation Rate
ESCALATION_PHRASES = ["please contact support", "unable to assist", "refer to our team"]

# Error Metrics Functions
def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

def mean_absolute_percentage_error(actual, predicted):
    return np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100

def root_mean_square_error(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted))**2))

# Benchmarking function
def benchmark_helpdesk(test_queries, output_csv="benchmark_results.csv"):
    results = []
    latencies, token_usages, completions, escalations = [], [], [], []

    for query in test_queries:
        start_time = time.time()  # Start time
        crew = Crew(
            agents=[helpdesk_agent()],
            tasks=[helpdesk_task(query)],
            verbose=True
        ) 
        response = crew.kickoff()
        end_time = time.time()  # End time

        # Calculate Latency
        latency = end_time - start_time  
        latencies.append(latency)

        # Token Usage (Estimated from output length)
        token_usage = len(str(response).split())  
        token_usages.append(token_usage)

        # Task Completion: Assume completion if response contains a clear answer
        completion = 1 if len(str(response).strip()) > 20 else 0  
        completions.append(completion)

        # Escalation Detection
        escalation = any(phrase in str(response).lower() for phrase in ESCALATION_PHRASES)
        escalations.append(int(escalation))

        # Store Results
        results.append([query, latency, token_usage, completion, escalation])

    # Calculate Error Metrics
    actual = [1] * len(test_queries)  # Assuming the ideal case is all tasks completed successfully
    mae = mean_absolute_error(actual, completions)
    mape = mean_absolute_percentage_error(actual, completions)
    rmse = root_mean_square_error(actual, completions)

    # Write results to CSV
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Query", "Latency (s)", "Token Usage", "Completed (1/0)", "Escalated (1/0)"])
        writer.writerows(results)
        writer.writerow([])
        writer.writerow(["MAE", mae])
        writer.writerow(["MAPE", mape])
        writer.writerow(["RMSE", rmse])

    print(f"Benchmarking completed. Results saved to {output_csv}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

# Example Queries for Testing
test_queries = [
    "How do I reset my password?",
    "What should I do if my internet is slow?",
    "Can you help me recover my account?",
    "The website is not loading. What can I do?",
    "I forgot my email password. How can I reset it?"
]

# Run Benchmarking
if __name__ == "__main__":
    benchmark_helpdesk(test_queries)
