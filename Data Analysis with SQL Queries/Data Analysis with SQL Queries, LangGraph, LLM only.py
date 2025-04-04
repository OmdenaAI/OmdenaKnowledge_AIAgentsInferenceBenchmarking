# Data Analysis with SQL Queries using Amazon Dataset

"""
This Q&A System App is built for Q&A over a SQL database. The application gives an LLM access to tools for 
querying and interacting with the data. It follows the recommendation from LangGraph to load the CSV file into a 
SQL database instead of directly working with a CSV file. Using SQL requires executing model-generated SQL queries. 
Using SQL to interact with CSV data is the recommended approach because it is easier to limit permissions and sanitize 
queries than with arbitrary Python. I already have loaded the CSV file in a SQL database. After that it is possible to use 
all of the chain and agent-creating techniques outlined in the SQL tutorial provided by LangGraph. In the below I use SQLite.
"""

# Import libraries
import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

import json
import logging
import csv
import re

import time

import pandas as pd

import getpass

from typing import TypedDict, Dict, Any, List, Optional
from typing_extensions import Annotated

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph

from langchain import hub

from data.questions import queries # Import the dictionary of queries containing the questions and expected results

from utils.metrics_collector import MetricsCollector

from utils.logging_config import *

# Set the framework that will be benchmarked
FRAMEWORK = "LangGraph"

# Load environment variables
load_dotenv()

# Load config.yaml file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the LLM
def get_llm(config):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model=config['model']['name'],
        api_key=groq_key,
        temperature=config['model']['temperature'],
    )

llm = get_llm(config)

# Get API key for GROQ
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# Connect to database
db = SQLDatabase.from_uri("sqlite:///data/amazon_cleaned.db")

# Function for extracting the questions from the queries dictionary
def extract_questions(queries):
    """Loops over the queries in the dictionary and returns each question formatted as 'question_X: question'."""
    questions = {}
    
    # Loop over each entry in the queries dictionary
    for key, value in queries.items():
        # Extract the question and store it in the questions dictionary
        questions[key] = value['question']
    
    return questions

# Extract the questions from the queries dictionary
questions = extract_questions(queries)

# Define the LangGraph state of the application
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    numeric: float

# Pull a prompt from the Prompt Hub to instruct the model.
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 1

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

# Function for writing the SQL query
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 1,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# Create a function for executing a SQL Query
def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

# Generate an answer to the question given the information pulled from the database
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        "\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# Extract the numerical value from the answer
# Return "Error" if no numerical value is found
def extract_numerical_value(state: State):
    """Extracts the first numerical value from a string (including decimals)."""
    match = re.search(r'\d+(\.\d+)?', state['answer'])  # Regex to find integers or decimal numbers
    if match:
        return {'numeric': float(match.group())}  # Convert the matched value to a float
    return 'Error retrieving numeric answer' # Return None if no numerical value is found

# Create initial state for the workflow
def create_initial_state() -> State:
    """Create initial state for the workflow.
    
    Returns:
        State: Initial workflow state
    """
    return State(
        question='',
        query='',
        result='None',
        answer='',
        numeric=0.0
    )

# Function to save the result in a CSV file for the question and final numeric answer
def save_results_to_csv(results, csv_filename, iteration, log_dir, framework):
    n_iteration = iteration + 1

    # Open the CSV file in append mode ('a')
    name = log_dir / (csv_filename + "_iter" + str(n_iteration) + "_" + str(framework) + ".csv")
    
    with open(name, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file is empty (only the first time)
        if file.tell() == 0:
            writer.writerow(["question", "answer"])
            
        n_results = len(results)

        # Write each entry of the results as a new row
        for i in range(n_results):
            writer.writerow(results[i])
        
    print(f"Results for iteration {n_iteration} saved to {name}")

# Build the graph
def build_graph(
    config: Dict[str, Any],
    logger: logging.Logger,
    llm: Optional[ChatGroq] = None
) -> StateGraph:
    
    """Build the complete agent workflow graph.
    Args:
        config: Configuration dictionary
        logger: Logger instance
        llm: Optional ChatGroq instance (will be created if not provided)
        
    Returns:
        StateGraph: Configured workflow graph
    """
    
    # Create LLM if not provided
    if llm is None:
        raise ValueError("LLM is not provided")

    # Create initial state
    initial_state = create_initial_state()

    # Save the results of each query in a list that will later be saved in a CSV file
    results = []

    # Orchestrate with LangGraph
    graph_builder = StateGraph(State).add_sequence(
        [write_query, execute_query, generate_answer, extract_numerical_value]
        )
    graph_builder.add_edge(START, "write_query")
    graph = graph_builder.compile()

    results = []

    # Loop over the queries in the dictionary with questions
    for key, value in questions.items():
        print(f"Processing question: {key, value}")

        # Execute the graph for the current question        
        for step in graph.stream(
            {"question": value}, stream_mode="updates"
        ):
            print(f"Step: {step}")

        # Append the results to the list
        results.append(
                    (
                        value,  # The question value
                        step['extract_numerical_value']['numeric']     # Safe access to numeric value
                    )
                )
        
    # After looping, print the results to verify
    logger.debug("Results generated")

    # Create final state
    final_state = StateGraph(State)

    return results, initial_state, final_state

# Function to find the latest metrics file (starts with 'metrics' and ends with '.json')
def get_latest_metrics_file(log_dir):
    # Use glob to search for files starting with 'metrics' and ending with '.json'
    metrics_files = glob.glob(str(log_dir / 'metrics*.json'))
    
    if not metrics_files:
        raise FileNotFoundError("No metrics file found in the directory.")
    
    # Find the most recent metrics file based on the filename (assumes filenames include timestamps)
    latest_file = max(metrics_files, key=lambda x: Path(x).stat().st_mtime)
    return Path(latest_file)

# Function to compare the expected results with the generated results
def compare_answers(log_dir, expected_file, csv_filename, num_iterations, framework):
    total_match_percentages = 0
    iteration_match_percentages = []  # This will store match percentage for each iteration

    # Read the CSV file with expected results into pandas DataFrame
    expected_df = pd.read_csv(expected_file)

    # Read the CSV files with generated results and compare for each iteration
    for iteration in range(num_iterations):
        n_iteration = iteration + 1
        
        name = log_dir / (csv_filename + "_iter" + str(n_iteration) + "_" + str(framework) + ".csv")
        results_df = pd.read_csv(name)
        
        # Ensure the DataFrames have the same number of rows
        if len(expected_df) != len(results_df):
            raise ValueError("The number of rows in the two files do not match.")
        
        # Compare the 'answer' columns of both DataFrames
        matches = (expected_df['answer'] == results_df['answer'])
        
        # Calculate the percentage of matching answers
        match_percentage = (matches.sum() / len(matches)) * 100

        # Add match percentage to the list and the total
        iteration_match_percentages.append(match_percentage)
        total_match_percentages += match_percentage


    # Load the existing metrics data (assuming it's a JSON file)    
    metrics_file = get_latest_metrics_file(log_dir)

    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    
    # Add the comparison results to each iteration in the LangGraph section
    for i in range(num_iterations):
        iteration_data = metrics_data[FRAMEWORK][i]
        iteration_data["match_percentage"] = iteration_match_percentages[i]
    
    # Add the average match percentage to the overall result
    metrics_data["LangGraph"].append({
        "iteration": "average",
        "match_percentage": total_match_percentages / num_iterations
    })
    
    # Save the updated metrics back to the file
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Comparison results added to {metrics_file}")

# Run the benchmark process with iterations
def run_benchmark(config: Dict[str, Any], llm: ChatGroq, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Run the benchmark process with iterations"""
    logger.debug("Starting benchmark run...")

    # Get number of iterations from config
    num_iterations = config['benchmarks']['iterations']
    logger.info(f"Running benchmark for {num_iterations} iterations")
    
    """
    # Set up path for logging and metrics    
    # Get the current directory where the script is running
    current_dir = Path(__file__).parent

    # Define the 'results' directory in the same folder as the script
    log_dir = current_dir / 'results'

    # Create the directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)"

    logger.info(f"Saving results to: {log_dir.resolve()}")

    """

    # Get the current working directory (works in both scripts and interactive environments)
    current_dir = Path.cwd()

    # Define the 'results' directory in the same folder as the script
    log_dir = current_dir / 'results'

    # Create the directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to: {log_dir.resolve()}")

    # Initialize MetricsCollector with the dynamic log_dir
    metrics = MetricsCollector(log_dir)

    # Run iterations
    for iteration in range(num_iterations):
        logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")

        metrics.start_iteration(FRAMEWORK, iteration)
        start_time = time.time()

        results, initial_state, final_state = build_graph(config, logger, llm)
        api_latency = time.time() - start_time

        metrics.increment_api_calls()
        metrics.add_api_latency(api_latency)
        
        save_results_to_csv(results, "results", iteration, log_dir, FRAMEWORK)

        logger.info(f"Completed iteration {iteration + 1}")
        print("Start time", start_time)
        print("Latency time", api_latency)

        metrics.end_iteration()
        metrics.save_iteration(FRAMEWORK)
    
    metrics.save_metrics()
    metrics.generate_plots()

    # Generate metric for calculating the percentage of correct answers
    expected_file = 'data/expected_results.csv'
    file_name = 'results'
    compare_answers(log_dir, expected_file, file_name, num_iterations, FRAMEWORK)
    
    logger.info("Benchmark completed successfully")

# Main execution function
def main():
    """Main entry point for the application."""

    # Setup logging
    logger = setup_logging(
        log_file=config['logging']['file'],
        log_level=config['logging']['level'],
        log_format=config['logging']['format']
    )
    try:        
        # Run workflow
        run_benchmark(config, llm, logger)  
        
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main()
