import json
import glob
from pathlib import Path
import pandas as pd
import re
from typing import Optional
import os
from langchain_groq import ChatGroq
from data.questions import queries  # Import the dictionary of queries containing the questions and expected results

def get_llm(config):
    """
    Uses the information stored in config to load the LLM

    Args:
        config: YAML file that contains information for the LLM model to be used in the application

    Returns:
        ChatGroq: model loaded using ChatGroq
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model=config['model']['name'],
        api_key=groq_key,
        temperature=config['model']['temperature'],
    )

def extract_questions(queries):
    """
    Loops over the queries in the dictionary and returns each question formatted as 'question_X: question'.
    
    Args:
        queries: Dictionary containing the questions, expected SQL queries and answers

    Returns:
        questions: Dictionary with each question formatted as {'question_X: question}

    """
    questions = {}
    
    # Loop over each entry in the queries dictionary
    for key, value in queries.items():
        # Extract the question and store it in the questions dictionary
        questions[key] = value['question']
    
    return questions

# Extract the questions from the queries dictionary
questions = extract_questions(queries)


def extract_numerical_value(answer: str) -> Optional[float]:
    """
    Extracts the first numerical value from a string (including decimals).
    
    Args:
        answer: String of the final generated answer
    
    Returns:
        float: Final numeric answer to the question
    """
    match = re.search(r'\d+(\.\d+)?', answer)  # Regex to find integers or decimal numbers
    if match:
        return float(match.group())  # Convert the matched value to a float
    return 'Error retrieving numeric answer' # Return None if no numerical value is found

def save_results_to_csv(results, csv_filename, iteration, log_dir, framework):
    """
    Function to save the result in a CSV file for the question and final numeric answer
    
    Args:
        results: List of the saved results
        csv_filename: String containing the name for the CSV file where the results will be saved
        iteration: Integer representing the current iteration
        log_dir: Directory where the CSV file will be saved
        framework: String representing the framework to be benchmarked
        
    Prints:
        Prints out the name of the CSV file where the results for the current iteration are saved.
    """
    n_iteration = iteration + 1

    # Define the full path to the CSV file
    file_path = log_dir / (csv_filename + "_iter" + str(n_iteration) + "_" + str(framework) + ".csv")

    # Convert the results into a pandas DataFrame
    df = pd.DataFrame(results, columns=["question", "answer"])

    # Write to CSV (appending to the file if it already exists)
    df.to_csv(file_path, mode='a', header=not file_path.exists(), index=False)

    print(f"Results for iteration {n_iteration} saved to {file_path}")

def get_latest_metrics_file(log_dir):
    """
    Function to find the latest metrics file (starts with 'metrics' and ends with '.json')

    Args:
        log_dir: Directory where the json files are saved
    
    Returns:
        Path(latest_file): The path to the latest file with metrics
    """
    # Use glob to search for files starting with 'metrics' and ending with '.json'
    metrics_files = glob.glob(str(log_dir / 'metrics*.json'))
    
    if not metrics_files:
        raise FileNotFoundError("No metrics file found in the directory.")
    
    # Find the most recent metrics file based on the filename (assumes filenames include timestamps)
    latest_file = max(metrics_files, key=lambda x: Path(x).stat().st_mtime)
    return Path(latest_file)

def compare_answers(log_dir, expected_file, csv_filename, num_iterations, framework):
    """
        Function to compare the expected answers with the generated answers
    
        Args:
            log_dir: Directory where the CSV file will be saved
            expected_file: The CSV file with the expected answers
            csv_filename: String containing the name for the CSV file where the results will be saved
            num_iterations: Integer representing the total number of iterations for the benchmark
            framework: String representing the framework to be benchmarked

        Returns:
        Saves the average match percentage to the overall result for each iteration of the benchmark stored in the json file
        """
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
        iteration_data = metrics_data[framework][i]
        iteration_data["match_percentage"] = iteration_match_percentages[i]
    
    # Add the average match percentage to the overall result
    metrics_data[framework].append({
        "iteration": "average",
        "match_percentage": total_match_percentages / num_iterations
    })
    
    # Save the updated metrics back to the file
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Comparison results added to {metrics_file}")