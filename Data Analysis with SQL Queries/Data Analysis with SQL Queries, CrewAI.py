# Import libraries
import os
import yaml
import time
import getpass
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from data.questions import queries  # Import the dictionary of queries containing the questions and expected results
from utils.metrics_collector import MetricsCollector
from utils.logging_config import setup_logging
from utils.helper_functions import get_llm, extract_questions, extract_numerical_value, save_results_to_csv, compare_answers
from langchain_core.tools import tool
from langchain_community.tools.sql_database.tool import ListSQLDatabaseTool, InfoSQLDatabaseTool
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Set the framework that will be benchmarked
FRAMEWORK = "CrewAI"

# Load environment variables
load_dotenv()

# Load config.yaml file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the LLM
llm = get_llm(config)

# Get API key for GROQ
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# Get API key for GROQ
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Connect to database
db = SQLDatabase.from_uri("sqlite:///data/amazon_cleaned.db")

# Extract the questions from the queries dictionary
questions = extract_questions(queries)

# Define tools that the agent will use to interact with the database
# Tool to fetch the available tables from the database
@tool("list_tables_tool")
def list_tables_tool() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("get_schema_tool")
def get_schema_tool(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

@tool("execute_sql_tool")
def execute_sql_tool(query: str) -> str:
    """
    Executes a SQL query against the database and returns the result. If the query fails,
    it returns an error message prompting the user to rewrite the query and try again.

    Args:
        query (str): The SQL query to be executed against the database.

    Returns:
        str: The result of the query if successful, or an error message if the query fails.
    """
    # Run the query against the database without throwing errors (i.e., returns None or failure without raising an exception)
    result = db.run_no_throw(query)

    # If the result is empty or None, it means the query failed, so return an error message
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."

    # Return the result of the successful query execution
    return result

# Create the agent that will write the SQL query
senior_data_analyst = Agent(
    role="Senior Data Analyst",
    goal="You are a SQL expert with a strong attention to detail."
    "Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer."
    "DO NOT call any tool besides the tools that you were given."
    "When generating the query:"
    "Use the tools list_tables_tool and get_schema_tool to find the correct table names and column names from the database schema."
    "Then use the information to write an SQL query that answers the input question."
    "Execute the SQL query using the execute_sql_tool tool."
    "Based on the results of the query, return the answer to the input question as a simple numerical value."
    "Never query for all the columns from a specific table, only ask for the relevant columns given the question."
    "If you get an error while executing a query, rewrite the query and try again."
    "If you get an empty result set, you should try to rewrite the query to get a non-empty result set."
    "NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information."
    "If you have enough information to answer the input question, simply output the final answer to the user."    
    "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.",
    backstory="Specializing in data analysis, this agent uses generated SQL queries to answer input questions.",
    allow_delegation=False,
    tools = [list_tables_tool, get_schema_tool, execute_sql_tool]
)

# Create the task for the agent
write_sql_query = Task(
    description = "In order to answer the question {question}, write a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the {question}.",
    expected_output="Short answer to {question} including only a numerical value or a string",
    agent=senior_data_analyst
)

# Set up the crew
crew = Crew(
    agents=[senior_data_analyst],
    tasks=[write_sql_query],
    process=Process.sequential,
    verbose=True,
    memory=False,
    manager_llm=llm,
    output_log_file="crew.log",
)

def kickoff_crew(
    logger: logging.Logger,
    llm: Optional[ChatGroq] = None,
    metrics: Optional[MetricsCollector] = None
) -> List[tuple[str,str]]:
    """
    This function initiates a sequence of tasks using a 'Crew' object to process a series of questions.
    It leverages a provided LLM to generate responses for each question,
    extracts numerical values from those responses, and stores the results for further use.

    The main tasks involved are:
    1. Validating that an LLM is provided.
    2. Iterating over a dictionary of questions, invoking a crew to process each question.
    3. Extracting numerical answers from the crew's responses.
    4. Storing the questions and their corresponding numerical answers in a results list.
    5. Logging the successful generation of results.

    Returns:
        results (list): A list of tuples, each containing a question and its corresponding numerical answer.
    """

    # Check if the LLM is provided; if not, raise an error.
    if llm is None:
        raise ValueError("LLM is not provided")

    # Initialize an empty list to store the results of the query processing
    results = []

    # Create a 'Crew' instance that manages the agents, tasks, and processes involved in the workflow
    crew = Crew(
        agents=[senior_data_analyst],
        tasks=[write_sql_query],
        process=Process.sequential, 
        verbose=False,  # Enable verbose logging for detailed information during execution
        memory=False,  # Disable memory, meaning each task will not retain data between executions
        manager_llm=llm,  # The LLM that will manage the crew's operations
        output_log_file="crew.log",  # The log file to capture the crew's activity and outputs
    )


    # Loop over the 'questions' dictionary, where each key is a question and each value is the corresponding question content
    for key, value in questions.items():       
        start_time = time.time()

        # Prepare the input for the 'crew' by creating a dictionary with the current question
        inputs = {"question": value}
        
        # Use the 'crew' to process the question and get a response (this is where the LLM is invoked)
        content = crew.kickoff(inputs=inputs)

        # Extract the numerical answer from the content returned by the crew's execution
        numerical_answer = extract_numerical_value(str(content))

        # Append the question and its corresponding numerical answer to the results list
        results.append(
            (
                value,  # The original question (value from the 'questions' dictionary)
                numerical_answer  # The extracted numerical answer
            )
        )
                        
        api_latency = time.time() - start_time
        metrics.increment_api_calls()
        metrics.add_api_latency(api_latency)

    # After the loop completes, log a debug message indicating that the results have been generated
    logger.debug("Results generated")

    # Return the list of results, where each entry contains a question and its corresponding numerical answer
    return results

# Run the benchmark process with iterations
def run_benchmark(config: Dict[str, Any], llm: ChatGroq, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Run the benchmark process with iterations"""
    logger.debug("Starting benchmark run...")

    # Get number of iterations from config
    num_iterations = config['benchmarks']['iterations']
    logger.info(f"Running benchmark for {num_iterations} iterations")
    
    # Set up path for logging and metrics    
    # Get the current directory where the script is running
    current_dir = Path(__file__).parent

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

        results = kickoff_crew(logger, llm, metrics)
        
        save_results_to_csv(results, "results", iteration, log_dir, FRAMEWORK)

        logger.info(f"Completed iteration {iteration + 1}")
        
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