# Import libraries
import os
import yaml
import time
import getpass
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
from langchain import hub
from data.questions import queries  # Import the dictionary of queries containing the questions and expected results
from utils.metrics_collector import MetricsCollector
from utils.logging_config import setup_logging
from utils.helper_functions import get_llm, extract_questions, extract_numerical_value, save_results_to_csv, compare_answers
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph.message import AnyMessage, add_messages

# Set the framework that will be benchmarked
FRAMEWORK = "LangGraph"

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

# Connect to database
db = SQLDatabase.from_uri("sqlite:///data/amazon_cleaned.db")

# Extract the questions from the queries dictionary
questions = extract_questions(queries)

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Creates a ToolNode with a fallback mechanism to handle errors and surface them to the agent.

    Args:
        tools (list): A list of tools that the ToolNode will interact with.

    Returns:
        RunnableWithFallbacks: A ToolNode with a fallback that handles errors and returns an appropriate response.
    """
    # Create a ToolNode using the provided list of tools
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def handle_tool_error(state) -> dict:
    """
    Handles errors that occur during the execution of a tool. It surfaces the error to the agent
    by generating a message with the error details.

    Args:
        state: The current state of the system, which contains error information.

    Returns:
        dict: A dictionary containing error messages to be returned to the agent.
    """
    # Retrieve the error from the state
    error = state.get("error")

    # Retrieve the tool calls made in the last message in the state
    tool_calls = state["messages"][-1].tool_calls

    # Return a dictionary with error messages formatted for each tool call
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],  # Include the tool call ID for reference
            )
            for tc in tool_calls  # Create an error message for each tool call
        ]
    }

# Define tools that the agent will use to interact with the database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Tool to fetch the available tables from the database
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")

# Tool to fetch the DDL for a table
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

@tool
def db_query_tool(query: str) -> str:
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

query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check = query_check_prompt | ChatGroq(model = config['model']['name']).bind_tools(
    [db_query_tool], tool_choice="required"
)

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a new graph
workflow = StateGraph(State)

# Add a node for the first tool call
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}

# Add node for the first tool call
workflow.add_node("first_tool_call", first_tool_call)

# Add nodes for the first two tools
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

# Add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = ChatGroq(model=config['model']['name'], temperature=config['model']['temperature']).bind_tools(
    [get_schema_tool]
)
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)

# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""

    final_answer: str = Field(..., description="The final answer to the user")

# Add a node for a model to generate a query based on the question and schema
query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

Output the SQL query that answers the input question without a tool call.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

If you get an error while executing a query, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatGroq(model=config['model']['name'], temperature=config['model']['temperature']).bind_tools(
    [SubmitFinalAnswer]
)

def query_gen_node(state: State):
    """
    Generates a query based on the given state and checks for tool call errors.

    Args:
        state (State): The current state of the graph that will be used to generate a query.

    Returns:
        dict: A dictionary containing the generated message and any tool call error messages (if applicable).
    """
    # Generate a message using the query generator, which likely involves calling an external model
    # like a language model to create a query based on the current state.
    message = query_gen.invoke(state)

    # Initialize an empty list to store any error messages related to tool calls
    tool_messages = []

    # Check if there are any tool calls generated by the message
    if message.tool_calls:
        # Iterate through the tool calls to check if they are correct
        for tc in message.tool_calls:
            # If the tool called is not 'SubmitFinalAnswer', it's considered an error
            if tc["name"] != "SubmitFinalAnswer":
                # Add an error message to the tool_messages list indicating the wrong tool was called
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        # If no tool calls are made, leave the tool_messages list empty
        tool_messages = []

    # Return a dictionary containing the generated message and any error tool messages
    # The 'messages' key will hold both the original message and any error messages from wrong tool calls
    return {"messages": [message] + tool_messages}

# Add a node for the model to generate the query
workflow.add_node("query_gen", query_gen_node)

# Add a node for the model to check the query before executing it
workflow.add_node("correct_query", model_check_query)

# Add node for executing the query
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    """
    Determines whether to continue the workflow or end it based on the last message in the state.

    Args:
        state (State): The current state of the system, which contains the list of messages.

    Returns:
        str: A string indicating whether to continue the workflow with 'correct_query',
             'query_gen', or end the workflow with 'END'.
    """
    # Retrieve the list of messages from the state
    messages = state["messages"]
    
    # Get the last message in the list of messages
    last_message = messages[-1]
    
    # If the last message contains a tool call (indicating completion), end the workflow
    if getattr(last_message, "tool_calls", None):
        return END
    
    # If the last message starts with "Error:", suggest restarting the query generation step
    if last_message.content.startswith("Error:"):
        return "query_gen"
    
    # Otherwise, continue the workflow to the 'correct_query' step
    else:
        return "correct_query"
    
# Specify the edges between the nodes
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges(
    "query_gen",
    should_continue,
)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")

# Compile the workflow into a runnable
app = workflow.compile()

# Create initial state for the workflow
def create_initial_state() -> State:
    """Create initial state for the workflow.
    
    Returns:
        State: Initial workflow state
    """
    return State(
        messages=[]
    )

# Build the graph
def build_graph(
    logger: logging.Logger,
    llm: Optional[ChatGroq] = None
) -> StateGraph:
    
    """Build the complete agent workflow graph.
    Args:
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
    app = workflow.compile()
    
    # Loop over the queries in the dictionary with questions
    for key, value in questions.items():
        print(f"Processing question: {key, value}")
        
        # Execute the graph for the current question        
        messages = app.invoke({"messages": [("user", value)]})

        # Extract the final answer from the last message in JSON string format
        json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]

        # Extract the numerical value from the answer
        numerical_answer = extract_numerical_value(json_str)

        # Append the results to the list
        results.append(
                    (
                        value,  # The question value
                        numerical_answer # The extracted numerical answer
                    )
                )
        for event in app.stream({"messages": [("user", value)]}):
            print(event)
                    
    # After looping, print the results to verify
    logger.debug("Results generated")

    # Create final state
    final_state = StateGraph(State)

    return results, initial_state, final_state

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
        start_time = time.time()

        results, initial_state, final_state = build_graph(logger, llm)
        api_latency = time.time() - start_time

        metrics.increment_api_calls()
        metrics.add_api_latency(api_latency)
        
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