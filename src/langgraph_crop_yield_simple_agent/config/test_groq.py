import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from pathlib import Path

def load_env_vars(env_path: str = Path('~/src/python/.env').expanduser()) -> None:
    """Load environment variables from .env file specified in config"""
    print(env_path)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found at {env_path}")
    
    # Load environment variables from .env file
    load_dotenv(env_path)
    
    # Print the first few characters of the API key for verification
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"GROQ API Key loaded...")
    
    # Validate required environment variables
    required_vars = ['GROQ_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}") 
    
def test_groq_chat():
    # Check for GROQ API key
    if "GROQ_API_KEY" not in os.environ:
        raise ValueError("GROQ_API_KEY environment variable must be set")

    groq_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=groq_key,
        temperature=0.01,
        max_tokens=1000
    )

    # Create messages
    messages = [
        SystemMessage(content="Be a helpful assistant and answer questions in a concise and friendly manner"),
        HumanMessage(content="What is the capital of Portugal?")
    ]

    # Get response
    try:
        response = llm.invoke(messages)
        print("\nGroq Response:")
        print("-------------")
        print(response.content)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    load_env_vars()
    test_groq_chat()