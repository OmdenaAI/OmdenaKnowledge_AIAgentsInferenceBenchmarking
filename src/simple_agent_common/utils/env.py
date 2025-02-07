from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Dict, Any

def load_env_vars(config: Dict[str, Any]) -> None:
    """Load environment variables from .env file specified in config"""
    env_path = Path(config['data']['paths']['env']).expanduser()
    
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