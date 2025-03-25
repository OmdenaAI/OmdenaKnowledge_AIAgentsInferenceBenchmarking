import yaml
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# Define the path relative to the package
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config():
    """Loads the configuration from config.yaml"""
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
    
    # Add environment variables
    config["api_key"] = os.getenv("API_KEY")
    return config

def get_config():
    """Returns the loaded configuration"""
    return load_config()