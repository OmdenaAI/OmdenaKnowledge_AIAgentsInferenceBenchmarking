import yaml
import os
import logging

logger = logging.getLogger(__name__)

def load_config(config_path="config/settings.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration data.
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"⚠️ settings.yaml not found at {config_path}")

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        logger.debug(f"Loaded configuration from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"⚠️ Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"⚠️ Unexpected error loading config: {e}")
        raise

def get_llm_config(model_name=None, config_path="config/settings.yaml"):
    """
    Extract LLM configuration from settings.yaml.

    Args:
        model_name (str): Optional model name to override the default.
        config_path (str): Path to the YAML config file.

    Returns:
        dict: LLM configuration settings.
    """
    config = load_config(config_path)
    llm_settings = config.get('llm_execution', {})

    return {
        'model_name': model_name or llm_settings.get('model_name', 'default_model'),
        'base_url': llm_settings.get('base_url', ''),
        'api_key': llm_settings.get('api_key', ''),
        'api_type': llm_settings.get('api_type', ''),
        'temperature': llm_settings.get('temperature', 0.7)
    }
