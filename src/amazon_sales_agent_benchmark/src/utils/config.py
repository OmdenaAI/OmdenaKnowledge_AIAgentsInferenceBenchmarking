import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("amazon_query_processor")

class ConfigLoader:
    """Loads and validates configuration."""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dict: Configuration dictionary
        """
        try:
            logger.info(f"Loading config from {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            ConfigLoader._validate_config(config)
            logger.info("Config loaded and validated successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    @staticmethod
    def _validate_config(config: Dict):
        """Validate configuration structure."""
        required_sections = ['model', 'logging', 'benchmarks', 'paths']
        missing = [s for s in required_sections if s not in config]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")

        # Validate model config
        if 'provider' not in config['model']:
            raise ValueError("Model provider not specified in config")

        # Validate paths
        for path_key in ['data_dir', 'amazon_data', 'questions_data']:
            if path_key not in config['paths']:
                raise ValueError(f"Missing required path: {path_key}")

        # Add validation for new config parameters
        if 'model' in config:
            required_model_params = [
                'name', 'temperature', 'max_new_tokens',
                'inference_endpoint', 'retry_config'
            ] 