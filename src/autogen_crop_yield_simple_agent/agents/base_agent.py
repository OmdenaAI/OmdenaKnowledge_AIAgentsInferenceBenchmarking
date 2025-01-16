import autogen
from typing import Dict, Any, Optional
import logging

def create_llm_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create LLM configuration for AutoGen"""
    return {
        "config_list": [{
            "model": model_config['name'],
            "temperature": model_config['temperature'],
            "max_tokens": model_config['max_tokens']
        }]
    }

class BaseAgentConfig:
    """Base configuration class for agents"""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Validate model configuration
        self._validate_model_config()
        
        # Set up model configuration
        self.model_config = {
            'name': config['model']['name'],
            'temperature': config['model']['temperature'],
            'max_tokens': config['model']['max_tokens']
        }
        
        # Set up benchmark configuration
        self.benchmark_config = {
            'iterations': config['benchmark']['iterations'],
            'random_few_shot': config['benchmark']['random_few_shot'],
            'num_few_shot': config['benchmark']['num_few_shot']
        }
        
        # Set up data paths
        self.data_paths = config['data']['paths']
        
        # Create LLM config
        self.llm_config = create_llm_config(self.model_config)
    
    def _validate_model_config(self) -> None:
        """Validate the model configuration"""
        required_model_fields = ['name', 'temperature', 'max_tokens']
        model_config = self.config.get('model', {})
        
        missing_fields = [field for field in required_model_fields 
                         if field not in model_config]
        
        if missing_fields:
            raise ValueError(f"Missing required model configuration fields: {missing_fields}") 