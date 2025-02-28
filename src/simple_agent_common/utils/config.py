import yaml
from pathlib import Path
from typing import Dict, Any, List

def load_config(required_paths: List[str]  = ['crop_data', 'questions', 'env', 'metrics'],
                required_model: List[str] = ['name', 'temperature', 'max_tokens'],
                required_benchmark: List[str] = ['iterations', 'random_few_shot', 'num_few_shot'], config_location: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_location)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    validate_config(config, required_paths, required_model, required_benchmark)
    return config

def validate_config(config: Dict[str, Any], required_paths: List[str], 
                    required_model: List[str], required_benchmark: List[str]) -> None:
    """Validate configuration structure and required fields"""
    required_sections = ['data', 'model', 'benchmark']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate data paths
    for path in required_paths:
        if path not in config['data']['paths']:
            raise ValueError(f"Missing required data path: {path}")
    
    # Validate model configuration
    for field in required_model:
        if field not in config['model']:
            raise ValueError(f"Missing required model fields: {required_model}")
    
    # Validate benchmark configuration
    for field in required_benchmark:
        if field not in config['benchmark']:
            raise ValueError(f"Missing required benchmark field: {field}") 