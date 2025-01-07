import pytest
from unittest.mock import mock_open, patch
from src.utils.config import ConfigLoader

@pytest.fixture
def valid_config():
    """Valid configuration dictionary."""
    return {
        "model": {
            "provider": "huggingface",
            "name": "test-model"
        },
        "logging": {
            "level": "INFO",
            "format": "test-format",
            "file": "test.log"
        },
        "benchmarks": {
            "iterations": 1
        },
        "metrics": ["bert_score", "latency"],
        "paths": {
            "data_dir": "data",
            "amazon_data": "amazon.csv",
            "questions_data": "questions.jsonl"
        }
    }

def test_load_valid_config(valid_config):
    """Test loading valid configuration."""
    mock_yaml = mock_open(read_data="""
        model:
          provider: huggingface
          name: test-model
        logging:
          level: INFO
          format: test-format
          file: test.log
        benchmarks:
          iterations: 1
        metrics:
          - bert_score
          - latency
        paths:
          data_dir: data
          amazon_data: amazon.csv
          questions_data: questions.jsonl
    """)
    
    with patch('builtins.open', mock_yaml):
        config = ConfigLoader.load_config("test_config.yaml")
        assert config["model"]["provider"] == "huggingface"
        assert config["paths"]["data_dir"] == "data"

def test_missing_required_section():
    """Test error on missing required section."""
    invalid_config = """
        model:
          provider: huggingface
        logging:
          level: INFO
    """
    with patch('builtins.open', mock_open(read_data=invalid_config)):
        with pytest.raises(ValueError, match="Missing required config sections"):
            ConfigLoader.load_config("test_config.yaml")

def test_missing_model_provider():
    """Test error on missing model provider."""
    invalid_config = """
        model:
          name: test-model
        logging:
          level: INFO
        benchmarks:
          iterations: 1
        metrics:
          - bert_score
        paths:
          data_dir: data
          amazon_data: amazon.csv
          questions_data: questions.jsonl
    """
    with patch('builtins.open', mock_open(read_data=invalid_config)):
        with pytest.raises(ValueError, match="Model provider not specified"):
            ConfigLoader.load_config("test_config.yaml")

def test_missing_required_path():
    """Test error on missing required path."""
    invalid_config = """
        model:
          provider: huggingface
        logging:
          level: INFO
        benchmarks:
          iterations: 1
        metrics:
          - bert_score
        paths:
          data_dir: data
    """
    with patch('builtins.open', mock_open(read_data=invalid_config)):
        with pytest.raises(ValueError, match="Missing required path"):
            ConfigLoader.load_config("test_config.yaml") 