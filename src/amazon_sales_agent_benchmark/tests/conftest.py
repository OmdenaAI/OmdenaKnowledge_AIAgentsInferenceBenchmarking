import pytest
from typing import Dict
import yaml

@pytest.fixture
def sample_config() -> Dict:
    """Sample configuration for testing."""
    return {
        "model": {
            "provider": "huggingface",
            "name": "tiiuae/Falcon3-1B-Instruct",
            "temperature": 0.1,
            "max_length": 3096,
            "max_new_tokens": 1000,
            "top_p": 0.95,
            "retry": {
                "max_attempts": 5,
                "min_wait": 4,
                "max_wait": 60
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "benchmark.log"
        },
        "benchmarks": {
            "iterations": 1,
            "metrics_output": "results/amazon_qa_metrics.json"
        }
    }

@pytest.fixture
def sample_amazon_data():
    """Sample Amazon product data for testing."""
    return [
        {
            "product_id": "B07JW9H4J1",
            "product_name": "Test Product",
            "category": "Electronics",
            "discounted_price": "₹399",
            "actual_price": "₹1,099",
            "rating": 4.2,
            "rating_count": "24,269"
        }
    ]

@pytest.fixture
def sample_qa_data():
    """Sample Q&A data for testing."""
    return [
        {
            "prompt": "What is the discounted price of the product 'Test Product'?",
            "completion": "The discounted price is ₹399.",
            "category": "price-related"
        }
    ] 