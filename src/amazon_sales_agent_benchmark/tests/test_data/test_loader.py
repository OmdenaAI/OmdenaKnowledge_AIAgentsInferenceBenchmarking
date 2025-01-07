import pytest
import pandas as pd
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.data.data_loader import DataLoader
from src.llm.base_llm_provider import BaseLLMProvider
from typing import List, Callable

class MockLLM(BaseLLMProvider):
    """Mock LLM for testing."""
    def __init__(self):
        self.calls = []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.calls.append(('get_embeddings', texts))
        return [[0.0] * 768] * len(texts)

    def infer(self, prompt: str) -> str:
        self.calls.append(('infer', prompt))
        return "Mock response"

    def get_token_count(self, text: str) -> int:
        self.calls.append(('get_token_count', text))
        return len(text) // 4

    def get_embeddings_function(self) -> Callable:
        # Create a class that matches ChromaDB's expected interface
        class MockEmbeddingFunction:
            def __call__(self, input: List[str]) -> List[List[float]]:
                return [[0.0] * 768] * len(input)
        
        return MockEmbeddingFunction()

@pytest.fixture
def sample_data():
    """Sample data fixture with all required columns and formats."""
    return pd.DataFrame({
        'product_id': ['A1', 'A2', 'A3'],
        'product_name': ['Product 1', 'Product 2', 'Product 3'],
        'category': ['Electronics', 'Electronics', 'Computers'],
        'rating': [4.5, 3.8, 4.9],
        'rating_count': ['1,234', '567', '2,345'],
        'discounted_price': ['₹999', '₹1,499', '₹2,999'],
        'actual_price': ['₹1,999', '₹2,999', '₹5,999'],
        'discount_percentage': ['50%', '30%', '40%']
    })

@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client and collection."""
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.create_collection.return_value = mock_collection  # Ensure consistency
    return mock_client, mock_collection


@pytest.fixture
def mock_logger():
    """Mocked logger fixture."""
    return MagicMock()

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'required_columns': [
            'product_id', 'product_name', 'category',
            'rating', 'rating_count', 'discounted_price',
            'actual_price', 'discount_percentage'
        ],
        'model': {
            'max_length': 2048
        }
    }

@pytest.fixture
def loader(tmp_path, config, mock_chromadb, mock_logger):
    """DataLoader fixture with temporary directory and mocked ChromaDB and logger."""
    mock_client, _ = mock_chromadb

    with patch('chromadb.Client', return_value=mock_client):
        return DataLoader(
            collection_name="test_collection",
            embedding_function=MockLLM().get_embeddings_function(),
            logger=mock_logger,
            llm=MockLLM(),
            config=config,
            persist_directory=str(tmp_path)
        )

@pytest.fixture
def validated_sample_data():
    """Sample data fixture with pre-validated/cleaned data."""
    return pd.DataFrame({
        'product_id': ['A1', 'A2', 'A3'],
        'product_name': ['Product 1', 'Product 2', 'Product 3'],
        'category': ['Electronics', 'Electronics', 'Computers'],
        'rating': [4.5, 3.8, 4.9],
        'rating_count': [1234, 567, 2345],  # Already converted to integers
        'discounted_price': [999.0, 1499.0, 2999.0],  # Already converted to float
        'actual_price': [1999.0, 2999.0, 5999.0],  # Already converted to float
        'discount_percentage': [50.0, 30.0, 40.0]  # Already converted to float
    })

class TestDataLoader:

    def test_data_validation(self, loader, sample_data):
        """Test data validation with valid and invalid inputs."""
        # Validate correct data
        loader._validate_data(sample_data)

        # Test invalid 'rating'
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'rating_count'] = None
        with pytest.raises(ValueError, match="Column 'rating_count' contains invalid data"):
            loader._validate_data(invalid_data)

        # Reset data and test invalid 'discounted_price'
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'discounted_price'] = 'invalid'
        with pytest.raises(ValueError):
            loader._validate_data(invalid_data)

    def test_embed_and_store(self, loader, validated_sample_data, mock_chromadb):
        """Test embedding and storing data."""
        mock_client, mock_collection = mock_chromadb
        
        # Set the client and collection
        loader.client = mock_client
        loader.product_collection = mock_collection

        # Mock the LLM to return specific embeddings
        loader.llm.get_embeddings = MagicMock(return_value=[[0.1] * 768] * len(validated_sample_data))

        # Adjust chunk size for batching
        loader.chunk_size = 15

        # Call the method under test
        loader.embed_and_store(validated_sample_data)

        # Verify that `add` was called
        assert mock_collection.add.called, "Collection.add was not called"
        add_calls = mock_collection.add.call_args_list
        assert len(add_calls) > 0, "Expected at least one call to 'add'."

        # Verify the content of the add calls
        first_call = add_calls[0]
        assert 'documents' in first_call.kwargs, "Expected documents in add call"
        assert 'metadatas' in first_call.kwargs, "Expected metadatas in add call"
        assert 'ids' in first_call.kwargs, "Expected ids in add call"

    def test_retrieve_chunks(self, loader, mock_chromadb):
        """Test retrieval of embedded chunks."""
        mock_client, mock_collection = mock_chromadb

        # Mock query response to match expected document structure
        mock_collection.query.return_value = {
            'documents': [["₹999"]]
        }

        # Set both client and product_collection
        loader.client = mock_client
        loader.product_collection = mock_collection

        # Create proper embedding function mock
        class MockEmbeddingFunction:
            def __call__(self, input: List[str]) -> List[List[float]]:
                return [[0.0] * 768] * len(input)
        
        loader._embedding_function = MockEmbeddingFunction()

        # Perform retrieval
        chunks = loader.retrieve_chunks(
            "What is the discounted price of the product 'Product 1'?",
            category="price-related"
        )

        # Validate the results
        assert len(chunks) > 0, "No chunks were retrieved"
        assert '₹999' in chunks[0], f"Expected '₹999' in chunks, got: {chunks[0]}"

        # Verify query was called with expected parameters
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args[1]  # Get kwargs
        assert 'query_texts' in call_args, "Expected query_texts parameter"

    def test_load_questions(self, loader, tmp_path):
        """Test loading questions from a JSONL file."""
        questions_file = tmp_path / "questions.jsonl"
        questions_file.write_text(
            '{"prompt": "What is the price?", "completion": "999", "category": "price-related"}\n'
        )

        questions = loader.load_questions(questions_file)
        assert len(questions) == 1
        assert questions[0]['question'] == "What is the price?"
        assert questions[0]['expected_answer'] == "999"

    @pytest.mark.parametrize("query,category,expected", [
        ("What is the discounted price of 'Product 1'?", "price-related", "₹999"),
        ("What is the rating of 'Product 2'?", "rating-related", "3.8"),
    ])
    def test_query_patterns(self, loader, mock_chromadb, query, category, expected):
        """Test multiple query patterns."""
        mock_client, mock_collection = mock_chromadb
        mock_collection.query.return_value = {
            'documents': [[expected]]
        }

        # Set both client and product_collection
        loader.client = mock_client
        loader.product_collection = mock_collection

        # Create proper embedding function mock
        class MockEmbeddingFunction:
            def __call__(self, input: List[str]) -> List[List[float]]:
                return [[0.0] * 768] * len(input)
        
        loader._embedding_function = MockEmbeddingFunction()

        # Perform retrieval
        chunks = loader.retrieve_chunks(query, category=category)
        
        # Validate results
        assert len(chunks) > 0, "No chunks were retrieved"
        assert expected in chunks[0], f"Expected {expected} in chunks, got: {chunks[0]}"

        # Verify query was called with correct parameters
        mock_collection.query.assert_called()
        call_args = mock_collection.query.call_args[1]
        assert query in call_args.get('query_texts', []), "Query text not found in call arguments"
