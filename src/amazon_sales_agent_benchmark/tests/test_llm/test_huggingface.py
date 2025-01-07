import pytest
from unittest.mock import patch, MagicMock

# Importing HuggingFaceProvider directly avoids circular import during mocking
from src.llm.huggingface_provider import HuggingFaceProvider

@pytest.fixture
def logger():
    """Mocked logger fixture for HuggingFaceProvider."""
    mock_logger = MagicMock()
    return mock_logger

@pytest.fixture
def provider(logger):
    """Fixture for the HuggingFaceProvider with mocked Hugging Face libraries."""
    # Localize patches to avoid circular imports
    with patch("src.llm.huggingface_provider.InferenceClient") as MockInferenceClient, \
         patch("src.llm.huggingface_provider.AutoTokenizer.from_pretrained") as MockAutoTokenizer, \
         patch("src.llm.huggingface_provider.HuggingFaceEndpoint") as MockHuggingFaceEndpoint:

        # Mock the InferenceClient
        mock_client = MockInferenceClient.return_value
        mock_client.text_generation.side_effect = lambda *args, **kwargs: "Mocked response"
        mock_client.feature_extraction.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        # Mock the AutoTokenizer
        mock_tokenizer = MockAutoTokenizer.return_value
        mock_tokenizer.encode.return_value = list(range(10))

        # Mock the HuggingFaceEndpoint
        mock_endpoint = MockHuggingFaceEndpoint.return_value
        mock_endpoint.max_new_tokens = 50
        mock_endpoint.temperature = 0.7
        mock_endpoint.top_p = 0.9

        return HuggingFaceProvider(
            logger=logger,
            model_name="gpt2",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=50,
            api_key="test_api_key",
            timeout=300,
            max_retries=3,
            min_wait=4,
            max_wait=60
        )

def test_huggingface_initialization(provider):
    """Test HuggingFace provider initialization."""
    assert provider.model_name == "gpt2"
    assert provider.langchain.max_new_tokens == 50
    assert provider.langchain.temperature == 0.7
    assert provider.langchain.top_p == 0.9

def test_huggingface_inference(provider):
    """Test inference with a mocked Hugging Face model."""
    prompt = "What is artificial intelligence?"
    response = provider.infer(prompt)
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"

def test_huggingface_token_count(provider):
    """Test token counting with a mocked Hugging Face tokenizer."""
    text = "Artificial intelligence is a field of computer science."
    count = provider.get_token_count(text)
    assert isinstance(count, int), "Token count should be an integer"
    assert count == 10, "Token count should match the mocked value"

def test_huggingface_embedding_generation(provider):
    """Test embedding generation with a mocked response."""
    texts = ["Artificial intelligence", "Machine learning"]
    embeddings = provider.get_embeddings(texts)
    assert len(embeddings) == len(texts), "Number of embeddings should match number of input texts"
    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "Embeddings should match the mocked values"

def test_huggingface_tokenizer_fallback(provider):
    """Test token counting fallback when tokenizer is unavailable."""
    provider.tokenizer = None  # Simulate tokenizer unavailability
    text = "Artificial intelligence is a field of computer science."
    count = provider.get_token_count(text)
    assert isinstance(count, int), "Token count should be an integer"
    assert count == len(text.split()), "Token count should match the number of words in text"

def test_logger_interception(logger):
    """Test if logger methods like info and error are intercepted."""
    logger.info("This is an info message")
    logger.error("This is an error message")

    logger.info.assert_called_with("This is an info message")
    logger.error.assert_called_with("This is an error message")
