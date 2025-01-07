import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.langchain_agent import LangChainAgent

# Utility function to mock agent for query tests
def setup_mock_agent_with_templates(agent, templates, invoke_mock=True):
    """Mock agent with initialized templates."""
    agent.initialize = Mock()
    agent.initialized = True

    # Mock chains based on templates
    mock_chains = {}
    for template_name in templates:
        mock_chain = Mock()
        if invoke_mock:
            mock_chain.invoke = Mock(return_value=f"Response from {template_name}")
        else:
            mock_chain.run = Mock(return_value=f"Response from {template_name}")
        mock_chains[template_name] = mock_chain
    agent.chains = mock_chains

    # Mock LLM invoke behavior
    agent.llm.invoke = Mock(return_value="Generic response")

    # Mock _build_judge_chain
    mock_judge_chain = Mock()
    mock_judge_chain.invoke = Mock(return_value="Final judged response")
    agent._build_judge_chain = Mock(return_value=mock_judge_chain)


@pytest.fixture
def logger():
    """Mocked logger fixture for HuggingFaceProvider."""
    mock_logger = MagicMock()
    return mock_logger

@pytest.fixture
def agent(logger):
    """Fixture to provide a LangChainAgent instance."""
    return LangChainAgent(llm=Mock(), templates={
        "price_template": "Price: {context} {question}",
        "rating_template": "Rating: {context} {question}",
        "judge_template": "Judge: {context} {question}"
    }, logger=logger)

@patch.object(LangChainAgent, "initialize", return_value=None)
def test_execute_price_query(mock_initialize, agent):
    """Test executing price-related query."""
    setup_mock_agent_with_templates(agent, ["price-related", "rating-related"])
    
    # Execute query
    response = agent.execute_query(
        "What is the price of product X?",
        "price-related",
        "Product X costs $100"
    )
    
    # Assertions
    assert response.strip() == "Response from price-related", \
        "The response should match the mocked chain output"



@patch.object(LangChainAgent, "initialize", return_value=None)
def test_execute_rating_query(mock_initialize, agent):
    """Test executing rating-related query."""
    setup_mock_agent_with_templates(agent, ["price-related", "rating-related"])

    # Execute query
    response = agent.execute_query("What is the rating of product X?", 
                                   "rating-related",
                                   "Product X has 4.5 stars")

    # Assertions
    assert response.strip() == "Response from rating-related", \
        "The response should match the mocked chain output"

@patch.object(LangChainAgent, "initialize", return_value=None)
def test_execute_query_without_initialization(mock_initialize, agent):
    """Test executing query without initialization."""
    agent.initialized = False

    with pytest.raises(RuntimeError, match="Agent not properly initialized"):
        agent.execute_query("What is the price of product X?", "Product X costs $100")

@patch.object(LangChainAgent, "initialize", return_value=None)
def test_execute_query_invalid_template(mock_initialize, agent):
    """Test executing query with an invalid template type."""
    setup_mock_agent_with_templates(agent, ["price-related"])

    with pytest.raises(ValueError, match="No chain found for category: foo-related"):
        agent.execute_query("What is the rating of product X?",
                            "foo-related",
                            "Product X has 4.5 stars")

@patch.object(LangChainAgent, "initialize", return_value=None)
def test_aggregate_results(mock_initialize, agent):
    """Test aggregating multiple responses."""
    # Mock the judge chain
    mock_judge_chain = MagicMock()
    mock_judge_chain.invoke.return_value = "Final judged response"

    # Set up the agent with the mocked judge chain
    agent.chains = {"judge_template": mock_judge_chain}

    # Responses to aggregate
    responses = ["Response 1", "Response 2"]

    # Execute aggregation
    response = agent.aggregate_results(responses, "Original prompt")

    # Assertions
    assert response.strip() == "Final judged response", \
        "The response should match the mocked judge chain output"

    # Validate the calls to the mock
    mock_judge_chain.invoke.assert_called_once_with({
        "responses": "1. Response 1\n2. Response 2",
        "original_prompt": "Original prompt"
    })

@patch.object(LangChainAgent, "initialize", return_value=None)
def test_aggregate_results_empty(mock_initialize, agent):
    """Test aggregating an empty list of responses."""
    setup_mock_agent_with_templates(agent, ["price_template", "rating_template"])

    # Execute aggregation
    responses = []
    response = agent.aggregate_results(responses, "Original prompt")

    # Assertions
    assert response.strip() == "No results to aggregate.", "The response should be empty for no inputs"

def test_init_error(logger):
    """Test initialization error handling."""
    with pytest.raises(Exception):
        LangChainAgent(None, None, logger)

def test_initialize_error_no_templates(agent):
    """Test initialization with no templates."""
    agent.templates = {}
    with pytest.raises(RuntimeError, match="Chain initialization failed: No chains could be initialized from templates. Templates are required"):
        agent.initialize()

def test_initialize_error_invalid_template(agent):
    """Test initialization with invalid template."""
    agent.templates = {"invalid": None}
    with pytest.raises(RuntimeError, match="Chain initialization failed: No template found for category: invalid"):
        agent.initialize()

def test_execute_query_error_handling(agent):
    """Test error handling in execute_query."""
    # Test uninitialized error
    agent.initialized = False
    with pytest.raises(RuntimeError, match="Agent not properly initialized"):
        agent.execute_query("query", "category")
    
    # Test unknown category error
    agent.initialized = True
    with pytest.raises(ValueError, match="No chain found for category"):
        agent.execute_query("query", "unknown_category")
    
    # Test chain execution error
    agent.initialized = True
    agent.category_mapping = {"test": "test_template"}
    agent.chains = {}
    with pytest.raises(ValueError, match="No chain found for category"):
        agent.execute_query("query", "test")