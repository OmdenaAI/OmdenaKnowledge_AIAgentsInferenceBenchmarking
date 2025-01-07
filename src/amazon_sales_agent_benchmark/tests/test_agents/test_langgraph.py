import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.langgraph_agent import LangGraphAgent, AgentState

@pytest.fixture
def mock_logger():
    """Mocked logger fixture for LangGraphAgent."""
    mock_logger = MagicMock()
    return mock_logger

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()
    mock.predict.return_value = "Test response"
    return mock

@pytest.fixture
def sample_templates():
    """Sample prompt templates for testing."""
    return {
        "price_template": "Context: {context}\nQuestion: {question}\nAnswer about price:",
        "rating_template": "Context: {context}\nQuestion: {question}\nAnswer about rating:",
        "judge_template": "Choose best answer from:\n{responses}"
    }

@pytest.fixture
def agent(mock_llm, sample_templates, mock_logger):
    """Create LangGraph agent instance."""
    with patch('src.agents.langgraph_agent.StateGraph'):
        return LangGraphAgent(mock_llm, sample_templates, logger=mock_logger)

def mock_initialize(agent):
    """Mock initialization for LangGraphAgent."""
    agent.initialized = True
    agent.category_mapping = {"price-related": "price_template", "rating-related": "rating_template"}
    agent.chains = {"price_template": Mock(), "rating_template": Mock()}
    agent.workflow = Mock()

def test_langgraph_initialization(mock_llm, sample_templates, mock_logger):
    """Test LangGraph agent initialization."""
    with patch('src.agents.langgraph_agent.StateGraph'):
        agent = LangGraphAgent(mock_llm, sample_templates, logger=mock_logger)
        assert not agent.initialized

def test_initialize(agent):
    """Test initialization of LangGraph workflow."""
    with patch('src.agents.langgraph_agent.StateGraph') as MockStateGraph:
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_graph
        MockStateGraph.return_value = mock_graph
        agent.initialize()
        assert agent.initialized, "Agent should be initialized successfully"
        mock_graph.compile.assert_called_once()

def test_execute_query_without_initialization(agent):
    """Test error when executing query without initialization."""
    with pytest.raises(RuntimeError, match="Agent not initialized"):
        agent.execute_query("test query", "test category", "test context")

def test_execute_query(agent):
    """Test query execution."""
    mock_initialize(agent)  # Use mock initialization
    agent.workflow.invoke.return_value = {
        "response": "Test response",  # Update the key to match the method
        "error": ""
    }
    response = agent.execute_query("test query", "price-related", "test context")
    assert response == "Test response"


def test_execute_query_with_error(agent):
    """Test error handling during query execution."""
    mock_initialize(agent)  # Use mock initialization
    agent.workflow.invoke.return_value = {
        "final_response": "",
        "error": "Test error"
    }
    with pytest.raises(RuntimeError, match="Test error"):
        agent.execute_query("test query", "price-related", "test context")

def test_aggregate_results_empty(agent):
    """Test aggregating empty results."""
    mock_initialize(agent)  # Use mock initialization
    response = agent.aggregate_results([], "Original prompt")
    assert response == "No results to aggregate."

def test_aggregate_results(agent):
    """Test aggregating multiple responses."""
    mock_initialize(agent)  # Use mock initialization

    # Mock the judge_chain and its invoke method
    judge_chain = Mock()
    judge_chain.invoke.return_value = (
        "Best Answer: Response 1\n"  # Removed quotes since they'll be stripped
        "Reason: This is the most relevant response to 'Original prompt'."
    )
    agent.chains["judge_template"] = judge_chain

    # Responses to aggregate
    responses = ["Response 1", "Response 2"]

    # Execute the method
    response = agent.aggregate_results(responses, "Original prompt")

    # Assert the result
    assert "Best Answer: Response 1" in response, "Expected response to contain the best answer"
    assert "Reason:" in response, "Expected the reason to be included in the output"
    assert "Original prompt" in response, "Expected original prompt to appear in the reason"

    # Verify the judge chain was called with correct input
    judge_chain.invoke.assert_called_once()
    call_args = judge_chain.invoke.call_args[0][0]
    assert "Response 1" in call_args["responses"]
    assert "Response 2" in call_args["responses"]
    assert call_args["original_prompt"] == "Original prompt"

def test_aggregate_results_missing_template(agent):
    """Test error when judge template is missing."""
    mock_initialize(agent)  # Use mock initialization
    agent.templates = {}
    with pytest.raises(ValueError, match="Judge template not found"):
        agent.aggregate_results(["Response 1", "Response 2"], "Original prompt")
