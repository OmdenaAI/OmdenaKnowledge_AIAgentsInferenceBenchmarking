import pytest
from src.agents.base_agent import BaseAgent
from unittest.mock import MagicMock

class TestBaseAgent:
    """Test suite for BaseAgent class."""

    def test_init(self):
        """Test BaseAgent initialization."""
        llm = MagicMock()
        templates = {"test": "template"}
        logger = MagicMock()
        
        class ConcreteAgent(BaseAgent):
            def initialize(self): pass
            def execute_query(self, query, category, context=None): pass
            def aggregate_results(self, results, original_prompt): pass
        
        agent = ConcreteAgent(llm, templates, logger)
        assert agent.llm == llm
        assert agent.templates == templates
        assert agent.logger == logger
        assert not agent.initialized

    def test_clean_response(self):
        """Test response cleaning with different formats."""
        class ConcreteAgent(BaseAgent):
            def initialize(self): pass
            def execute_query(self, query, category, context=None): pass
            def aggregate_results(self, results, original_prompt): pass
        
        # Create mock templates and logger
        mock_templates = {
            "test_template": "Test template content"
        }
        mock_logger = MagicMock()
        
        agent = ConcreteAgent(MagicMock(), mock_templates, mock_logger)
        
        # Test user/assistant format
        response = "\n<|user|>\nInput\n\n<|assistant|>\nOutput"
        assert agent._clean_response(response) == "Output"
        
        # Test simple format
        response = "Simple response"
        assert agent._clean_response(response) == "Simple response"
        
        # Test empty response
        assert agent._clean_response("") == ""

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(TypeError):
            BaseAgent(MagicMock(), {}, MagicMock()) 