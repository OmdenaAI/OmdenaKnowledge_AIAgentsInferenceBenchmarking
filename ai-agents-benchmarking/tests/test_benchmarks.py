# tests/test_benchmarks.py

import unittest
import logging
from frameworks.crewai_benchmark import benchmark_crewai
from frameworks.langchain_benchmark import benchmark_langchain
from frameworks.langgraph_benchmark import benchmark_langgraph
from frameworks.swarm_benchmark import benchmark_swarm
from frameworks.autogen_benchmark import benchmark_autogen
from scripts.run_benchmark import visualize_results
import os

def setUp(self):
    """Set up test environment."""
    self.results_file = "results/public_leaderboard.csv"
    self.backup_file = f"{self.results_file}.bak"
    # Backup existing results
    if os.path.exists(self.results_file):
        os.rename(self.results_file, self.backup_file)
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

def tearDown(self):
    """Clean up test environment."""
    # Remove test results
    if os.path.exists(self.results_file):
        os.remove(self.results_file)
    # Restore backup if it exists
    if os.path.exists(self.backup_file):
        os.rename(self.backup_file, self.results_file)
        
def check_results_file():
    """Helper function to check if results file exists."""
    return os.path.exists("results/public_leaderboard.csv")

class TestBenchmarkingFrameworks(unittest.TestCase):
    """Unit tests for AI benchmarking frameworks."""
    
    def setUp(self):
        """Set up logging for test tracking."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    def test_crewai_benchmark(self):
        """Test CrewAI benchmarking execution."""
        try:
            benchmark_crewai()
            self.assertTrue(check_results_file(), "CrewAI benchmark did not generate results.")
        except Exception as e:
            self.fail(f"CrewAI benchmark failed: {e}")
    
    # def test_langchain_benchmark(self):
    #     """Test LangChain benchmarking execution."""
    #     try:
    #         benchmark_langchain()
    #         self.assertTrue(check_results_file(), "LangChain benchmark did not generate results.")
    #     except Exception as e:
    #         self.fail(f"LangChain benchmark failed: {e}")
    
    # def test_langgraph_benchmark(self):
    #     """Test LangGraph benchmarking execution."""
    #     try:
    #         benchmark_langgraph()
    #         self.assertTrue(check_results_file(), "LangGraph benchmark did not generate results.")
    #     except Exception as e:
    #         self.fail(f"LangGraph benchmark failed: {e}")
    
    # def test_swarm_benchmark(self):
    #     """Test Swarm benchmarking execution."""
    #     try:
    #         benchmark_swarm()
    #         self.assertTrue(check_results_file(), "Swarm benchmark did not generate results.")
    #     except Exception as e:
    #         self.fail(f"Swarm benchmark failed: {e}")
    
    # def test_autogen_benchmark(self):
    #     """Test AutoGen benchmarking execution."""
    #     try:
    #         benchmark_autogen()
    #         self.assertTrue(check_results_file(), "AutoGen benchmark did not generate results.")
    #     except Exception as e:
    #         self.fail(f"AutoGen benchmark failed: {e}")
    
    def test_visualization(self):
        """Test visualization function does not fail."""
        try:
            visualize_results()
        except Exception as e:
            self.fail(f"Visualization failed: {e}")

if __name__ == "__main__":
    unittest.main()