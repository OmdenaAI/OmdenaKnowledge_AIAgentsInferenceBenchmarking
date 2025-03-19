import unittest
import logging
from frameworks.crewai_benchmark import benchmark_crewai, get_crewai_framework_name
from frameworks.langchain_benchmark import benchmark_langchain, get_langchain_framework_name
from frameworks.langgraph_benchmark import benchmark_langgraph, get_langgraph_framework_name
from frameworks.swarm_benchmark import benchmark_swarm, get_swarm_framework_name
from frameworks.autogen_benchmark import benchmark_autogen, get_autogen_framework_name
from scripts.visualize_results import visualize_results
import os
from utils.consolidator import consolidate_framework_results, consolidate_and_generate_leaderboard
from utils.config_loader import load_config

def check_results_file():
    """Helper function to check if results file exists."""
    return os.path.exists("results/public_leaderboard.csv")

class TestBenchmarkingFrameworks(unittest.TestCase):
    """Unit tests for AI benchmarking frameworks."""
    
    def setUp(self):
        """Set up logging for test tracking."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.framework_config = load_config("config/settings.yaml").get('frameworks', [])
    
    def test_all_benchmarks(self):
        """Run all enabled benchmarks and then consolidate results."""
        frameworks_to_run = []

        # Check if each framework is enabled and add it to the list
        if any(f['name'] == get_crewai_framework_name() and f['enabled'] for f in self.framework_config):
            frameworks_to_run.append(benchmark_crewai)

        if any(f['name'] == get_autogen_framework_name() and f['enabled'] for f in self.framework_config):
            frameworks_to_run.append(benchmark_autogen)

        if any(f['name'] == get_langchain_framework_name() and f['enabled'] for f in self.framework_config):
            frameworks_to_run.append(benchmark_langchain)

        if any(f['name'] == get_langgraph_framework_name() and f['enabled'] for f in self.framework_config):
            frameworks_to_run.append(benchmark_langgraph)

        if any(f['name'] == get_swarm_framework_name() and f['enabled'] for f in self.framework_config):
            frameworks_to_run.append(benchmark_swarm)

        # Run all the selected benchmarks
        for benchmark_function in frameworks_to_run:
            try:
                benchmark_function()
            except Exception as e:
                self.fail(f"{benchmark_function.__name__} benchmark failed: {e}")
        
        # Consolidate and generate leaderboard only after all benchmarks are run
        consolidate_framework_results()
        consolidate_and_generate_leaderboard()

        self.assertTrue(check_results_file(), "Benchmarks did not generate results.")

    def test_visualization(self):
        """Test visualization function does not fail."""
        try:
            visualize_results(show_plots=True)
        except Exception as e:
            self.fail(f"Visualization failed: {e}")

if __name__ == "__main__":
    unittest.main()
