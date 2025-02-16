import warnings
# Disable specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
from orchestrator import MultiAgentOrchestrator
from simple_agent_common.utils import load_env_vars, load_config, setup_logging
from simple_agent_common.multiagent import BenchmarkRunner
from pathlib import Path


def main():
    config = load_config(required_paths = ['env', 'metrics', 'input_dir'], 
                        required_benchmark = ['iterations', 'total_questions'])
    load_env_vars(config)
    logger = setup_logging(framework_name="langgraph_multi_agent")
   
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(config, logger)

    benchmark_runner = BenchmarkRunner("langgraph_multi_agent", orchestrator, config, logger)
    benchmark_runner.run()

if __name__ == "__main__":
    main() 