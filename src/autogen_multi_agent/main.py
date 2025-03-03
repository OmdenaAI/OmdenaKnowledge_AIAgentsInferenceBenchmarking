from simple_agent_common.utils import setup_logging, load_config, load_env_vars
from orchestrator import MultiAgentOrchestrator
from simple_agent_common.multiagent import BenchmarkRunner
import argparse
def main():
    parser = argparse.ArgumentParser(description='Run the benchmark process')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the yaml config file')
    args = parser.parse_args()
    
    config = load_config(required_paths = ['env', 'metrics', 'input_dir'], 
                        required_benchmark = ['iterations', 'total_questions'],
                        config_location=args.config)
    
    load_env_vars(config)
    logger = setup_logging(framework_name="autogen_multi_agent")

    orchestrator = MultiAgentOrchestrator(config, logger)

    benchmark_runner = BenchmarkRunner("autogen_multi_agent", orchestrator, config, logger)
    benchmark_runner.run()

if __name__ == "__main__":
    main()