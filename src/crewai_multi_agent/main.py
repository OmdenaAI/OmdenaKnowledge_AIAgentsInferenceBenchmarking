import argparse
from simple_agent_common.utils import load_env_vars, load_config, setup_logging
from orchestrator import MultiAgentOrchestrator
from simple_agent_common.multiagent import BenchmarkRunner

FRAMEWORK = "crewai"

def main():
    parser = argparse.ArgumentParser(description='Run the benchmark process')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the yaml config file')
    args = parser.parse_args()
    
    config = load_config(required_paths = ['env', 'metrics', 'input_dir'], 
                        required_benchmark = ['iterations', 'total_questions'],
                        config_location=args.config)
    load_env_vars(config)
    logger = setup_logging(framework_name=FRAMEWORK)
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(config, logger)
    
    benchmark_runner = BenchmarkRunner(FRAMEWORK, orchestrator, config, logger)
    benchmark_runner.run()

if __name__ == "__main__":
    main() 
