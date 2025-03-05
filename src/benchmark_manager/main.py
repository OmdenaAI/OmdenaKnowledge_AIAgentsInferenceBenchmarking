import yaml
from pathlib import Path
from typing import List
from benchmark_data import BenchmarkData
from benchmark import Benchmark
from simple_agent_common.utils import setup_logging

def load_config(config_path: str, logger) -> tuple[List[BenchmarkData], str]:
    """
    Load config and create BenchmarkData objects.
    
    Returns:
        tuple: (List of BenchmarkData objects, output directory path)
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create BenchmarkData objects
    benchmark_results = []
    for benchmark in config['benchmarks']:
        logger.info(f"Processing benchmark: {benchmark['name']}")
        benchmark_data = BenchmarkData(
            name=benchmark['name'],
            input_dir=benchmark['input_dir'],
            logger=logger
        )
        benchmark_results.append(benchmark_data)
    
    output_dir = config['data'][0]['output_dir']
    logger.info(f"Found {len(benchmark_results)} benchmarks in config")
    
    return benchmark_results, output_dir

def main():
    logger = setup_logging(framework_name="benchmark_processor")
    logger.info("Starting benchmark processing application")

    try:
        # Load configuration and get benchmark objects
        benchmark_results, output_dir = load_config(config_path="config/config.yaml", logger=logger)
        
        # Process benchmarks and generate visualizations
        logger.info("Creating benchmark processor")
        processor = Benchmark(benchmark_results, output_dir, logger=logger)
        
        logger.info("Processing benchmarks and generating visualizations")
        processor.process()
        
        logger.info("Benchmark processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing benchmarks: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 