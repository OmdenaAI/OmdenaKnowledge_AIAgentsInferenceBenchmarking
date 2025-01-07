import argparse
from pathlib import Path
from dotenv import load_dotenv
from pathlib import Path
from src.utils.config import ConfigLoader
from src.utils.logging_config import setup_logging
from src.utils.metrics_collector import MetricsCollector
from src.data.data_loader import DataLoader
from src.llm.huggingface_provider import HuggingFaceProvider
from src.agents.langchain_agent import LangChainAgent
from src.agents.langgraph_agent import LangGraphAgent
from src.prompts.templates import PromptTemplates
import time

def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(description='Amazon Query Processing System')
    parser.add_argument('--agents', type=str, nargs='+', required=True,
                       choices=['langchain', 'crewai', 'autogen', 'langgraph', 'all'],
                       help='Agent framework(s) to benchmark')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    return parser

def get_agent_class(agent_name: str):
    """Get agent class by name."""
    agents = {
        'langchain': LangChainAgent,
        'langgraph': LangGraphAgent
    }
    return agents.get(agent_name)

def load_env_variables(logger):
    env_path = Path.home() / 'src' / 'python' / '.env'
    if not env_path.exists():
        logger.error(f"Could not find .env file at {env_path}")
        raise FileNotFoundError(f".env file not found at {env_path}")
            
    logger.info(f"Loading .env file from: {env_path}")
    load_dotenv(dotenv_path=env_path, override=True)  

def main():
    """Main execution function."""
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_config(args.config)

    # Setup logging
    logger = setup_logging(
        log_file=config['logging']['file'],
        log_level=config['logging']['level'],
        log_format=config['logging']['format']
    )

    try:
        load_env_variables(logger)
        
        # Initialize metrics collector
        metrics = MetricsCollector(args.output_dir)

        # Extract model config
        model_config = config['model'].copy()
        
        # Remove parameters we're passing explicitly
        explicit_params = ['name', 'embedding_model', 'temperature', 'max_new_tokens', 
                         'top_p', 'retry_config']
        for param in explicit_params:
            model_config.pop(param, None)
        
        # Initialize LLM provider
        llm = HuggingFaceProvider(
            logger=logger,
            model_name=config['model']['name'],
            embedding_model=config['model']['embedding_model'],
            temperature=config['model']['temperature'],
            max_new_tokens=config['model']['max_new_tokens'],
            top_p=config['model']['top_p'],
            max_retries=config['model']['retry_config']['max_retries'],
            min_wait=config['model']['retry_config']['min_wait'],
            max_wait=config['model']['retry_config']['max_wait'],
            **model_config  # Pass remaining config parameters
        )

        # Initialize data loader
        data_loader = DataLoader(
            collection_name="amazon_products",
            embedding_function=llm.get_embeddings_function(),
            logger=logger,
            llm=llm,
            config=config,
            persist_directory=Path(args.output_dir) / "chromadb"
        )

        # Load questions
        questions_path = Path(config['paths']['data_dir']) / config['paths']['questions_data']
        questions = data_loader.load_questions(questions_path)

        # If necessary store updated sales data
        data_path = Path(config['paths']['data_dir']) / config['paths']['amazon_data']
        df = data_loader.load_csv(data_path)

        # Only reload data if it has changed
        if (data_loader.needs_processing(data_path)):
            data_loader.embed_and_store(df)

        # Get prompt templates
        templates = PromptTemplates.get_all_templates()

        # Initialize agents with templates
        agents = {
            'langchain': LangChainAgent(
                llm=llm,
                templates=templates,
                logger=logger
            ),
            'langgraph': LangGraphAgent(
                llm=llm,
                templates=templates,
                logger=logger
            )
        }

        # Run benchmarks for each agent
        agent_names = ['langchain', 'langgraph'] if 'all' in args.agents else args.agents
        
        for agent_name in agent_names:
            logger.info(f"Running benchmark with {agent_name}")
            
            agent = agents[agent_name]
            agent.initialize()

            for i in range(config['benchmarks']['iterations']):
                metrics.start_iteration(agent_name, i)
                
                for query in questions:
                    chunks = data_loader.retrieve_chunks(
                        query_text=query['question'],
                        category=query['category']
                    )
                    
                    responses = []
                    for chunk in chunks:
                        start_time = time.time()
                        response = agent.execute_query(query['question'], query['category'], chunk)
                        api_latency = time.time() - start_time
                        
                        responses.append(response)
                        metrics.increment_api_calls()
                        metrics.add_api_latency(api_latency)
                    
                    if (len(responses) > 1):
                        start_time = time.time()
                        final_response = agent.aggregate_results(responses, query['question'])
                        api_latency = time.time() - start_time
                        metrics.increment_api_calls()
                    else:
                        final_response = responses[0]

                    metrics.add_bert_score(final_response, query['expected_answer'])
                
                metrics.end_iteration()
                metrics.save_iteration(agent_name)

        metrics.save_metrics()
        metrics.generate_plots()
        logger.info("Benchmark completed successfully")

    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 