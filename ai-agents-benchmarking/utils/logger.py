import logging

import os

def setup_logger(log_file="logs/benchmark.log", level=logging.INFO):
    if not os.path.exists("logs"):
        os.makedirs("logs")  # Create logs directory if missing
    
    logger = logging.getLogger("BenchmarkLogger")
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

