import logging
from pathlib import Path
from typing import Optional

def setup_logging(framework_name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Configure logging for the application"""
    if log_dir:
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{framework_name}_benchmark.log"
    
    # Create logger
    logger = logging.getLogger(f"{framework_name}_benchmark")
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_dir provided
    if log_dir:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 