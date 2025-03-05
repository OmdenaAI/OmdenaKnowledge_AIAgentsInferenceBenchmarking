import os
from typing import Optional
from datetime import datetime

class BenchmarkData:
    def __init__(self, name: str, input_dir: str, logger, date_to_process: Optional[str] = None):
        self.name = name
        self.input_dir = input_dir
        self.logger = logger
        self.date_to_process = date_to_process

        self.logger.info(f"Initializing benchmark data for {name}")
        if date_to_process:
            self.logger.info(f"Will process data for date: {date_to_process}")
        
        if not os.path.exists(self.input_dir):
            self.logger.error(f"Input directory {self.input_dir} does not exist")
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist") 