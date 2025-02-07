import json
import os
import random
import hashlib
from datetime import datetime
from typing import List, Dict, Any

class Dataset:
    def __init__(self, data_dir: str, total_questions: int):
        """
        Initialize Dataset with directory containing JSONL files
        
        Args:
            data_dir: Directory containing JSONL files
            total_questions: Total number of questions to extract across all files
        """
        self.data_dir = data_dir
        self.total_questions = total_questions
        self.seed = self._generate_daily_seed()
        random.seed(self.seed)
        self.jsonl_files = self._get_jsonl_files()
        self.questions = self._load_questions()
        random.shuffle(self.questions)
        
    def _generate_daily_seed(self) -> int:
        """Generate a seed based on the day to ensure different but reproducible sampling per day"""
        today = datetime.now().strftime('%Y-%m-%d')
        hash_object = hashlib.md5(today.encode())
        return int(hash_object.hexdigest(), 16) % (2**32)
    
    def _get_jsonl_files(self) -> List[str]:
        """Get all JSONL files from the data directory"""
        files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.jsonl'):
                files.append(os.path.join(self.data_dir, file))
        return files
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from JSONL files"""
        questions = []
        all_questions = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'r') as f:
                    file_questions = []
                    for line in f:
                        question = json.loads(line)
                        question['source_file'] = filename  # Add source file to each question
                        file_questions.append(question)
                    all_questions.extend(file_questions)
        
        # Randomly sample questions if we have more than needed
        if len(all_questions) > self.total_questions:
            return random.sample(all_questions, self.total_questions)
        return all_questions
    
    def get_questions(self) -> List[Dict[str, Any]]:
        """Return the loaded questions"""
        return self.questions
    
    def save_questions(self, output_dir: str, filename: str):
        """Save the selected questions to the output directory"""
        output_file = os.path.join(output_dir, filename)
        
        with open(output_file, 'w') as f:
            for question in self.questions:
                json.dump(question, f)
                f.write('\n') 