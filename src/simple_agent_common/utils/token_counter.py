from typing import Optional, List, Dict
from transformers import AutoTokenizer

class TokenCounter:
    def __init__(self):
        # Use LLaMA tokenizer since we're using llama-3.1-70b-versatile
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def count_messages(self, messages: list) -> dict:
        """Count tokens in a list of messages"""
        prompt_tokens = 0
        completion_tokens = 0
        
        for message in messages:
            if message.get('role') == 'assistant':
                completion_tokens += self.count_tokens(message.get('content', ''))
            else:
                prompt_tokens += self.count_tokens(message.get('content', ''))
                
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }