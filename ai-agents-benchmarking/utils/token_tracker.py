import tiktoken
import logging

logger = logging.getLogger(__name__)

class TokenTracker:
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name
        self.total_tokens = 0
        self.encoding = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text):
        """Count tokens in a given text."""
        tokens = self.encoding.encode(text)
        token_count = len(tokens)
        self.total_tokens += token_count
        logger.info(f"Tokens in current request: {token_count}")
        return token_count

    def reset(self):
        """Reset the token count."""
        self.total_tokens = 0

    def get_total_tokens(self):
        """Get total tokens used."""
        return self.total_tokens

if __name__ == "__main__":
    tracker = TokenTracker("gpt-4")
    text = "Hello, how are you doing today?"
    print(f"Tokens used: {tracker.count_tokens(text)}")
    print(f"Total tokens so far: {tracker.get_total_tokens()}")
