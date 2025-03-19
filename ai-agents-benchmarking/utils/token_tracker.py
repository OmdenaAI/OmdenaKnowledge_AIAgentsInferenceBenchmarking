import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Mapping of Ollama model names to Hugging Face model names
OLLAMA_TO_HF_MODEL_MAPPING = {
    "llama3": "hf-internal-testing/llama-tokenizer",  
    "mistral": "meta-mistral/Mistral-2-7b-hf", 
    "gemma": "google/gemma-7b-it", 
}

class TokenTracker:
    """Tracks token usage for a given model using the Hugging Face transformers library.

    This class provides a more flexible and comprehensive way to handle tokenization
    for a wide range of models, including those not directly supported by tiktoken.
    """

    def __init__(self, model_name="gpt-4", api_type="openai"):
        """
        Initialize the TokenTracker.

        Args:
            model_name (str): The name of the model for which to track tokens.
        """
        self.model_name = model_name
        self.total_tokens = 0
        
        # Translate Ollama model name to Hugging Face model name if needed
        if api_type == "ollama":
            hf_model_name = OLLAMA_TO_HF_MODEL_MAPPING.get(model_name, model_name)
        else:
            hf_model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            logger.info(f"Using tokenizer for model: {hf_model_name}")
        except Exception as e:
            logger.warning(
                f"Model '{hf_model_name}' not found in Hugging Face's model list. "
                "Using 'gpt2' as fallback. Token counts will be approximations. Error: {e}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def count_tokens(self, text):
        """
        Count the number of tokens in a given text.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        tokens = self.tokenizer.encode(text)
        num_tokens = len(tokens)
        self.total_tokens += num_tokens
        # logger.info(f"Tokens in current request: {num_tokens}")
        return num_tokens

    def get_total_tokens(self):
        """
        Get the total number of tokens used.

        Returns:
            int: The total number of tokens used.
        """
        return self.total_tokens
