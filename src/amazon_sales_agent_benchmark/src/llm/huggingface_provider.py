from typing import List, Optional, Callable
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoTokenizer
import numpy as np
import os
from src.llm.base_llm_provider import BaseLLMProvider
from src.utils.errors import EndpointError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace provider using Inference Endpoints."""
    
    def __init__(
        self,
        logger,
        model_name: str,
        embedding_model: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        api_key: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3,
        min_wait: int = 4,
        max_wait: int = 60,
        **kwargs
    ):
        """Initialize HuggingFace provider."""
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.logger = logger
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait
        
        # Handle API key
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HuggingFace API key not found")
            
        try:
            # Initialize clients with retry decorator
            self.client = InferenceClient(
                token=self.api_key,
                timeout=self.timeout
            )
            
            # Test model availability
            self.client.text_generation(
                "test",
                model=self.model_name,
                max_new_tokens=1
            )
        except HfHubHTTPError as e:
            if "does not exist" in str(e):
                self.logger.error(f"Model {model_name} not found. Please check the model name.")
                raise ValueError(f"Model {model_name} not found on HuggingFace Hub") from e
            raise
            
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer: {str(e)}")
            self.tokenizer = None
        
        # Extract model kwargs from config
        self.model_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['provider', 'inference_endpoint', 'retry_config']
        }
        
        # Initialize LangChain endpoint
        self.langchain = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=self.api_key,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            timeout=self.timeout,
            model_kwargs={
                **self.model_kwargs,
                "retry_on_error": True,
                "max_retries": self.max_retries
            }
        )
        
        self.logger.info(f"Initialized HuggingFace provider for {model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        reraise=True
    )
    def infer(self, prompt: str) -> str:
        """Generate text using Inference API with retry logic."""
        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=self.langchain.max_new_tokens,
                temperature=self.langchain.temperature,
                top_p=self.langchain.top_p
            )
            return response
        except HfHubHTTPError as e:
            if "Model too busy" in str(e):
                self.logger.warning(f"Model busy, retrying: {str(e)}")
                raise  # Trigger retry
            if "does not exist" in str(e):
                self.logger.error(f"Model {self.model_name} not found")
                raise ValueError(f"Model {self.model_name} not found") from e
            self.logger.error(f"HuggingFace inference failed: {str(e)}")
            raise EndpointError(f"HuggingFace inference failed: {str(e)}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error during inference: {str(e)}")
            raise

    def get_token_count(self, text: str) -> int:
        """Get token count using local tokenizer or estimate."""
        try:
            if self.tokenizer:
                return len(self.tokenizer.encode(text))
            else:
                # Fallback to rough estimation if tokenizer not available
                return len(text.split())
        except Exception as e:
            self.logger.error(f"Token counting failed: {str(e)}")
            # Fallback to rough estimation
            return len(text.split())

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for given texts using Inference API."""
        try:
            # Use sentence-transformers model for consistent embeddings
            embeddings = self.client.feature_extraction(
                texts,
                model=self.embedding_model
            )
            
            # Convert to numpy, take mean of token embeddings if needed
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            
            # If we got token-level embeddings, take mean
            if len(embeddings.shape) > 2:
                embeddings = np.mean(embeddings, axis=1)
            
            # Ensure we have 2D array of shape (n_texts, embedding_dim)
            embeddings = embeddings.reshape(len(texts), -1)
            
            # Convert to list format that ChromaDB expects
            return embeddings.tolist()
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise RuntimeError("HuggingFace embedding generation failed") from e

    def get_embeddings_function(self) -> Callable:
        """Get embedding function for ChromaDB."""
        class ChromaEmbeddingFunction:
            def __init__(self, provider):
                self.provider = provider

            def __call__(self, input: List[str]) -> List[List[float]]:
                """ChromaDB-compatible embedding function."""
                try:
                    return self.provider.get_embeddings(input)
                except Exception as e:
                    self.provider.logger.error(f"Embedding failed: {str(e)}")
                    raise RuntimeError("Embedding generation failed") from e

        return ChromaEmbeddingFunction(self)

