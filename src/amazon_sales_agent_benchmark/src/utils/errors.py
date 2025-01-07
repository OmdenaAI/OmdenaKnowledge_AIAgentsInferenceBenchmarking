class HuggingFaceError(Exception):
    """Base exception for HuggingFace-related errors."""
    pass

class EndpointError(HuggingFaceError):
    """Errors related to HuggingFace Inference Endpoints."""
    pass

class RateLimitError(EndpointError):
    """Rate limiting errors from HuggingFace API."""
    pass 