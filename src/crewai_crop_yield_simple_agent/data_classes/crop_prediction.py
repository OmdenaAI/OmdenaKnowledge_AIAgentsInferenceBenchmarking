from pydantic import BaseModel, Field
from typing import Optional, Dict, Union
from pydantic import ConfigDict

class CropPrediction(BaseModel):
    """Single crop yield prediction"""
    predicted_yield: float = Field(..., description="Predicted crop yield")
    actual_yield: Optional[float] = Field(None, description="Actual crop yield")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    features: Dict[str, Union[float, str]] = Field(..., description="Input features")
    question: str = Field(default="", description="Original question")
    context: str = Field(default="", description="Prediction context")
    model_config = ConfigDict(arbitrary_types_allowed=True) 