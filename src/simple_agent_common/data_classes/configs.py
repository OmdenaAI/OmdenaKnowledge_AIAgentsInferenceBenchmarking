from pydantic import BaseModel, Field, ConfigDict
from .metrics import LLMMetrics, PredictionMetrics

class AgentMetricsConfig(BaseModel):
    """Configuration for Agent metrics"""
    llm_metrics: LLMMetrics = Field(default_factory=LLMMetrics)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TaskMetricsConfig(BaseModel):
    """Configuration for Task metrics"""
    prediction_metrics: PredictionMetrics = Field(default_factory=PredictionMetrics)
    model_config = ConfigDict(arbitrary_types_allowed=True)
