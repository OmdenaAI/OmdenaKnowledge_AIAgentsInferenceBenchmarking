from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any
from pathlib import Path
from .metrics import LLMMetrics, PredictionMetrics

class PathConfig(BaseModel):
    """Configuration for file paths"""
    crop_data: Path
    questions: Path
    env: Path
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ModelConfig(BaseModel):
    """Configuration for LLM model"""
    name: str
    temperature: float
    max_tokens: int

class AppConfig(BaseModel):
    """Application configuration"""
    data: Dict[str, Dict[str, str]]
    model: ModelConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

class AgentMetricsConfig(BaseModel):
    """Configuration for Agent metrics"""
    llm_metrics: LLMMetrics = Field(default_factory=LLMMetrics)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TaskMetricsConfig(BaseModel):
    """Configuration for Task metrics"""
    prediction_metrics: PredictionMetrics = Field(default_factory=PredictionMetrics)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TaskConfig(BaseModel):
    """Base configuration for tasks"""
    data: Dict[str, Any] = Field(default_factory=dict)
    model: str = Field(default="default")
    config: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True) 