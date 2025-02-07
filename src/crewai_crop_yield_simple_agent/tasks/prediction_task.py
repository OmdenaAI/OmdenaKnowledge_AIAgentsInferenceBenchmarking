from crewai import Task
from simple_agent_common.data_classes import TaskMetricsConfig, CropPrediction
from typing import Optional, List, Dict, Any, Annotated, Tuple
from pydantic import Field, ConfigDict
from .data_preparation_task import DataPreparationTask
from .question_loading_task import QuestionLoadingTask
from simple_agent_common.utils import RateLimiter, MemoryManager
import logging


class PredictionTask(Task):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        validate_assignment=False,
        validate_default=False,
        frozen=False,
        from_attributes=True,
        revalidate_instances='never'
    )
    
    data: Dict = Field(default_factory=dict)
    metrics: TaskMetricsConfig = Field(default_factory=TaskMetricsConfig)
    predictions: List[CropPrediction] = Field(default_factory=list)
    config: Optional[Dict[str, Any]] = Field(None, exclude=True)
    logger: Optional[logging.Logger] = Field(None, exclude=True)
    memory_manager: Optional[MemoryManager] = Field(None, exclude=True)
    benchmark: Dict[str, Any] = Field(default_factory=dict)
    agent: Optional[Any] = Field(None, exclude=True)
    prep_task: Optional[DataPreparationTask] = Field(None)
    questions_task: Optional[QuestionLoadingTask] = Field(None)

    def __init__(self, agent: Any, prep_task: DataPreparationTask, questions_task: QuestionLoadingTask, 
                 config: Dict[str, Any], logger: logging.Logger, memory_manager: MemoryManager, **kwargs):
        super().__init__(
            description="Generate yield predictions using LLM",
            expected_output="Yield predictions with accuracy metrics",
            agent=agent,
            context=[],
            tools=[],
        )
        
        self.model_post_init({
            "agent": agent,
            "config": config,
            "logger": logger,
            "memory_manager": memory_manager,
            "prep_task": prep_task,
            "questions_task": questions_task
        })
        
        self.logger = logger
        self.memory_manager = memory_manager
        self.config = config
        self.agent = agent
        self.prep_task = prep_task
        self.questions_task = questions_task

    def execute(self, context: Optional[Dict[str, Any]] = None) -> str:
        if not self.prep_task.dataset:
            raise ValueError("No dataset available from preparation task")
            
        if not self.questions_task.questions:
            raise ValueError("No questions available from questions task")
        
        rate_limiter = RateLimiter(max_calls=4, pause_time=20)

        for prompt, completion in self.questions_task.questions:
            
            # Track memory around prediction
            memory_stats = self.memory_manager.get_memory_stats()

            with rate_limiter:
                prediction = self.agent.predict_yield(
                    prompt=prompt,
                    completion=completion,
                    dataset=self.prep_task.dataset
                )
            self.predictions.append(prediction)

            self.metrics.prediction_metrics.predictions.append(
                (prediction.predicted_yield, prediction.actual_yield, None)
            )
            
            # Track memory around prediction
            memory_stats = self.memory_manager.get_memory_stats()
            
        self.metrics.prediction_metrics.calculate_metrics()
        return f"Made {len(self.predictions)} yield predictions" 