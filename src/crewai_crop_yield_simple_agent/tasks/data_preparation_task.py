from pydantic import Field, ConfigDict, PrivateAttr
from crewai import Task
from data_classes import TaskMetricsConfig, CropDataset
from typing import Optional, Dict, Any, Annotated
import logging

class DataPreparationTask(Task):
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
    dataset: Optional[CropDataset] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(None, exclude=True)
    logger: Optional[logging.Logger] = Field(None, exclude=True)
    model: str = Field(default="default")
    benchmark: Dict[str, Any] = Field(default_factory=dict)
    agent: Optional[Any] = Field(None, exclude=True)

    def __init__(self, agent: Any, config: Dict[str, Any], logger: logging.Logger, **kwargs):
        # Initialize Task first
        super().__init__(
            description="Prepare and validate crop yield dataset",
            expected_output="Cleaned and validated dataset ready for predictions",
            agent=agent,
            context=[],
            tools=[],
        )
        
        # Then set our fields
        self.model_post_init({
            "agent": agent,
            "config": config,
            "logger": logger
        })
        self.logger = logger
        self.config = config

    def execute(self, context: Optional[Dict[str, Any]] = None) -> str:
        self.logger.info("Starting DataPreparationTask execution...")
        csv_path = self.config['data']['paths']['crop_data']
        self.dataset = self.agent.prepare_dataset(csv_path)
        result = f"Prepared dataset with {len(self.dataset.crops)} crops"
        self.logger.info(result)
        return result


