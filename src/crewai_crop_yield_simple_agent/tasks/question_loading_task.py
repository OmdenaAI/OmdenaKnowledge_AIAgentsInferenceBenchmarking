from crewai import Task
from simple_agent_common.data_classes import TaskMetricsConfig
from typing import Optional, List, Dict, Any, Tuple, Annotated
from pydantic import Field, ConfigDict
import json
import logging

class QuestionLoadingTask(Task):
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
    questions: List[Tuple[str, str]] = Field(default_factory=list)
    config: Optional[Dict[str, Any]] = Field(None, exclude=True)
    logger: Optional[logging.Logger] = Field(None, exclude=True)
    model: str = Field(default="default")
    benchmark: Dict[str, Any] = Field(default_factory=dict)
    agent: Optional[Any] = Field(None, exclude=True)

    def __init__(self, agent: Any, config: Dict[str, Any], logger: logging.Logger, **kwargs):
        # Initialize Task first
        super().__init__(
            description="Load and validate question-answer pairs",
            expected_output="List of validated questions with expected completions",
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

    # TBD the guts of this should move to data_preparation_agent
    def execute(self, context: Optional[Dict[str, Any]] = None) -> str:
        questions_path = self.config['data']['paths']['questions']
        with open(questions_path) as f:
            for line in f:
                data = json.loads(line)
                self.questions.append((data['prompt'], data['completion']))
        return f"Loaded {len(self.questions)} questions" 