from crewai import Agent
from simple_agent_common.data_classes import AgentMetricsConfig, CropDataset
from typing import Any, Dict, Annotated, Optional
import pandas as pd
import logging
from pydantic import Field, ConfigDict

class DataPreparationAgent(Agent):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        validate_assignment=False,
        validate_default=False,
        frozen=False,
        from_attributes=True,
        revalidate_instances='never'
    )
    
    metrics: AgentMetricsConfig = Field(default_factory=AgentMetricsConfig)
    logger: Optional[logging.Logger] = Field(None, exclude=True)
    config: Optional[Dict[str, Any]] = Field(None, exclude=True)
    data: Dict = Field(default_factory=dict)
    model: str = Field(default="default")
    benchmark: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, llm: Any, logger: logging.Logger, config: Dict[str, Any]):
        # Initialize Agent first
        super().__init__(
            llm=llm,
            role="Data Preparation Specialist",
            goal="Prepare and validate crop yield datasets",
            backstory="Expert in agricultural data processing and validation",
            verbose=False,
            tools=[],
            allow_delegation=False
        )
        
        # Then set our fields
        self.model_post_init({
            "logger": logger,
            "config": config
        })

        self.logger = logger
        self.config = config

    def prepare_dataset(self, csv_path: str) -> CropDataset:
        self.logger.info(f"\n📊 Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Add detailed crop distribution analysis
        crop_counts = df['Crop'].value_counts()
        self.logger.info("\n🌾 Crop Distribution:")
        self.logger.info("=" * 50)
        for crop, count in crop_counts.items():
            yield_stats = df[df['Crop'] == crop]['Yield'].agg(['min', 'max', 'mean'])
            self.logger.info(f"- {crop}:")
            self.logger.info(f"  Records: {count}")
            self.logger.info(f"  Yield Range: {yield_stats['min']:.0f} - {yield_stats['max']:.0f}")
            self.logger.info(f"  Average Yield: {yield_stats['mean']:.0f}")
            self.logger.info("-" * 30)
        
        summary = {
            'total_records': len(df),
            'crops': df['Crop'].unique().tolist(),
            'yield_range': f"{df['Yield'].min():.2f}-{df['Yield'].max():.2f}",
            'crop_distribution': {
                crop: {
                    'count': int(count),
                    'yield_stats': df[df['Crop'] == crop]['Yield'].agg(['min', 'max', 'mean', 'std']).to_dict()
                } for crop, count in crop_counts.items()
            }
        }
        self.logger.info(f"\n📈 Dataset Summary:")
        self.logger.info(f"Total Records: {summary['total_records']}")
        self.logger.info(f"Number of Crops: {len(summary['crops'])}")
        self.logger.info(f"Overall Yield Range: {summary['yield_range']}")
        
        return CropDataset(
            df=df,
            summary=summary,
            crops=df['Crop'].unique().tolist(),
            data_path=csv_path
        )
