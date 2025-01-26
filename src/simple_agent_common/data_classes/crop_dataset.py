from pydantic import BaseModel, Field, ConfigDict
import pandas as pd

class CropDataset(BaseModel):
    """Represents cleaned crop yield dataset"""
    df: pd.DataFrame = Field(..., description="Pandas DataFrame with crop data")
    summary: dict = Field(..., description="Dataset summary statistics")
    crops: list[str] = Field(..., description="List of unique crops")
    data_path: str = Field(..., description="Path to source data")
    
    model_config = ConfigDict(arbitrary_types_allowed=True) 