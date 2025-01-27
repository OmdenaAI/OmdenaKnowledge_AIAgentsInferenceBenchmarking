from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from simple_agent_common.data_classes import CropDataset
from .base_agent import BaseAgentConfig
import logging
import json
from pathlib import Path
import re

class DataPreparationAgent:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):        
        # Then create base config
        self.base_config = BaseAgentConfig(config, logger)

        # Store config and logger first
        self.config = config
        self.logger = self.base_config.logger
        
        # Now validate paths (after self.config is set)
        self._validate_data_paths()
        
        # Initialize dataset as None
        self.dataset = None

    def _validate_data_paths(self) -> None:
        """Validate that required data files exist"""
        crop_data_path = Path(self.base_config.data_paths['crop_data'])
        
        if not crop_data_path.is_file():
            raise FileNotFoundError(
                f"Crop data file not found at: {crop_data_path.absolute()}\n"
                f"Current working directory: {Path.cwd()}"
            )

    def prepare_data(self, data_path: Path) -> Dict[str, Any]:
        """Prepare and validate the dataset"""
        try:
            # Load data with proper path resolution
            self.logger.info(f"Loading crop data from: {data_path.absolute()}")
            
            try:
                df = pd.read_csv(data_path)
            except pd.errors.EmptyDataError:
                raise ValueError(f"Crop data file is empty: {data_path}")
            except pd.errors.ParserError:
                raise ValueError(f"Failed to parse CSV file: {data_path}")
            
            # Validate data has required columns
            required_columns = [
                'Crop',
                'Precipitation (mm day-1)',
                'Specific Humidity at 2 Meters (g/kg)',
                'Relative Humidity at 2 Meters (%)',
                'Temperature at 2 Meters (C)',
                'Yield'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Basic cleaning
            df = self._clean_data(df)
            
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
            
            self.dataset =  CropDataset(
                df=df,
                summary=summary,
                crops=df['Crop'].unique().tolist(),
                data_path=str(data_path)
            )

            return self.dataset
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        # Remove missing values
        df = df.dropna()
        
        # Validate numeric columns
        numeric_cols = [
            'Precipitation (mm day-1)',
            'Specific Humidity at 2 Meters (g/kg)',
            'Relative Humidity at 2 Meters (%)',
            'Temperature at 2 Meters (C)',
            'Yield'
        ]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        # Validate ranges
        df = df[
            (df['Precipitation (mm day-1)'] >= 0) &
            (df['Specific Humidity at 2 Meters (g/kg)'] >= 0) &
            (df['Relative Humidity at 2 Meters (%)'].between(0, 100)) &
            (df['Temperature at 2 Meters (C)'].between(-50, 60)) &
            (df['Yield'] > 0)
        ]
        
        return df


    def load_questions(self) -> List[Dict[str, Any]]:
        """Load and validate test questions"""
        try:
            validated_questions = []
            with open(self.base_config.data_paths['questions']) as f:
                for line in f:
                    question = json.loads(line)
                    # Validate question format and extract features
                    is_valid, extracted = self._validate_question(question)
                    if is_valid and extracted:
                        validated_questions.append({
                            'original': question,
                            'features': extracted['features'],
                            'actual_yield': extracted['actual_yield'],
                            'completion': question['completion']  # Preserve completion
                        })
                    else:
                        self.logger.warning(f"Skipping invalid question: {question}")
            
            if not validated_questions:
                raise ValueError("No valid questions found")
            
            self.logger.info(f"Loaded {len(validated_questions)} valid questions")
            return validated_questions
            
        except Exception as e:
            self.logger.error(f"Failed to load questions: {str(e)}")
            raise

    def _validate_question(self, question: Dict[str, Any]) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Validate question format and extract features if valid"""
        try:
            # Check if prompt and completion exist
            if 'prompt' not in question or 'completion' not in question:
                self.logger.warning("Missing prompt or completion in question")
                return False, None
            
            try:
                # Extract features using our single regex pattern
                features = self._extract_features(question['prompt'])
                
                # Extract yield from completion
                yield_match = re.search(r'Yield is (\d+)', question['completion'])
                if not yield_match:
                    raise ValueError("Invalid completion format")
                actual_yield = float(yield_match.group(1))
                
                # Add yield to features for historical context
                features['Yield'] = actual_yield
                
                return True, {
                    'features': features,
                    'actual_yield': actual_yield
                }
                
            except ValueError as e:
                self.logger.warning(f"Invalid numeric values in question: {e}")
                return False, None
            
        except Exception as e:
            self.logger.warning(f"Question validation failed: {str(e)}")
            return False, None

    def _extract_features(self, prompt: str) -> Dict[str, Any]:
        """Extract features from prompt text using a single regex pattern"""
        try:
            # Single pattern to extract all features
            pattern = r"precipitation of ([\d.]+).*humidity of ([\d.]+).*humidity of ([\d.]+)%.*temperature of ([\d.]+)°C.*crop ([^.]+)\."
            match = re.search(pattern, prompt)
            if not match:
                raise ValueError(f"Could not extract features from prompt: {prompt}")
            
            return {
                'precipitation': float(match.group(1)),
                'specific_humidity': float(match.group(2)),
                'relative_humidity': float(match.group(3)),
                'temperature': float(match.group(4)),
                'crop': match.group(5).strip()
            }
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise 