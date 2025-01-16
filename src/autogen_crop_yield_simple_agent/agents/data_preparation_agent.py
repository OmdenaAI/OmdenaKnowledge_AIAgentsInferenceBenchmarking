from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import autogen
from .base_agent import BaseAgentConfig
import random
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
            
            # Calculate summary statistics
            summary = {
                'crop_distribution': self._get_crop_stats(df),
                'feature_stats': self._get_feature_stats(df)
            }
            
            self.dataset = {
                'df': df,
                'summary': summary,
                'crops': df['Crop'].unique().tolist()
            }
            
            self.logger.info(f"Prepared dataset with {len(df)} records across {len(self.dataset['crops'])} crops")
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

    def _get_crop_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate crop-specific statistics"""
        stats = {}
        for crop in df['Crop'].unique():
            crop_data = df[df['Crop'] == crop]
            stats[crop] = {
                'count': len(crop_data),
                'yield_stats': {
                    'min': float(crop_data['Yield'].min()),
                    'max': float(crop_data['Yield'].max()),
                    'mean': float(crop_data['Yield'].mean()),
                    'std': float(crop_data['Yield'].std())
                }
            }

        return stats

    def _get_feature_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate feature statistics"""
        feature_cols = [
            'Precipitation (mm day-1)',
            'Specific Humidity at 2 Meters (g/kg)',
            'Relative Humidity at 2 Meters (%)',
            'Temperature at 2 Meters (C)'
        ]
        
        stats = {}
        for col in feature_cols:
            stats[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        return stats

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