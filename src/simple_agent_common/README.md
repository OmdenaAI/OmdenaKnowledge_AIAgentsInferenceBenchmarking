# Simple Agent Common

A Python package providing common utilities and data classes for AI agent benchmarking and metrics collection.

## Overview

This package provides reusable components for:
- Metrics collection for LLM-based agents
- Standardized data structures for crop yield predictions
- Configuration management for agent and task metrics

## Installation

Install in development mode:
bash
pip install -e .

Or in another project's environment.yaml:

## Components

### Data Classes

#### Metrics
- `LLMMetrics`: Metrics for LLM interactions
- `PredictionMetrics`: Metrics for prediction tasks
- `BaseMetrics`: Base class for all metrics
- `BenchmarkMetrics`: Metrics for benchmarking
- `IterationMetrics`: Metrics for iterative processes

#### Configurations
- `AgentMetricsConfig`: Configuration for agent metrics collection
- `TaskMetricsConfig`: Configuration for task-specific metrics

#### Crop Data
- `CropDataset`: Data structure for crop datasets
- `CropPrediction`: Data structure for crop yield predictions

## Usage

from simple_agent_common.data_classes import (
AgentMetricsConfig,
CropDataset,
CropPrediction
)
Initialize metrics configuration
metrics_config = AgentMetricsConfig()
Work with crop data
dataset = CropDataset(...)
prediction = CropPrediction(...)
