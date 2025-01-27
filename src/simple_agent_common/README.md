# Simple Agent Common

A Python package providing common utilities and data classes for AI agent benchmarking and metrics collection.

## Overview

This package provides reusable components for:
- Metrics collection for LLM-based agents
- Standardized data structures for crop yield predictions
- Configuration management for agent and task metrics
- Utility functions for logging, memory management, and rate limiting

## Installation

Install in development mode:
```bash
pip install -e .
```

Or in another project's environment.yaml:
```yaml
dependencies:
  - pip:
      - "-e ../simple_agent_common/."
```

## Components

### Data Classes

#### Metrics
- `LLMMetrics`: Collects and tracks LLM interaction metrics (tokens, latency, costs)
- `PredictionMetrics`: Tracks accuracy and performance of prediction tasks
- `BaseMetrics`: Common metrics functionality for all metric types
- `BenchmarkMetrics`: Performance benchmarking across different runs
- `IterationMetrics`: Tracks metrics across multiple iterations

#### Configurations
- `AgentMetricsConfig`: Settings for agent metrics collection and thresholds
- `TaskMetricsConfig`: Task-specific settings and performance thresholds

#### Crop Data
- `CropDataset`: Standardized structure for crop-related data
- `CropPrediction`: Output format for crop yield predictions

### Utils

- `config.py`: Load and manage YAML configuration files
- `env.py`: Handle environment variables and defaults
- `logging.py`: Structured logging setup and configuration
- `memory.py`: Track and manage memory usage
- `rate_limitier.py`: Control API request rates

## Usage

```python
from simple_agent_common.data_classes import (
    AgentMetricsConfig,
    CropDataset,
    CropPrediction
)

from simple_agent_common.utils import (
    load_config,
    setup_logging,
    track_memory,
    rate_limit,
    load_env
)
```

## Project Structure
