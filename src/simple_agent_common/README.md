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

### MultiAgent

The `multiagent` package provides a framework for benchmarking and orchestrating multiple AI agents:

#### Core Components
- `BenchmarkRunner`: Orchestrates benchmark execution, collecting performance metrics and managing resources
- `OrchestratorBase`: Abstract base class for implementing agent coordination strategies
- `prompts`: Centralized prompt management and templating for agent interactions

Key features:
- Standardized benchmarking across different agent implementations
- Memory usage tracking and performance monitoring
- Configurable iteration management
- Extensible orchestration patterns
- Centralized prompt management

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
- `token_counter.py`: Count tokens for text using tiktoken encoders

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
    load_env_vars,
    TokenCounter
)

# MultiAgent Usage
from simple_agent_common.multiagent import (
    BenchmarkRunner,
    OrchestratorBase
)

# Create a custom orchestrator
class CustomOrchestrator(OrchestratorBase):
    def run(self, input_text: str) -> dict:
        # Implement agent coordination logic
        pass

# Run benchmarks
config = load_config("config.yaml")
orchestrator = CustomOrchestrator()
runner = BenchmarkRunner("custom_benchmark", orchestrator, config)
runner.run()
```

## Project Structure
