# AutoGen Multi-Agent Benchmarking System

A benchmarking system for evaluating multi-agent LLM performance using AutoGen framework integration.

## Overview

This system implements a multi-agent architecture using AutoGen to evaluate domain-specific problem-solving capabilities. It measures performance, resource usage, and accuracy across multiple iterations.

### Key Components

- **Assistant Agents**:
  - Math Assistant: Specialized in mathematical computations
  - Physics Assistant: Focused on physics problem-solving
  - Chemistry Assistant: Handles chemistry calculations
  - Biology Assistant: Processes biological queries
- **User Proxy Agent**: Manages task delegation and result validation

### Architecture

- **Agent Manager**: Coordinates agent interactions and conversation flow
- **Dataset Handler**: Manages question distribution from JSONL files
- **Metrics Collector**: Captures performance and accuracy data

## Dependencies from simple_agent_common

This system leverages several core components from the simple_agent_common library:

### Data Classes
- `BenchmarkMetrics`: Handles collection and analysis of:
  - Accuracy measurements (MAE, MAPE, RMSE)
  - Performance statistics
  - Resource utilization
  
- `IterationMetrics`: Tracks per-iteration:
  - Question processing times
  - Agent performance
  - Success rates
  
- `PredictionMetrics`: Measures:
  - Answer accuracy
  - Confidence scores
  - Error rates

### Utilities
- `RateLimiter`: Manages API request pacing to prevent rate limiting
- `TokenCounter`: Monitors token usage across all agents
- `MemoryManager`: Tracks memory consumption during execution
- `load_config`: Handles YAML configuration parsing
- `setup_logging`: Configures structured logging for the system

These components provide the foundational infrastructure for benchmarking, allowing the AutoGen agents to focus on their specialized problem-solving capabilities.

## Configuration (config.yaml)
yaml
data:
paths:
input_dir: Location of input JSONL files
env: Environment variables path
metrics: Output metrics directory
benchmark:
iterations: Number of benchmark iterations
total_questions: Questions per iteration
model:
name: LLM model identifier
temperature: Model temperature setting
max_tokens: Maximum token limit


## Data Files

The system processes domain-specific questions from:
- `biology_questions.jsonl`
- `chemistry_questions.jsonl`
- `math_questions.jsonl`
- `physics_questions.jsonl`

Each file contains numerical questions with precise answers that can be evaluated for accuracy. Questions were generated using domain-specific prompts and validated by subject matter experts to ensure numerical answers that fit within Python float datatypes.

## Outputs

The system generates:

1. **Metrics JSON Files**:
   - Format: `{model_name}_autogen_multi_agent_{timestamp}_{iteration}.json`
   - Contents:
     - Accuracy metrics (MAE, MAPE, RMSE)
     - Performance metrics (latency, token usage)
     - Resource utilization
     - Agent interaction statistics

2. **Conversation Logs**:
   - Records agent interactions
   - Question processing flow
   - Solution attempts and validations

3. **Performance Analysis**:
   - Agent-specific performance metrics
   - Token consumption
   - Response latency
   - Memory usage

## Usage

1. Set up configuration in `config.yaml`
2. Place question JSONL files in the data directory
3. Execute `main.py` to run benchmarks
4. Access results in the metrics directory

## Metrics Collection

The system monitors:
- Solution accuracy (MAE, MAPE, RMSE)
- API usage statistics
- Token consumption per agent
- Memory utilization
- Processing time
- Inter-agent communication patterns