# LangGraph Multi-Agent Benchmarking System

A benchmarking system for evaluating multi-agent LLM performance using LangGraph and Groq integration.

## Overview

This system implements a multi-agent architecture to route and solve domain-specific questions using specialized agents. It measures performance, resource usage, and accuracy across multiple iterations.

### Key Components

- **Router**: Routes questions to appropriate domain experts using LLM classification
- **Specialized Agents**:
  - Math Agent: Handles mathematical computations
  - Physics Agent: Solves physics problems
  - Chemistry Agent: Processes chemistry queries
  - Biology Agent: Manages biological calculations

### Architecture

- **Orchestrator**: Manages agent workflow and state transitions using LangGraph
- **Dataset Handler**: Loads and randomizes questions from JSONL files
- **Metrics Collection**: Tracks performance, latency, and accuracy metrics

## Dependencies from simple_agent_common

- **Data Classes**:
  - `BenchmarkMetrics`: Core metrics collection and analysis
  - `IterationMetrics`: Per-iteration metrics tracking
  - `PredictionMetrics`: Prediction accuracy measurements

- **Utilities**:
  - `RateLimiter`: Controls API call frequency
  - `TokenCounter`: Tracks token usage
  - `MemoryManager`: Monitors memory usage
  - `load_config`: Configuration management
  - `setup_logging`: Logging setup

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

The system uses the following data files:

- `biology_questions.jsonl`: Biology questions
- `chemistry_questions.jsonl`: Chemistry questions
- `math_questions.jsonl`: Math questions
- `physics_questions.jsonl`: Physics questions

The questions were created using the following prompt:

Act as an Associate {discipline} Processor. Generate 25 {discipline} questions with answers in a jsonl file. Answers must be purely numerical and not be in scientific notation. These questions must specifically relate to {discipline} and not other scientific disciplines. Act as a {discipline} Professor Emeritus and evaluate all generated questions and answers to ensure their accuracy and that the output is numerical and fit in a Python float datatype. In other words do not include questions whose answers are in scientific notation and would overflow a python decimal datatype.Once confirmed output the {discipline} dataset in jsonl format

All the answers are numerical values that can be easily meansured for correctness. From config.yaml we get total_questions and use that to randomly select questions equally dividedfrom each of the four files.

## Outputs

The system generates several outputs in the metrics directory:

1. **Metrics JSON Files**:
   - Filename format: `{model_name}_langgraph_multi_agent_{timestamp}_{iteration}.json`
   - Contains:
     - Accuracy metrics (MAE, MAPE, RMSE)
     - Performance metrics (latency, token usage)
     - Resource usage (memory consumption)
     - Agent distribution statistics

2. **Questions Log**:
   - Records questions processed in each iteration
   - Includes agent assignments and predictions

3. **Performance Analysis**:
   - Per-agent performance metrics
   - Token usage statistics
   - Latency measurements
   - Memory utilization

## Usage

1. Configure `config.yaml` with desired settings
2. Place input JSONL files in the data directory
3. Run `main.py` to execute benchmarks
4. Results are saved in the specified metrics directory

## Metrics Collection

The system tracks:
- Prediction accuracy (MAE, MAPE, RMSE)
- API call statistics
- Token usage per agent
- Memory consumption
- Processing latency
- Agent routing distribution

