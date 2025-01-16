# Crop Yield Prediction Benchmark

A benchmarking framework for evaluating Large Language Model (LLM) performance on agricultural yield predictions using Autogen architecture.

## System Architecture

The system uses a modular architecture with several key components:

### Core Components

1. **Agents**
   - `PredictionAgent`: Manages LLM interactions for yield predictions
     - Uses AutoGen's AssistantAgent for predictions
     - Handles context building with few-shot examples
     - Implements retry logic and error handling
   - `DataPreparationAgent`: Handles dataset management
     - Validates and cleans input data
     - Extracts features from questions
     - Manages data statistics and summaries
   - `TokenCounter`: Tracks token usage
     - Uses LLaMA tokenizer for accurate counting
     - Monitors prompt and completion tokens

2. **Utilities**
   - `RateLimiter`: Controls API call frequency
   - `MetricsHandler`: Manages performance tracking
   - `ConfigLoader`: Handles configuration validation
   - `RetryDecorator`: Implements exponential backoff
   - `LoggingSetup`: Configures application logging

### Data Flow

1. **Initialization**
   - Load configuration from YAML
   - Setup logging and environment
   - Initialize agents and utilities

2. **Benchmark Process**
   - Data preparation phase
     - Load and validate crop dataset
     - Process test questions
   - Iteration phase
     - Select few-shot examples (similarity/random)
     - Make predictions via AutoGen
     - Track performance metrics
   - Metrics collection
     - Calculate error metrics (MAE, MAPE, RMSE)
     - Monitor resource usage
     - Generate visualizations

### Performance Monitoring

1. **Prediction Metrics**
   - Error calculations per iteration
   - Aggregate statistics across runs

2. **System Metrics**
   - Memory usage tracking
   - API latency monitoring
   - Token consumption analysis

### Output Generation

1. **Metrics Output**
   - Detailed JSON reports
   - Performance visualizations
   - Runtime analysis graphs

2. **Logging**
   - Console and file logging
   - Debug information
   - Error tracking

## Configuration (config.yaml)
yaml
data:
paths:
crop_data: "data/crop+yield+predictiondata_crop_yield.csv" # Source dataset
questions: "data/crop_yield_questions_10.jsonl" # Test questions
env: "~/src/python/.env" # Environment variables
metrics: "output/metrics" # Output directory
model:
name: "llama-3.1-70b-versatile" # LLM model identifier
temperature: 0.01 # Randomness control (lower = more deterministic)
max_tokens: 1000 # Maximum response length
benchmark:
iterations: 3 # Number of test iterations
random_few_shot: false # Use random vs similarity-based examples
num_few_shot: 5 # Number of examples per prediction

## Performance Metrics

### Prediction Performance (example_performance.png)
![Performance Metrics](docs/images/example_performance.png)

Tracks three key error metrics across iterations:
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual yields
- **MAPE (Mean Absolute Percentage Error)**: Percentage error relative to actual yields
- **RMSE (Root Mean Square Error)**: Square root of average squared errors

### Runtime Performance (example_runtime.png)
![Runtime Metrics](docs/images/example_runtime.png)

Monitors system performance:
- **API Latency**: LLM response time per call
- **Total Runtime**: Complete iteration processing time
- **Memory Delta**: Memory usage changes between iterations
- **Tokens per Call**: Token consumption rate

## Metrics Output

The system generates detailed JSON metrics:
```json
{
  "model_info": {
    "name": "llama-3.1-70b-versatile",
    "temperature": 0.01,
    "max_tokens": 1000,
    "num_few_shot": 5,
    "random_few_shot": false
  },
  "performance_metrics": {
    "mae": 1660.5,
    "mape": 7.39446804003398,
    "rmse": 1971.494331718963
  },
  "runtime_metrics": {
    "total_runtime": 100.52886867523193,
    "avg_runtime": 33.50962289174398,
    "total_llm_calls": 30,
    "avg_latency": 0.10822827816009521
  },
  "token_metrics": {
    "total_prompt_tokens": 17007,
    "total_response_tokens": 0,
    "total_tokens": 17007,
    "avg_tokens_per_call": 5669.0
  },
  "memory_metrics": {
    "peak_memory_mb": 277.671875,
    "avg_memory_delta_mb": 1.4322916666666667
  },
  "iterations": [
    {
      "iteration": 1,
      "runtime": 21.794018030166626,
      "memory_delta": 4.203125,
      "peak_memory": 277.578125,
      "llm_calls": 10,
      "avg_latency": 0.2989490509033203,
      "prompt_tokens": 5669,
      "response_tokens": 0,
      "total_tokens": 5669,
      "mae": 1660.5,
      "mape": 7.39446804003398,
      "rmse": 1971.494331718963
    },
    {
      "iteration": 2,
      "runtime": 39.15412187576294,
      "memory_delta": 0.046875,
      "peak_memory": 277.625,
      "llm_calls": 10,
      "avg_latency": 0.012729954719543458,
      "prompt_tokens": 5669,
      "response_tokens": 0,
      "total_tokens": 5669,
      "mae": 1660.5,
      "mape": 7.39446804003398,
      "rmse": 1971.494331718963
    },
    {
      "iteration": 3,
      "runtime": 39.58072876930237,
      "memory_delta": 0.046875,
      "peak_memory": 277.671875,
      "llm_calls": 10,
      "avg_latency": 0.013005828857421875,
      "prompt_tokens": 5669,
      "response_tokens": 0,
      "total_tokens": 5669,
      "mae": 1660.5,
      "mape": 7.39446804003398,
      "rmse": 1971.494331718963
    }
  ],
  "timestamp": "2025-01-16T09:46:59.840475",
  "framework": "autogen"
}