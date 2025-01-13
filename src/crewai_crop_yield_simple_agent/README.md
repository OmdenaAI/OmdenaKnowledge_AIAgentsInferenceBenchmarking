# Crop Yield Prediction Benchmark

A benchmarking framework for evaluating Large Language Model (LLM) performance on agricultural yield predictions using CrewAI architecture.

## System Architecture

The system uses a modular architecture with three main components:

1. **Agents**
   - `PredictionAgent`: Handles yield predictions using LLM
   - `DataPreparationAgent`: Manages dataset preprocessing
   - `TokenCounter`: Protocol for token counting implementation

2. **Tasks**
   - `PredictionTask`: Orchestrates prediction workflow
   - `DataPreparationTask`: Handles data cleaning and preparation
   - `QuestionLoadingTask`: Manages few-shot example selection

3. **Data Classes**
   - `CropDataset`: Encapsulates crop yield data and statistics
   - `CropPrediction`: Represents individual predictions
   - `Metrics`: Various metric tracking classes (LLM, Prediction, Benchmark)

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
![JSON Metrics](docs/json/example.json)

