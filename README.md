# OmdenaKnowledge_AIAgentsInferenceBenchmarking

## Overview
This project demonstrates benchmarking of AI agents for date fruit classification using LLaMA 3.1-8B-instant model via the Groq API. The implementation includes comprehensive data analysis, feature extraction, model evaluation, and performance benchmarking within a Jupyter Notebook environment.

## Frameworks and Libraries
- **AI and ML Libraries**:
  - `groq`: API connection to LLaMA 3.1-8B-instant model
  - `langgraph`: For building agent workflow graphs
  - `sklearn`: For data preprocessing, scaling, and evaluation metrics

- **Data Processing**:
  - `pandas`: For dataset manipulation and analysis
  - `numpy`: For numerical operations
  - `matplotlib` & `seaborn`: For data visualization and benchmark reporting

- **Utilities**:
  - `dotenv`: For secure API key management
  - `re`: For text processing with regular expressions
  - `json`: For benchmark data storage
  - `datetime`: For timestamping benchmark results

## Dataset
The project uses the Date Fruit Dataset (`Date_Fruit_Datasets.xlsx`) containing features such as:
- Area, Perimeter, Major/Minor Axis measurements
- Eccentricity, Solidity, Convex Area
- Texture and color features
- Classification labels (BERHI, DEGLET, DOKOL, etc.)

## Key Components

### 1. DateFruitAgent
A sophisticated agent that processes and analyzes date fruit features:
- Feature preprocessing and scaling
- Analysis of fruit characteristics
- Classification into fruit categories
- Comprehensive reporting

### 2. Benchmarking System
Metrics tracked and visualized:
- Latency (processing time)
- Model response analysis
- Classification accuracy
- Feature importance

### 3. Visualization & Reporting
- Performance charts and metrics visualizations
- Benchmark summaries
- Classification distribution reports
- Feature analysis documentation

## Benchmark Results
The benchmarking shows:
- Average analysis time: ~2-3 seconds for feature analysis
- Classification latency: ~4-5 seconds per sample
- Varying performance based on sample complexity
- Model accuracy evaluation against ground truth

## Output Files
- Benchmark JSON files in `reports/benchmark/`
- Visualization charts in `reports/charts/`
- Comprehensive analysis reports in `reports/`
