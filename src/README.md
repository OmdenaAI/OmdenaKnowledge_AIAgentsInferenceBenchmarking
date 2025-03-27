# OmdenaKnowledge_AIAgentsInferenceBenchmarking


## Setting Up the Virtual Environment

To create and activate a virtual environment, follow the steps below based on your operating system.

### Creating a Virtual Environment:
```sh
python -m venv <environment_name>
```

### Activating the Virtual Environment:

#### Windows:
```sh
<environment_name>\Scripts\activate
```

#### macOS:
```sh
source <environment_name>/bin/activate
```

## Installation of Dependencies
Once the virtual environment is activated, install the required dependencies:
```sh
pip install -r requirements.txt
```

## Running the Benchmarking Script
To execute the benchmarking process, run the following command:
#### Windows:
```sh
cd OmdenaKnowledge_AIAgentsInferenceBenchmarking\src>
```
#### macOS:
```sh
cd OmdenaKnowledge_AIAgentsInferenceBenchmarking/src
```
#### For Autogen:
```sh
autogen/python autogen_benchmark.py
```
#### For CrewAI:
```sh
crewai/python crewai_benchmark.py
```
#### For Langgraph:
```sh
langgraph/python langgraph_benchmark.py
```

## Project Overview
This repository is designed to benchmark different AI agents based on the following metrics:
Response Time (Latency)
Token Efficiency (Cost)
Generated Paragraph Quality (Rating)
Throughput (Keywords per second)
Memory Usage (Peak & Delta)
Total Tokens Used

## Folder Structure
```
OmdenaKnowledge_AIAgentsInferenceBenchmarking/
│-- src/                        # Source code directory
│   │-- config/                 #config
│   │   │-- config_loader.py     #cofig loader
│   │   │-- config.yaml          # Configuration file for the benchmarking setup
│   │-- crewai/                  # Folder containing CrewAI-based agents
│   │   │-- agents.py             # Agents
│   │   │-- crewai_benchmark.py   # Main benchmarking script
│   │   │-- crewai_benchmark_results.csv # Benchmark results
│   │   │-- tasks.py              # Task definitions
│   │-- autogen/                  # Folder containing AutoGen-based agents
│   │   │-- agents.py             # Agents
│   │   │-- autogen_benchmark.py  # Main benchmarking script
│   │   │-- autogen_benchmark_results.csv # Benchmark results
│   │-- langgraph/                # Folder containing LangGraph-based agents
│   │   │-- nodes_agents.py       # Agents
│   │   │-- langgraph_benchmark.py # Main benchmarking script
│   │   │-- langgraph_benchmark_results.csv # Benchmark results
│   │-- utils/                      #utils
│   │   │--common_functions.py     # saving to csv, paragraph rating
│   │-- results/                    #results
│   │   │-- results.py             # plot results
│   │-- README.md                 # Project documentation
│   │-- requirements.txt           # List of dependencies
│   │-- .env                       # Environment variables
│   │-- setup.py                   # Package installation & CLI setup
```

## Contributing
If you would like to contribute to this project, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

---

For any queries, feel free to reach out!

